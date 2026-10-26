# Independent inputs in one ESMFold2 batch

`ESMFold2InputBuilder.fold_batch()` pads independent inputs to common token and
atom dimensions, executes **one model forward**, and returns
`results[input_index][sample_index]`. Each input can have a different length
and can itself contain several chains. Diffusion samples are a separate axis.

```python
from pathlib import Path

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)

model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda:0").eval()
builder = ESMFold2InputBuilder()
ids = ["protein_a", "protein_b"]
sequences = ["MQIFVKTLTGKT", "ACDEFGHIKLMNPQRSTVWY"]
inputs = [
    StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
    for sequence in sequences
]

results = builder.fold_batch(
    model,
    inputs,
    complex_ids=ids,
    num_loops=20,
    num_sampling_steps=200,
    num_diffusion_samples=5,
    seed=0,
)

output = Path("predictions")
output.mkdir(exist_ok=True)
for name, samples in zip(ids, results, strict=True):
    for sample_index, result in enumerate(samples):
        (output / f"{name}_sample_{sample_index}.cif").write_text(
            result.complex.to_mmcif()
        )
        print(name, sample_index, result.ptm, result.iptm, result.plddt.shape)
```

The outer list always has one entry per input; the inner list always has one
entry per diffusion sample, including when either count is one. Structures,
per-token confidence, PAE, residue metadata and per-chain confidence are
associated with their own input and cropped to its dimensions. No best sample
is selected implicitly. If selecting a sample, apply the chosen score **within
each input's samples**. For example, selecting by pTM is not a replacement for
BoltzGen's task-specific ipSAE scoring.

## Multiple GPUs

Keep one persistent model per GPU in your existing worker system. Each worker
can now submit a group of independent inputs using `fold_batch()`. For example,
eight GPU workers with B=2 can process up to 16 independent inputs concurrently,
each with its own S diffusion samples. Each worker must receive disjoint input
IDs, and write and validate one complete output set per ID.

Start with B=2 and measure completed inputs/second and peak VRAM on every GPU.
Group inputs by similar token/atom counts to reduce padding; the pair state
scales roughly as `B * L_max**2 * pair_width`, and diffusion/confidence also
scale with `B * S`. Keep loops, diffusion steps, sample count, precision and
backend fixed when comparing throughput. This API does not configure workers
or integrate with the separate BoltzGen checkout automatically.

## Low-level API for full-source ESMC and cropped folding

`collate_features()` accepts a list of feature dictionaries from
`builder.prepare_input()`, each with a leading B=1 dimension. It uses explicit
per-field axis schemas for tokens, atoms and MSA rows; it rejects unrecognized
fields and incompatible feature shapes instead of guessing from their sizes.
Inputs must share feature keys, dtypes and devices. Each per-input token axis
must be unpadded; the atom axis retains the featurizer's trailing alignment padding.

For workflows that encode full source sequences before folding a structural
crop, preserve that order. Prepare and crop each input using its own source
mapping, attach the corresponding cropped LM states, then collate:

```python
import torch

from esm.models.esmfold2 import collate_features, split_outputs

# features_per_input[i]: prepared B=1 structural-crop features.
# cropped_lm_states[i]: matching [1, L_i, num_layers, hidden_size] states,
# computed from the full source chains before selecting the crop's tokens.
features_with_lm = [
    {**features, "lm_hidden_states": states}
    for features, states in zip(features_per_input, cropped_lm_states, strict=True)
]
batch = collate_features(features_with_lm)
with torch.no_grad():
    raw = model(
        **batch,
        num_loops=20,
        num_sampling_steps=200,
        num_diffusion_samples=5,
        lm_mask_pct=0.0,
    )

predictions = split_outputs(raw, features_with_lm, num_diffusion_samples=5)
```

The model's sample axis is flattened in input-major order:
`flat_index = input_index * S + sample_index`. `split_outputs()` returns each
input's S samples, preserving the sample dimension and removing token padding.
By default it restores each input's original atom tensor extent, including its
32-atom alignment padding, so its masks and structure metadata remain usable.
For an archive copy, call it separately with `trim_atom_padding=True` to remove
atom padding too. Never pass trimmed atom coordinates together with padded
structure masks. The returned tensors are views; clone before modifying them.

`builder.decode_batch(raw, features_per_input, chain_infos_per_input,
num_diffusion_samples=5, complex_ids=ids)` converts those same model outputs
into the nested result objects. The existing `decode()` accepts B=1 only and
raises for B>1, preventing accidental reuse of input zero's metadata.

BoltzGen's pinned runtime must explicitly load an ESM build containing this
API. Its scoring worker still needs to group requests, preserve full-source
ESMC/crop mappings, score each input's samples independently and publish its
existing per-ID artifacts. Its acceleration adapters and CUDA graphs require
separate qualification for B>1. Increasing Boltz's `FOLD_BATCH` or ESMFold2's
`workers_per_device` does not call this new API.

## Correctness and reproducibility

Batch padding must be masked through the model and removed from per-input
outputs. Tests cover explicit axis slicing, different sequence and atom counts,
sample ownership, multichain metadata, confidence consistency with fixed
coordinates, and B=1 compatibility. These checks establish tensor and output
contracts; they do not establish biological prediction accuracy on a dataset.

The seed is **batch-wide**. Changing batch size, membership, order or padding
changes random-number consumption in pair-state initialization, LM dropout and
diffusion sampling. Independent singleton runs and a mixed-length batch are
therefore not promised to give byte-identical structures at the same seed.
Serialization of an already-computed prediction must still preserve its exact
input/sample association. Compare deterministic components under controlled
randomness, and use repeated runs and task metrics to assess prediction quality.

The high-level batch API supports the released ESMFold2 architecture; it rejects
the experimental architecture and Fold-CP wrappers. Backends and
downstream graph adapters should be benchmarked and validated separately before
large campaigns. A failed forward raises; callers must account for **all B
inputs** in that failed batch and must not mark them complete.

## Local qualification (2026-10-21)

The batch, existing API and MSA regression suites passed together: 58 passed,
one pre-existing expected failure. They include byte-identical CIF serialization
of the same raw predictions decoded together versus individually. CPU models
use tiny random weights; those tests check contracts, not structure quality.

A full-checkpoint smoke run on one H200 used two independent control proteins
(ubiquitin, 76 residues; lysozyme, 129 residues) with five samples each. It wrote
ten correctly sized structures with finite per-input confidence arrays. The
checkpoint revisions matched BoltzGen's pinned 2021 folding model and ESMC-6B.

A throughput comparison used 20 loops, 200 diffusion steps, five samples,
BF16 ESMC, reference kernels and chunk size 64. FlashAttention was unavailable;
the model used its PyTorch fallback. Weights were already loaded, every shape
was warmed with a short fold, and measurements included input preparation,
ESMC, folding and decoding (but not writing CIF files). Two measurements per
configuration were interleaved:

| Per-GPU input batch size | Mean time for both inputs | Inputs/second | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| 1 (two sequential calls) | 7.19 s | 0.278 | 13.72 GiB |
| 2 (one forward) | 5.88 s | 0.340 | 14.61 GiB |

That is approximately 22% higher throughput on this small control set, not an
eight-GPU measurement or an estimate for 300-token design campaigns.

Mixed-precision arithmetic also matters independently of stochastic sampling.
With **identical trunk states and coordinates**, recomputing confidence per
input versus together differed by up to 0.58 Å in PAE and 0.078 in per-token
pLDDT (0–1 scale) in the short smoke run. Differences remained when padding
was held constant. A separate FP32 confidence-trunk control reduced maximum
differences to 0.000058 Å PAE and 0.0000073 pLDDT, consistent with batch/shape
dependent BF16 arithmetic rather than cross-input mixing. Production precision
was not changed. This does not establish dataset-level accuracy or guarantee
unchanged rankings around hard score thresholds; qualify those on representative
inputs before a large campaign.
