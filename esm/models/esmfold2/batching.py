"""Explicit input/output axis schemas for independent ESMFold2 inputs.

Never infer an axis from its size: the sample, token, atom and feature-channel
counts can coincide. Model outputs flatten samples in design-major order:
``flat_index = design_index * num_diffusion_samples + sample_index``.
"""

from collections.abc import Sequence

from torch import Tensor

# Named axes are padded. None means a fixed feature/conformer dimension.
# Shapes include the leading batch dimension (each input must have B=1).
FEATURE_AXES = {
    **{
        key: ("B", "L")
        for key in (
            "token_index",
            "residue_index",
            "asym_id",
            "sym_id",
            "entity_id",
            "mol_type",
            "res_type",
            "input_ids",
            "token_attention_mask",
            "pocket_feature",
            "distogram_atom_idx",
            "deletion_mean",
        )
    },
    **{
        key: ("B", "A")
        for key in (
            "ref_element",
            "ref_charge",
            "ref_space_uid",
            "atom_attention_mask",
            "atom_to_token",
            "is_resolved",
        )
    },
    "ref_pos": ("B", "A", None),
    "ref_atom_name_chars": ("B", "A", None),
    "gt_coords": ("B", None, "A", None),
    "frames_idx": ("B", "L", None),
    "token_bonds": ("B", "L", "L", None),
    "disto_cond": ("B", "L", "L"),
    "disto_cond_mask": ("B", "L", "L"),
    **{
        key: ("B", "M", "L")
        for key in ("msa", "has_deletion", "deletion_value", "msa_attention_mask")
    },
    # Allows callers to encode full source chains, crop their LM states, then
    # batch only the structural crops (e.g. the BoltzGen scoring adapter).
    "lm_hidden_states": ("B", "L", None, None),
}

# S is the flattened B * num_diffusion_samples axis, B is per-design.
# C is a chain-ID axis; chain IDs remain local to each design.
OUTPUT_AXES = {
    "sample_atom_coords": ("S", "A", None),
    "distogram_logits": ("B", "L", "L", None),
    "plddt_logits": ("S", "A", None),
    "plddt": ("S", "L"),
    "plddt_per_atom": ("S", "A"),
    "plddt_ca": ("S", "L"),
    "complex_plddt": ("S",),
    "complex_iplddt": ("S",),
    "pae_logits": ("S", "L", "L", None),
    "pae": ("S", "L", "L"),
    "pde_logits": ("S", "L", "L", None),
    "pde": ("S", "L", "L"),
    "resolved_logits": ("S", "A", None),
    "ptm": ("S",),
    "iptm": ("S",),
    "pair_chains_iptm": ("S", "C", "C"),
    "atom_pad_mask": ("B", "A"),
    "residue_index": ("B", "L"),
    "entity_id": ("B", "L"),
    "output_embedding_sequence": ("B", "L", None),
    "output_embedding_pair_pooled": ("B", "L", None),
}


def _validate_features(features: dict[str, Tensor]) -> dict[str, int]:
    unknown = set(features) - FEATURE_AXES.keys()
    if unknown:
        raise ValueError(f"Features missing an axis schema: {sorted(unknown)}")
    sizes = {"B": 1}
    for key, value in features.items():
        axes = FEATURE_AXES[key]
        if not isinstance(value, Tensor) or value.ndim != len(axes):
            raise ValueError(f"{key} must be a tensor with axes {axes}")
        for axis, size in zip(axes, value.shape):
            if axis is not None:
                if axis in sizes and sizes[axis] != size:
                    raise ValueError(f"Inconsistent {axis} dimension in {key}")
                sizes[axis] = size
    for key in ("token_attention_mask", "atom_attention_mask"):
        if key not in features:
            raise ValueError(f"Missing required feature: {key}")
        mask = features[key][0].bool()
        count = int(mask.sum())
        if not count or not mask[:count].all() or mask[count:].any():
            raise ValueError(
                f"{key} must contain valid entries followed by trailing padding"
            )
        if key == "token_attention_mask" and count != mask.numel():
            raise ValueError(
                "Per-design token features must be unpadded before collating"
            )
    if sizes["A"] % 32:
        raise ValueError("Atom axis must be padded to a multiple of 32")
    return sizes


def collate_features(inputs: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad B=1 feature dictionaries into one batch, without changing local indices.

    Inputs must have identical key sets, dtypes/devices and fixed feature dims.
    Masks mark padding false; gather indices use safe zero padding. MSA rows,
    token axes and atom axes have separate padding extents. Unknown fields fail
    closed until their axis schema is declared. No inputs are modified.
    """
    if not inputs:
        raise ValueError("Cannot collate an empty batch")
    if any(set(item) != set(inputs[0]) for item in inputs):
        raise ValueError("All inputs must have the same feature keys")
    sizes = [_validate_features(item) for item in inputs]
    maxima = {axis: max(size[axis] for size in sizes) for axis in sizes[0]}
    maxima["B"] = len(inputs)
    batch = {}
    for key, first in inputs[0].items():
        axes = FEATURE_AXES[key]
        shape = tuple(
            maxima[axis] if axis else size for axis, size in zip(axes, first.shape)
        )
        # input_ids uses the ESMC PAD token; MSA padding is masked out.
        padded = first.new_full(shape, 1 if key == "input_ids" else 0)
        for index, item in enumerate(inputs):
            value = item[key]
            if value.dtype != first.dtype or value.device != first.device:
                raise ValueError(f"Inconsistent dtype/device for {key}")
            if any(
                axis is None and size != first.shape[d]
                for d, (axis, size) in enumerate(zip(axes, value.shape))
            ):
                raise ValueError(f"Inconsistent fixed feature dimension for {key}")
            slices = (
                slice(index, index + 1),
                *(slice(0, size) for size in value.shape[1:]),
            )
            padded[slices] = value
        batch[key] = padded
    return batch


def split_outputs(
    output: dict[str, Tensor],
    inputs: Sequence[dict[str, Tensor]],
    *,
    num_diffusion_samples: int,
    trim_atom_padding: bool = False,
) -> list[dict[str, Tensor]]:
    """Restore each design's samples and dimensions using explicit field schemas.

    Token padding is removed. By default the atom extent is restored to the
    original B=1 feature tensor (including its 32-atom alignment padding), so
    these outputs can be passed to the existing decoder with those features.
    Set ``trim_atom_padding=True`` only for an archive copy; do not combine
    trimmed atom arrays with still-padded structure-building masks or indices.
    Returns views, retaining a sample axis even when there is just one sample.
    """
    if (
        not inputs
        or not isinstance(num_diffusion_samples, int)
        or num_diffusion_samples < 1
    ):
        raise ValueError("Need at least one input and one diffusion sample")
    unknown = set(output) - OUTPUT_AXES.keys()
    if unknown:
        raise ValueError(f"Outputs missing an axis schema: {sorted(unknown)}")
    sizes = [_validate_features(item) for item in inputs]
    maxima = {axis: max(size[axis] for size in sizes) for axis in sizes[0]}
    maxima.update(B=len(inputs), S=len(inputs) * num_diffusion_samples)
    maxima["C"] = max(int(item["asym_id"].max()) + 1 for item in inputs)
    for key, value in output.items():
        axes = OUTPUT_AXES[key]
        if not isinstance(value, Tensor) or value.ndim != len(axes):
            raise ValueError(f"{key} must be a tensor with axes {axes}")
        if any(
            axis in maxima and value.shape[d] != maxima[axis]
            for d, axis in enumerate(axes)
        ):
            raise ValueError(f"Unexpected output shape for {key}: {tuple(value.shape)}")
    results = []
    for index, (features, size) in enumerate(zip(inputs, sizes)):
        n_tokens = int(features["token_attention_mask"].sum())
        n_atoms = (
            int(features["atom_attention_mask"].sum())
            if trim_atom_padding
            else size["A"]
        )
        n_chains = int(features["asym_id"][0, :n_tokens].max()) + 1
        slices = {
            "B": slice(index, index + 1),
            "S": slice(
                index * num_diffusion_samples, (index + 1) * num_diffusion_samples
            ),
            "L": slice(0, n_tokens),
            "A": slice(0, n_atoms),
            "C": slice(0, n_chains),
            None: slice(None),
        }
        results.append(
            {
                key: value[tuple(slices[axis] for axis in OUTPUT_AXES[key])]
                for key, value in output.items()
            }
        )
    return results
