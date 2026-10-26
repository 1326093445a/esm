"""Mixed-length batching: design/sample ownership, padding and real model paths."""

import numpy as np
import pytest
import torch

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
    collate_features,
    layers,
    split_outputs,
)
from esm.models.esmfold2.batching import FEATURE_AXES, OUTPUT_AXES
from esm.models.esmfold2.protein_utils import prepare_protein_features

FAST = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=3)


@pytest.fixture(autouse=True)
def reference_attention(monkeypatch):
    monkeypatch.setattr(layers, "FLASH_ATTN_AVAILABLE", False)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def features():
    return [prepare_protein_features(seq) for seq in ("GA", "ACDEFGHIK")]


@pytest.fixture
def builder(ccd_pickle):
    return ESMFold2InputBuilder()


def protein(sequence, chain="A"):
    return StructurePredictionInput(
        sequences=[ProteinInput(id=chain, sequence=sequence)]
    )


def test_collator_pads_each_axis_and_preserves_local_indices(features):
    features[1]["msa"] = features[1]["msa"].expand(1, 3, -1).clone()
    for key in ("has_deletion", "deletion_value", "msa_attention_mask"):
        features[1][key] = features[1][key].expand(1, 3, -1).clone()
    batch = collate_features(features)
    assert batch["token_attention_mask"].shape == (2, 9)
    assert batch["msa"].shape == (2, 3, 9)
    assert batch["token_bonds"].shape == (2, 9, 9, 1)
    assert batch["ref_pos"].shape == (2, 96, 3)
    assert batch["token_attention_mask"].sum(1).tolist() == [2, 9]
    assert not batch["msa_attention_mask"][0, 1:].any()
    assert not batch["msa_attention_mask"][0, :, 2:].any()
    assert (batch["input_ids"][0, 2:] == 1).all()
    for index, original in enumerate(features):
        for key, value in original.items():
            slices = (slice(index, index + 1), *(slice(0, n) for n in value.shape[1:]))
            torch.testing.assert_close(batch[key][slices], value, atol=0, rtol=0)
    assert features[0]["token_attention_mask"].shape == (1, 2)


def test_cropped_lm_states_are_batched_on_token_axis(features):
    for item in features:
        length = item["input_ids"].shape[1]
        item["lm_hidden_states"] = torch.randn(1, length, 3, 8)
    batch = collate_features(features)
    assert batch["lm_hidden_states"].shape == (2, 9, 3, 8)
    assert not batch["lm_hidden_states"][0, 2:].any()


@pytest.mark.parametrize(
    "fault", ["unknown", "keys", "dtype", "mask", "batch", "fixed"]
)
def test_collator_rejects_ambiguous_or_malformed_features(features, fault):
    if fault == "unknown":
        for item in features:
            item["new_field"] = torch.ones(1, 1)
    elif fault == "keys":
        del features[1]["input_ids"]
    elif fault == "dtype":
        features[1]["ref_pos"] = features[1]["ref_pos"].double()
    elif fault == "mask":
        features[1]["atom_attention_mask"][0, 1] = False
    elif fault == "batch":
        features[1]["input_ids"] = features[1]["input_ids"].expand(2, -1)
    else:
        features[1]["ref_pos"] = features[1]["ref_pos"][..., :2]
    with pytest.raises(ValueError):
        collate_features(features)


def test_msa_subsampling_does_not_replace_valid_rows_with_batch_padding():
    msa = torch.arange(20).reshape(2, 5, 2)
    mask = torch.ones_like(msa, dtype=torch.bool)
    mask[0, 2:] = False
    sampled, sampled_mask, _, _ = layers.maybe_subsample_msa(
        msa, mask, None, None, max_depth=3, enabled=True
    )
    assert sampled_mask.any(-1).sum(-1).tolist() == [2, 3]
    torch.testing.assert_close(sampled[0, :2], msa[0, :2])
    torch.testing.assert_close(sampled[:, 0], msa[:, 0])
    assert not sampled_mask[0, 2].any()
    assert len(sampled[1, :, 0].unique()) == 3


def test_split_does_not_confuse_design_samples_xyz_or_feature_channels():
    # Lmax == number of samples == XYZ dimension. The first design has L=2.
    inputs = [prepare_protein_features(seq) for seq in ("GA", "GGG")]
    coords = torch.arange(6 * 32 * 3).reshape(6, 32, 3).float()
    plddt = torch.arange(18).reshape(6, 3).float()
    distogram = torch.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)
    output = dict(
        sample_atom_coords=coords,
        plddt=plddt,
        ptm=torch.tensor([0.1, 0.9, 0.2, 0.95, 0.1, 0.2]),
        distogram_logits=distogram,
    )
    split = split_outputs(output, inputs, num_diffusion_samples=3)
    assert [int(item["ptm"].argmax()) for item in split] == [1, 0]
    torch.testing.assert_close(split[0]["sample_atom_coords"], coords[:3])
    torch.testing.assert_close(split[1]["sample_atom_coords"], coords[3:])
    torch.testing.assert_close(split[0]["plddt"], plddt[:3, :2])
    torch.testing.assert_close(split[1]["plddt"], plddt[3:])
    assert split[0]["distogram_logits"].shape == (1, 2, 2, 3)
    archive = split_outputs(
        output, inputs, num_diffusion_samples=3, trim_atom_padding=True
    )
    assert archive[0]["sample_atom_coords"].shape == (3, 9, 3)
    assert archive[1]["sample_atom_coords"].shape == (3, 12, 3)
    assert split[0]["sample_atom_coords"].shape == (3, 32, 3)
    assert coords.shape == (6, 32, 3)


def test_split_rejects_wrong_sample_counts_and_unknown_outputs(features):
    with pytest.raises(ValueError, match="Unexpected output shape"):
        split_outputs({"ptm": torch.ones(2)}, features, num_diffusion_samples=3)
    with pytest.raises(ValueError, match="axis schema"):
        split_outputs({"new_output": torch.ones(6)}, features, num_diffusion_samples=3)


def test_real_model_mixed_lengths_and_all_output_schemas(tiny_esmfold2, features):
    torch.manual_seed(9)
    with torch.no_grad():
        output = tiny_esmfold2(
            **collate_features(features), **FAST, include_embeddings=True
        )
    assert set(output) <= OUTPUT_AXES.keys()
    split = split_outputs(
        output, features, num_diffusion_samples=3, trim_atom_padding=True
    )
    for item, prediction in zip(features, split):
        length = int(item["token_attention_mask"].sum())
        atoms = int(item["atom_attention_mask"].sum())
        assert prediction["sample_atom_coords"].shape == (3, atoms, 3)
        assert prediction["plddt"].shape == (3, length)
        assert prediction["pae"].shape == (3, length, length)
        assert prediction["ptm"].shape == (3,)
        assert all(torch.isfinite(value).all() for value in prediction.values())


def test_trunk_padding_does_not_change_valid_outputs(
    tiny_esmfold2, features, monkeypatch
):
    # Control stochastic initialization. This tests numerical masking, not
    # whether differently shaped diffusion RNG draws happen to be identical.
    monkeypatch.setattr(tiny_esmfold2, "_init_pair_state", torch.zeros_like)
    with torch.no_grad():
        batch = tiny_esmfold2(
            **collate_features(features), **FAST, include_embeddings=True
        )
        separated = split_outputs(batch, features, num_diffusion_samples=3)
        for item, prediction in zip(features, separated):
            single = tiny_esmfold2(**item, **FAST, include_embeddings=True)
            for key in ("distogram_logits", "output_embedding_pair_pooled"):
                torch.testing.assert_close(
                    prediction[key], single[key], atol=2e-5, rtol=2e-5
                )


def test_confidence_matches_singletons_for_identical_coordinates(
    tiny_esmfold2, features, monkeypatch
):
    # Hold pair initialization and sampled coordinates fixed, isolating the
    # confidence head from random draws that legitimately depend on batch size.
    monkeypatch.setattr(tiny_esmfold2, "_init_pair_state", torch.zeros_like)
    # Padding uses chain ID 0. A valid chain with ID 1 must not gain a fake
    # interface through padding's representative atom at index zero.
    features[0]["asym_id"].fill_(1)

    def fixed_sample(**kwargs):
        return {"sample_atom_coords": kwargs["ref_pos"].repeat_interleave(3, 0)}

    monkeypatch.setattr(tiny_esmfold2.structure_head, "sample", fixed_sample)
    with torch.no_grad():
        batch = tiny_esmfold2(**collate_features(features), **FAST)
        separated = split_outputs(batch, features, num_diffusion_samples=3)
        for item, prediction in zip(features, separated):
            single = tiny_esmfold2(**item, **FAST)
            for key in (
                "plddt",
                "pae",
                "ptm",
                "iptm",
                "complex_plddt",
                "complex_iplddt",
            ):
                torch.testing.assert_close(
                    prediction[key], single[key], atol=2e-5, rtol=2e-5
                )


def test_diffusion_padding_with_controlled_noise(tiny_esmfold2, features, monkeypatch):
    monkeypatch.setattr(tiny_esmfold2, "_init_pair_state", torch.zeros_like)

    def fixed_noise(*size, device=None, dtype=None, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 else size
        # Every batch row gets the same atom-index-dependent noise; extending
        # the atom axis preserves each valid prefix. Includes rotations too.
        shape_tail = shape[1:]
        count = int(np.prod(shape_tail))
        values = torch.arange(count, device=device, dtype=dtype or torch.float32)
        return (values + 1).sin().reshape(1, *shape_tail).expand(shape).clone()

    monkeypatch.setattr(torch, "randn", fixed_noise)
    monkeypatch.setattr(
        torch,
        "randn_like",
        lambda x, **kw: fixed_noise(x.shape, dtype=x.dtype, device=x.device),
    )
    with torch.no_grad():
        batch = tiny_esmfold2(**collate_features(features), **FAST)
        separated = split_outputs(batch, features, num_diffusion_samples=3)
        for item, prediction in zip(features, separated):
            single = tiny_esmfold2(**item, **FAST)
            real_atoms = int(item["atom_attention_mask"].sum())
            torch.testing.assert_close(
                prediction["sample_atom_coords"][:, :real_atoms],
                single["sample_atom_coords"][:, :real_atoms],
                atol=2e-3,
                rtol=2e-4,
            )


def test_prepare_and_decode_preserve_design_identity_and_every_sample(
    builder, tiny_esmfold2
):
    inputs = [
        protein("GA", "X"),
        StructurePredictionInput(
            sequences=[
                ProteinInput(id="Y", sequence="ACD"),
                ProteinInput(id="Z", sequence="GG"),
            ]
        ),
    ]
    prepared = [builder.prepare_input(item) for item in inputs]
    features = [item[0] for item in prepared]
    assert all(set(item) <= FEATURE_AXES.keys() for item in features)
    with torch.no_grad():
        output = tiny_esmfold2(**collate_features(features), **FAST)
    results = builder.decode_batch(
        output,
        features,
        [item[1] for item in prepared],
        num_diffusion_samples=3,
        complex_ids=["first", "second"],
    )
    assert [len(samples) for samples in results] == [3, 3]
    for design, samples in enumerate(results):
        n_tokens = (2, 5)[design]
        atoms = int(features[design]["atom_attention_mask"].sum())
        for sample, result in enumerate(samples):
            assert result.complex.id == ("first", "second")[design]
            assert result.complex.atom_positions.shape == (atoms, 3)
            assert result.plddt.shape == (n_tokens,)
            assert result.pae.shape == (n_tokens, n_tokens)
            assert result.pde.shape == (n_tokens, n_tokens)
            assert result.pair_chains_iptm.shape == ((1, 1) if design == 0 else (2, 2))
            expected = output["sample_atom_coords"][design * 3 + sample, :atoms].numpy()
            np.testing.assert_array_equal(result.complex.atom_positions, expected)
            torch.testing.assert_close(
                result.plddt, output["plddt"][design * 3 + sample, :n_tokens]
            )
            # CIF identity survives writing, not merely tensor slicing.
            cif = result.complex.to_mmcif()
            assert f"data_{result.complex.id}" in cif
            flat_index = design * 3 + sample
            singleton = builder.decode(
                {
                    "sample_atom_coords": output["sample_atom_coords"][
                        flat_index : flat_index + 1
                    ],
                    "plddt": output["plddt"][flat_index : flat_index + 1, :n_tokens],
                },
                features[design],
                prepared[design][1],
                complex_id=result.complex.id,
            )
            assert cif == singleton.complex.to_mmcif()
        assert set(samples[0].complex.metadata.chain_lookup.values()) == (
            {"X"} if design == 0 else {"Y", "Z"}
        )


@pytest.mark.parametrize("batch_size,samples", [(1, 1), (2, 3), (3, 2)])
def test_fold_batch_uses_one_forward_and_returns_nested_samples(
    builder, tiny_esmfold2, batch_size, samples
):
    calls = []
    hook = tiny_esmfold2.register_forward_pre_hook(
        lambda model, args, kwargs: calls.append(kwargs["token_attention_mask"].shape),
        with_kwargs=True,
    )
    try:
        sequences = ["GA", "ACDEFG", "MKV"][:batch_size]
        results = builder.fold_batch(
            tiny_esmfold2,
            [protein(seq) for seq in sequences],
            **{**FAST, "num_diffusion_samples": samples},
            seed=7,
        )
    finally:
        hook.remove()
    assert calls == [(batch_size, max(map(len, sequences)))]
    assert [len(item) for item in results] == [samples] * batch_size


def test_singleton_fold_batch_agrees_exactly_with_fold(builder, tiny_esmfold2):
    kwargs = dict(FAST, seed=4, complex_id="a")
    single = builder.fold(tiny_esmfold2, protein("ACDEFG"), **kwargs)
    batch = builder.fold_batch(
        tiny_esmfold2, [protein("ACDEFG")], **FAST, seed=4, complex_ids=["a"]
    )
    assert len(batch) == 1
    for left, right in zip(single, batch[0]):
        np.testing.assert_array_equal(
            left.complex.atom_positions, right.complex.atom_positions
        )
        torch.testing.assert_close(left.pae, right.pae, atol=0, rtol=0)


def test_single_design_decoder_rejects_a_batch(builder, features):
    with pytest.raises(ValueError, match="decode_batch"):
        builder.decode({}, collate_features(features), [])
