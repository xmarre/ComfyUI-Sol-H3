import torch

from sol_h3.mixed_measure import (
    kv_gather_indices,
    representative_spatial_indices,
    validate_measure_contract,
)


def production_external():
    return {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 49741,
        "sequence_rows": 56029,
        "video_start": 16261,
        "temporal": 62,
        "prefix_t": 12,
        "source_rows_per_frame": 540,
        "prefix_rows_per_frame": 1064,
    }


def production_measure():
    return {
        "api": 1,
        "mode": "prefix_kv_stratified_subsample",
        "video_start": 16261,
        "sequence_rows": 56029,
        "temporal": 62,
        "prefix_t": 12,
        "source_grid_h": 20,
        "source_grid_w": 27,
        "prefix_grid_h": 28,
        "prefix_grid_w": 38,
        "source_rows_per_frame": 540,
        "prefix_rows_per_frame": 1064,
        "expected_kv_rows": 49741,
        "exact_prefix_queries_preserved": True,
        "suffix_kv_unchanged": True,
    }


def test_00318_measure_contract_reduces_to_native_carrier_rows():
    validated = validate_measure_contract(production_measure(), production_external(), q_rows=56029, kv_rows=56029)
    assert validated["expected_kv_rows"] == 49741
    representatives = validated["representative_spatial_indices"]
    assert len(representatives) == 540
    assert len(set(representatives)) == 540
    assert min(representatives) >= 0
    assert max(representatives) < 1064
    indices = kv_gather_indices(validated, device="cpu")
    assert indices.dtype == torch.long
    assert indices.numel() == 49741
    assert torch.equal(indices[:16261], torch.arange(16261))
    suffix_start = 16261 + 12 * 1064
    assert torch.equal(indices[-50 * 540 :], torch.arange(suffix_start, 56029))
    assert indices.unique().numel() == indices.numel()


def test_representative_map_is_deterministic_and_spatially_ordered():
    first = representative_spatial_indices(20, 27, 28, 38)
    second = representative_spatial_indices(20, 27, 28, 38)
    assert first == second
    assert all(a < b for a, b in zip(first, first[1:]))
