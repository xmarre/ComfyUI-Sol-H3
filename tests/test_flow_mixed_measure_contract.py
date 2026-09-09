from __future__ import annotations

import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH") or not os.environ.get("FLOW_PATH"),
    reason="set COMFYUI_PATH and FLOW_PATH for real Flow source contracts",
)


def _activate_real_sources():
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["FLOW_PATH"])

    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def test_real_flow_00318_measure_contract_matches_sol_and_native_coordinates():
    _activate_real_sources()

    from comfy.ldm.minimax.model import _frame_grid
    from h3_flow_regenerate.mixed_grid import MixedGridPlan, mixed_attention_measure_contract
    from sol_h3.mixed_measure import kv_gather_indices, reduce_kv, validate_measure_contract

    plan = MixedGridPlan(
        prefix=torch.empty(1, 24, 12, 56, 76),
        temporal=62,
        source_h=40,
        source_w=54,
        attention_measure=True,
    )
    external = {
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
    measure = mixed_attention_measure_contract(
        plan,
        video_start=external["video_start"],
        sequence_rows=external["sequence_rows"],
    )
    assert measure is not None
    assert measure["expected_kv_rows"] == external["native_sequence_rows"] == 49741

    validated = validate_measure_contract(
        measure,
        external,
        q_rows=external["sequence_rows"],
        kv_rows=external["sequence_rows"],
    )
    reps = tuple(validated["representative_spatial_indices"])
    assert len(reps) == 540
    assert len(set(reps)) == 540

    # Check the Sol-side representative mapping against the exact pinned Comfy
    # MiniMax-H3 area-normalized coordinate implementation rather than a copied
    # formula. Cartesian nearest-neighbour selection must agree row-for-row.
    source_frame, _ = _frame_grid(40, 54)
    prefix_frame, _ = _frame_grid(56, 76)
    distances = ((source_frame[:, None, :] - prefix_frame[None, :, :]) ** 2).sum(dim=-1)
    native_nearest = tuple(distances.argmin(dim=1).tolist())
    assert reps == native_nearest

    indices = kv_gather_indices(validated, device=torch.device("cpu"))
    assert indices.numel() == 49741
    assert torch.equal(indices[:16261], torch.arange(16261))

    selected_prefix_stop = 16261 + 12 * 540
    original_suffix_start = 16261 + 12 * 1064
    assert torch.equal(
        indices[selected_prefix_stop:],
        torch.arange(original_suffix_start, 56029),
    )

    # K/V selection is exactly the validated gather; Q is not an input to the
    # operation and therefore cannot be shortened by this contract.
    rows = torch.arange(56029, dtype=torch.float32).reshape(1, 1, 56029, 1)
    reduced_k, reduced_v, stats = reduce_kv(rows, rows.clone(), validated)
    expected = rows.index_select(2, indices)
    assert torch.equal(reduced_k, expected)
    assert torch.equal(reduced_v, expected)
    assert stats == {
        "kv_rows_before": 56029,
        "kv_rows_after": 49741,
        "kv_rows_removed": 6288,
    }
