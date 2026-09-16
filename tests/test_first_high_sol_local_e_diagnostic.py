from __future__ import annotations

import math

import pytest
import torch

from sol_h3 import first_high_sol_local_diagnostic as e


def _request():
    return (
        ("api", 1),
        ("capture_id", "capture-e"),
        ("mode", e.MODE),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", 0.8780487775802612),
        ("target_shapes_digest", "a" * 64),
        ("source_contract_digest", "b" * 64),
    )


def test_request_parser_is_bounded_and_rejects_w_or_unrelated_attention_modes():
    options = {e.REQUEST_KEY: _request(), "h3_flow_stage": "high"}
    parsed = e.parse_request(options)
    assert parsed is not None
    assert parsed["mode"] == e.MODE
    assert parsed["logical_call_limit"] == 1

    with pytest.raises(RuntimeError, match="cannot coexist with W"):
        e.parse_request({**options, "h3_first_high_operator_diagnostic_v1": object()})
    with pytest.raises(RuntimeError, match="external/reduced"):
        e.parse_request({**options, "vdn_h3_external_sequence_v1": object()})
    with pytest.raises(RuntimeError, match="weighted/Mixed-Grid"):
        e.parse_request({**options, "attention_measure_v1": object()})
    with pytest.raises(RuntimeError, match="no-Untwist"):
        e.parse_request({**options, "minimax_h3_untwist_rope": {"enabled": False}})


def test_detailed_bthd_metrics_report_tile_head_p99_and_exact_worst_coordinate():
    want = torch.zeros((1, 4, 2, 3), dtype=torch.float32)
    got = want.clone()
    got[0, 2, 1, 2] = 1.0

    metrics = e._detailed_bthd_metrics(got, want)

    assert metrics["finite"] is True
    assert metrics["max_abs"] == 1.0
    assert metrics["method"] == "exact_all_values"
    assert metrics["sample_count"] == got.numel()
    assert metrics["worst_coordinate"] == {"row": 2, "q_block": 0, "head": 1, "dim": 2}
    assert len(metrics["per_head"]) == 2
    assert len(metrics["per_q64"]) == 1
    assert len(metrics["per_head_q64"]["max_abs"]) == 1
    assert metrics["per_head_q64"]["max_abs"][0][1] == 1.0


def test_frozen_route_reference_all_exact_matches_dense_attention_and_denominator():
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, -1.0]]]])
    v = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]])
    scale = 1.0 / math.sqrt(2.0)
    scores = q[0, :, 0] @ k[0, :, 0].T * scale
    probabilities = torch.softmax(scores, dim=-1)
    dense = (probabilities @ v[0, :, 0]).view(1, 3, 1, 2)
    lse = torch.logsumexp(scores, dim=-1).view(1, 3, 1)
    kc = k.mean(dim=1, keepdim=True)
    vc = v.sum(dim=1, keepdim=True)
    routes = torch.ones((1, 1, 1), dtype=torch.bool)

    report = e._frozen_route_reference(q, k, v, kc, vc, routes, dense, lse, scale=scale)

    assert report["finite"] is True
    assert report["output"]["max_abs"] < 1.0e-6
    assert report["numerator_scaled_to_reference_rowmax"]["max_abs"] < 1.0e-5
    assert report["denominator_scaled_to_reference_rowmax"]["max_abs"] < 1.0e-6
    assert report["lse"]["max_abs"] < 1.0e-6
    assert report["denominator_relative_error"]["max"] < 1.0e-6
    assert report["score_chunk_keys"] == 1024
    assert report["max_live_score_bytes_fp32"] <= 64 * 1024 * 4


def test_route_margin_summary_distinguishes_near_threshold_and_geometry_forced_pairs():
    margins = torch.tensor([[[1.0e-5, -2.0e-3, 0.5]]], dtype=torch.float32)
    forced = torch.tensor([[[1, 0, 2]]], dtype=torch.uint8)
    mismatch = torch.tensor([[[True, True, False]]])

    report = e._route_margin_summary(margins, forced, mismatch)

    assert report["pairs_within_1e-4_log2"] == 1
    assert report["geometry_forced_pairs"] == 1
    assert report["mismatch_within_1e-3_log2"] == 1
    assert report["mismatch_farther_than_1e-3_log2"] == 1
