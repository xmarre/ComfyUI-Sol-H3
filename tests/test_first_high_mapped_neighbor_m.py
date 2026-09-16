from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sol_h3 import first_high_mapped_neighbor_diagnostic as m
from sol_h3 import first_high_sol_local_diagnostic as e


def _request():
    return (
        ("api", 1),
        ("capture_id", "capture-m"),
        ("mode", m.MODE),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", 0.8780487775802612),
        ("target_shapes_digest", "a" * 64),
        ("source_contract_digest", "b" * 64),
    )


def test_m_overlay_retargets_exact_e_request_mode_only():
    parsed = e.parse_request({e.REQUEST_KEY: _request(), "h3_flow_stage": "high"})
    assert parsed is not None
    assert parsed["mode"] == "mapped_neighbor_m"


def test_interval_contract_is_one_bounded_interval_per_q64_tile():
    assert m._validate_intervals(128, 320, ((0, 3), (2, 6))) == ((0, 3), (2, 6))
    with pytest.raises(RuntimeError, match="one bounded"):
        m._validate_intervals(128, 320, ((0, 3),))
    with pytest.raises(RuntimeError, match="reviewed.*bound"):
        m._validate_intervals(64, 640, ((0, 5),))
    with pytest.raises(RuntimeError, match="outside"):
        m._validate_intervals(64, 128, ((1, 4),))


def test_route_evidence_is_strictly_additive_and_counts_old_ordinal_route():
    options = {e.REQUEST_KEY: _request(), e.EVIDENCE_KEY: SimpleNamespace(append=lambda _x: None)}
    token = m.enter_mapped_group(
        options,
        {
            "block_index": 2,
            "group_index": 10,
            "q_rows": 64,
            "kv_rows": 256,
            "original_sink_rows": 64,
            "scale": 1.0,
            "mapped_neighbor_intervals": ((2, 4),),
            "query_positions_sha256": "c" * 64,
        },
    )
    try:
        group = m._GROUP.get()
        qbar = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
        kc = torch.zeros((1, 4, 1, 2), dtype=torch.bfloat16)
        threshold = torch.ones((1, 1, 1), dtype=torch.float32)
        report = m._route_evidence(group, qbar, kc, threshold)
    finally:
        m.exit_mapped_group(token)

    # Old selector: sink block 0 plus ordinal neighbour block 1 -> two pairs.
    # M interval [2,4) adds blocks 2 and 3 and removes nothing.
    assert report["original_selected_pairs"] == 2
    assert report["added_selected_pairs"] == 2
    assert report["effective_selected_pairs"] == 4
    assert report["total_block_pairs"] == 4
    assert report["additive_only"] is True
    assert report["restricted_domain_unchanged"] is True
    assert math.isclose(report["exact_work_increase_fraction"], 1.0)


def test_group_context_rejects_non_m_or_oversized_descriptor():
    with pytest.raises(RuntimeError):
        m.enter_mapped_group(
            {},
            {
                "block_index": 2,
                "group_index": 0,
                "q_rows": 64,
                "kv_rows": 640,
                "original_sink_rows": 0,
                "scale": 1.0,
                "mapped_neighbor_intervals": ((0, 2),),
                "query_positions_sha256": "d" * 64,
            },
        )
