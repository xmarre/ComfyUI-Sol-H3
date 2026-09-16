import os
from pathlib import Path

import pytest
import torch

from tools.mapped_neighbor_probe import (
    CONTROL_DESCRIPTOR_SHA256,
    CONTROL_KV_ROWS,
    CONTROL_Q_ROWS,
    CONTROL_QUERY_POSITION_SHA256,
    _decode_route_trace,
    _descriptor_sha256,
    _mapped_mask,
    _operator_witness,
    _query_positions_from_runs,
    _route_counts,
    _tensor_sha256,
    controlled_geometry_report,
)


def test_route_union_is_additive_and_preserves_old_selection():
    old = torch.tensor(
        [
            [[True, False, False, True, False], [False, True, False, False, False]],
            [[False, False, True, False, False], [True, False, False, False, True]],
        ]
    )
    mapped = _mapped_mask(((1, 3), (3, 5)), heads=2, k_blocks=5, device=torch.device("cpu"))
    counts = _route_counts(old, mapped)
    effective = old | mapped

    assert not (old & ~effective).any()
    assert counts == {
        "original_selected_pairs": int(old.sum()),
        "mapped_candidate_pairs": int(mapped.sum()),
        "added_selected_pairs": int((mapped & ~old).sum()),
        "effective_selected_pairs": int(effective.sum()),
        "total_block_pairs": old.numel(),
    }
    assert counts["effective_selected_pairs"] == (
        counts["original_selected_pairs"] + counts["added_selected_pairs"]
    )


def test_decode_route_trace_round_trip_two_words():
    trace = torch.zeros((1, 1, 1, 1, 2), dtype=torch.int32)
    # word 0 sets bits 0 and 31; int32 stores that ballot as signed -2147483647.
    trace[0, 0, 0, 0, 0] = -2147483647
    trace[0, 0, 0, 0, 1] = (1 << 1) | (1 << 30)
    decoded = _decode_route_trace(trace, 64)
    selected = decoded[0, 0].nonzero(as_tuple=False).flatten().tolist()
    assert selected == [0, 31, 33, 62]


def test_query_position_run_expansion_and_hash_are_deterministic():
    runs = ((0, 3, 5), (3, 5, 10))
    positions = _query_positions_from_runs(runs, 5)
    assert positions.tolist() == [5, 6, 7, 10, 11]
    assert _tensor_sha256(positions) == _tensor_sha256(positions.clone())
    assert _descriptor_sha256(((0, 2), (3, 4))) == _descriptor_sha256(((0, 2), (3, 4)))


def test_operator_witness_accepts_full_payload_or_bounded_shard():
    record = {
        "kind": "operator_witness",
        "block_index": 2,
        "group_index": 10,
        "q": torch.zeros(1),
    }
    assert _operator_witness(record, 2, 10) is record
    payload = {"schema_version": 1, "evidence": [{"kind": "other"}, record]}
    assert _operator_witness(payload, 2, 10) is record


@pytest.mark.skipif(not os.environ.get("VDN_PATH"), reason="set VDN_PATH for production v4 geometry integration")
def test_controlled_v4_geometry_matches_preserved_m_digests():
    vdn_path = Path(os.environ["VDN_PATH"])
    for group_index in range(11):
        report = controlled_geometry_report(vdn_path, group_index)
        assert report["q_rows"] == CONTROL_Q_ROWS[group_index]
        assert report["kv_rows"] == CONTROL_KV_ROWS[group_index]
        assert report["sink_rows"] == 3101
        assert report["diagnostic_descriptor_sha256"] == CONTROL_DESCRIPTOR_SHA256[group_index]
        assert report["query_positions_sha256"] == CONTROL_QUERY_POSITION_SHA256[group_index]
        assert report["matches_preserved_m_geometry"] is True
