import ast
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
    _validate_preserved_route_counts,
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


def test_preserved_public_m_route_report_does_not_require_private_candidate_count():
    counts = {
        "original_selected_pairs": 88393,
        "mapped_candidate_pairs": 3584,
        "added_selected_pairs": 585,
        "effective_selected_pairs": 88978,
        "total_block_pairs": 158592,
    }
    witness = {"sparse_selected_block_pairs": 88393}
    m_record = {
        "valid": True,
        "original_selected_pairs": 88393,
        "added_selected_pairs": 585,
        "effective_selected_pairs": 88978,
        "total_block_pairs": 158592,
        "exact_work_increase_fraction": 0.006618171122147682,
    }

    fraction = _validate_preserved_route_counts(counts, witness, m_record)

    assert fraction == 585 / 88393
    assert "mapped_candidate_pairs" not in m_record


def test_preserved_route_count_validation_rejects_e_m_divergence():
    counts = {
        "original_selected_pairs": 88393,
        "mapped_candidate_pairs": 3584,
        "added_selected_pairs": 585,
        "effective_selected_pairs": 88978,
        "total_block_pairs": 158592,
    }
    witness = {"sparse_selected_block_pairs": 88392}
    m_record = {
        "valid": True,
        "original_selected_pairs": 88393,
        "added_selected_pairs": 585,
        "effective_selected_pairs": 88978,
        "total_block_pairs": 158592,
        "exact_work_increase_fraction": 0.006618171122147682,
    }

    with pytest.raises(RuntimeError, match="preserved E witness"):
        _validate_preserved_route_counts(counts, witness, m_record)


def test_decode_route_trace_round_trip_two_words():
    trace = torch.zeros((1, 1, 1, 1, 2), dtype=torch.int32)
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


def test_same_input_report_serializes_effective_tau():
    source = (Path(__file__).parents[1] / "tools" / "mapped_neighbor_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    matching_reports = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        fields = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        kind = fields.get("kind")
        if isinstance(kind, ast.Constant) and kind.value == "production_mapped_neighbor_same_input_probe_v1":
            matching_reports.append(fields)

    assert len(matching_reports) == 1
    tau = matching_reports[0].get("tau")
    assert isinstance(tau, ast.Attribute)
    assert tau.attr == "tau"
    assert isinstance(tau.value, ast.Name)
    assert tau.value.id == "args"


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
