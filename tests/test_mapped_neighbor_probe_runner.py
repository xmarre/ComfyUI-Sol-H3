import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_mapped_neighbor_probe import (
    CAPTURE_ID,
    EXPECTED_RUNTIME,
    EXPECTED_SOL_BLOBS,
    EXPECTED_VDN_BLOBS,
    _child_report_matches_request,
    _extract_final_json,
    _git_blob_sha,
    _require_preserved_runtime,
    _require_preserved_tau,
    _source_gate,
)


def test_current_sol_probe_sources_match_reviewed_blob_identities():
    root = Path(__file__).resolve().parents[1]
    reports = _source_gate(root, EXPECTED_SOL_BLOBS, "sol")
    assert {item["relative_path"] for item in reports} == set(EXPECTED_SOL_BLOBS)
    assert all(item["git_blob_sha"] == EXPECTED_SOL_BLOBS[item["relative_path"]] for item in reports)


@pytest.mark.skipif(not os.environ.get("VDN_PATH"), reason="set VDN_PATH for production VDN source gate")
def test_current_vdn_probe_sources_match_reviewed_blob_identities():
    root = Path(os.environ["VDN_PATH"])
    reports = _source_gate(root, EXPECTED_VDN_BLOBS, "vdn")
    assert {item["relative_path"] for item in reports} == set(EXPECTED_VDN_BLOBS)
    assert all(item["git_blob_sha"] == EXPECTED_VDN_BLOBS[item["relative_path"]] for item in reports)


def test_extract_final_json_ignores_compile_noise():
    stdout = "compiler: {not json}\nmore noise\n{\n  \"kind\": \"production_mapped_neighbor_same_input_probe_v1\",\n  \"production_same_input_gate_pass\": true\n}\n"
    value = _extract_final_json(stdout)
    assert value["kind"] == "production_mapped_neighbor_same_input_probe_v1"
    assert value["production_same_input_gate_pass"] is True


def test_extract_final_json_rejects_missing_expected_report():
    with pytest.raises(RuntimeError, match="expected JSON report"):
        _extract_final_json("compile complete\n{\"kind\": \"other\"}\n")


def test_preserved_runtime_gate_is_exact():
    _require_preserved_runtime(dict(EXPECTED_RUNTIME))
    changed = dict(EXPECTED_RUNTIME)
    changed["cutlass_dsl"] = "4.7.2"
    with pytest.raises(RuntimeError, match="preserved M runtime"):
        _require_preserved_runtime(changed)


def test_preserved_tau_gate_rejects_same_input_drift():
    assert _require_preserved_tau(1.0) == 1.0
    for value in (0.9, 1.1, float("nan"), True, "1.0"):
        with pytest.raises(ValueError, match="requires tau=1.0"):
            _require_preserved_tau(value)


def test_child_report_must_match_requested_capture_block_group_and_tau():
    args = SimpleNamespace(block=2, group=10, tau=1.0)
    report = {
        "production_same_input_gate_pass": True,
        "capture_id": CAPTURE_ID,
        "block_index": 2,
        "group_index": 10,
        "tau": 1.0,
    }
    assert _child_report_matches_request(report, args) is True

    for key, value in (
        ("capture_id", "wrong"),
        ("block_index", 3),
        ("group_index", 2),
        ("tau", 0.9),
        ("production_same_input_gate_pass", False),
    ):
        changed = dict(report)
        changed[key] = value
        assert _child_report_matches_request(changed, args) is False


def test_git_blob_sha_matches_git_object_formula(tmp_path):
    path = tmp_path / "value.txt"
    path.write_bytes(b"abc\n")
    assert _git_blob_sha(path) == "8baef1b4abc478178b004d62031cf7fe6db6f903"
