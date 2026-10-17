from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_h3.keyless_real_h3_replay_node import (
    SolH3KeylessRealH3Replay,
    _extract_json_object,
    _report_path,
)


def test_extract_json_object_accepts_leading_noise():
    payload = {"all_blocks_pass": True, "results": [{"block_index": 0}]}
    text = "warning before json\n" + json.dumps(payload)
    assert _extract_json_object(text) == payload


def test_extract_json_object_rejects_missing_payload():
    with pytest.raises(RuntimeError, match="did not return a JSON object"):
        _extract_json_object("warning only")


def test_report_path_is_small_report_under_output(tmp_path: Path):
    payload = {
        "checkpoint_kind": "teacher",
        "results": [
            {"block_index": 0},
            {"block_index": 25},
            {"block_index": 49},
        ],
    }
    report = _report_path(tmp_path, payload, "a" * 64)
    assert report == (
        tmp_path
        / "keyless_real_h3_replay_reports"
        / "real-h3-replay-aaaaaaaaaaaa.teacher.blocks-0-25-49.json"
    )
    assert report.parent.is_dir()


def test_node_is_output_node_and_exposes_no_dataset_case_inputs():
    assert SolH3KeylessRealH3Replay.OUTPUT_NODE is True
    required = SolH3KeylessRealH3Replay.INPUT_TYPES()["required"]
    assert "capture_path" in required
    assert "checkpoint_name" in required
    assert "checkpoint_kind" in required
    assert "case_id" not in required
    assert "dataset_manifest_path" not in required
    assert "split" not in required
