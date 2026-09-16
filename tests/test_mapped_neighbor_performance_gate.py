import json

import pytest

from tools.historical_m_timing_probe import _m_summary_record
from tools.run_mapped_neighbor_performance_gate import (
    _candidate_geometry,
    _extract_historical_report,
    _extract_runner_result,
    evaluate_performance_gate,
)


def _timing(median, minimum=None, maximum=None):
    minimum = median if minimum is None else minimum
    maximum = median if maximum is None else maximum
    return {
        "median_cuda_ms": median,
        "min_cuda_ms": minimum,
        "max_cuda_ms": maximum,
        "samples_cuda_ms": [minimum, median, maximum],
    }


def _geometry_report():
    return {
        "geometry": {
            "matches_preserved_m_geometry": True,
            "q_rows": 128,
            "kv_rows": 256,
            "sink_rows": 64,
            "query_positions_sha256": "a" * 64,
            "diagnostic_descriptor_sha256": "b" * 64,
            "mapped_neighbor_intervals": [[0, 2], [1, 3]],
        }
    }


def test_performance_gate_accepts_five_percent_budget_boundary():
    result = evaluate_performance_gate(_timing(1.05), _timing(1.0))
    assert result["median_budget_pass"] is True
    assert result["candidate_over_historical_delta_fraction"] == pytest.approx(0.05)
    assert result["observed_sample_range_classification"] == "clear_pass"


def test_performance_gate_rejects_over_budget():
    result = evaluate_performance_gate(_timing(1.051), _timing(1.0))
    assert result["median_budget_pass"] is False
    assert result["observed_sample_range_classification"] == "clear_fail"


def test_performance_gate_reports_observed_range_overlap_separately_from_median():
    result = evaluate_performance_gate(
        _timing(1.04, minimum=1.00, maximum=1.08),
        _timing(1.00, minimum=0.98, maximum=1.02),
    )
    assert result["median_budget_pass"] is True
    assert result["observed_sample_range_classification"] == "overlap"


def test_performance_gate_rejects_invalid_timing():
    with pytest.raises(RuntimeError, match="median CUDA timing is invalid"):
        evaluate_performance_gate(_timing(0.0), _timing(1.0))


def test_candidate_geometry_accepts_preserved_m_matched_descriptor():
    value = _candidate_geometry(_geometry_report())
    assert value["q_rows"] == 128
    assert value["mapped_neighbor_intervals"] == [[0, 2], [1, 3]]
    assert value["diagnostic_descriptor_sha256"] == "b" * 64


def test_candidate_geometry_rejects_unverified_or_wrong_length_descriptor():
    report = _geometry_report()
    report["geometry"]["matches_preserved_m_geometry"] = False
    with pytest.raises(RuntimeError, match="preserved-M-matched"):
        _candidate_geometry(report)

    report = _geometry_report()
    report["geometry"]["mapped_neighbor_intervals"] = [[0, 2]]
    with pytest.raises(RuntimeError, match="descriptor length"):
        _candidate_geometry(report)


def test_m_summary_record_requires_unique_block_group():
    report = {
        "operator_witnesses": {
            "reports": [
                {"block_index": 2, "group_index": 10, "valid": True},
                {"block_index": 2, "group_index": 0, "valid": True},
            ]
        }
    }
    assert _m_summary_record(report, 2, 10)["valid"] is True
    with pytest.raises(RuntimeError, match="exactly one summary"):
        _m_summary_record(report, 3, 10)


def test_extract_runner_result_uses_final_pointer_record():
    stdout = "noise\n" + json.dumps({"result_path": "/tmp/result.json", "complete": True}) + "\n"
    assert _extract_runner_result(stdout) == {"result_path": "/tmp/result.json", "complete": True}


def test_extract_historical_report_ignores_prior_braces():
    stdout = "compile {noise}\n{\n  \"kind\": \"historical_m_same_input_timing_probe_v1\",\n  \"historical_m_same_input_gate_pass\": true\n}\n"
    value = _extract_historical_report(stdout)
    assert value["historical_m_same_input_gate_pass"] is True
