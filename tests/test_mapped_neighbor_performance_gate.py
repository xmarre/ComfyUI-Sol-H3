import json

import pytest

from tools.run_mapped_neighbor_performance_gate import (
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


def test_extract_runner_result_uses_final_pointer_record():
    stdout = "noise\n" + json.dumps({"result_path": "/tmp/result.json", "complete": True}) + "\n"
    assert _extract_runner_result(stdout) == {"result_path": "/tmp/result.json", "complete": True}


def test_extract_historical_report_ignores_prior_braces():
    stdout = "compile {noise}\n{\n  \"kind\": \"historical_m_same_input_timing_probe_v1\",\n  \"historical_m_same_input_gate_pass\": true\n}\n"
    value = _extract_historical_report(stdout)
    assert value["historical_m_same_input_gate_pass"] is True
