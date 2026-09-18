import json

import pytest

from tools.check_arithmetic_validation_diagnostics import (
    DiagnosticEvidenceError,
    extract_sol_summaries,
    validate_cuda_attribution,
)


def _summary(*, compile_misses=1):
    gate = {
        "kind": "arithmetic_gate",
        "ordinal": 1,
        "device": "cuda:0",
        "context": {"flow_stage_id": "low-1"},
        "initial_stream_drain_host_wall_s": 0.01,
        "cuda_event_ms": {
            "prepare": 1.0,
            "compiled_dispatch": 2.0,
            "all_selected_call": 3.0,
            "dense_reference": 4.0,
            "error_reduction": 5.0,
        },
    }
    production = {
        "kind": "production_sparse",
        "ordinal": 1,
        "device": "cuda:0",
        "context": {"flow_stage_id": "low-1"},
        "initial_stream_drain_host_wall_s": None,
        "cuda_event_ms": {
            "prepare": 0.5,
            "compiled_dispatch": 1.5,
            "production_call": 2.0,
        },
    }
    return {
        "success": True,
        "validation": {
            "validation_generation": 0,
            "hits": 2,
            "misses": 1,
            "failures": 0,
            "compile_hits": 2,
            "compile_misses": compile_misses,
            "gate_total_s": 0.2,
            "production_host_wall_s": 0.3,
        },
        "runtime_lease": {
            "request_id": "sol-h3-1",
            "source_verify_count": 1,
            "source_generation": "source",
            "implementation_generation": "impl",
        },
        "cuda_diagnostics": {
            "enabled": True,
            "max_samples": 128,
            "selected_samples": 2,
            "resolved_samples": 2,
            "overflow": {},
            "resolve_sync_wall_s": 0.01,
            "resolution_error": None,
            "details": [gate, production],
        },
    }


def test_extract_and_validate_cuda_attribution():
    summary = _summary()
    text = "noise\nINFO comfy.sol_h3 Sol-H3 " + json.dumps(summary) + "\n"
    assert extract_sol_summaries(text) == [summary]
    report = validate_cuda_attribution(summary, require_compile_miss=True)
    assert report["compile_misses"] == 1
    assert report["cuda_event_ms_totals"]["dense_reference"] == 4.0
    assert report["resolved_samples"] == 2


def test_primed_gate_rejects_compiler_miss():
    with pytest.raises(DiagnosticEvidenceError, match="compiler misses"):
        validate_cuda_attribution(_summary(compile_misses=1), require_no_compile_miss=True)


def test_missing_gate_span_fails_closed():
    summary = _summary()
    del summary["cuda_diagnostics"]["details"][0]["cuda_event_ms"]["error_reduction"]
    with pytest.raises(DiagnosticEvidenceError, match="missing required spans"):
        validate_cuda_attribution(summary)
