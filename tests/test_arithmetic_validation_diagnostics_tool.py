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


def _add_replay(summary, target="partitioned_suffix", *, first_compile_misses=1):
    arms = (
        "first_executable_fresh_validation",
        "primed_executable_fresh_validation",
        "primed_executable_retained_validation",
    )
    summary["replay_diagnostics"] = {
        "enabled": True,
        "configuration_error": None,
        "claimed_targets": [target],
        "completed_targets": [target],
        "errors": 0,
        "total_host_wall_s": 0.6,
        "reports": [
            {
                "target": target,
                "context": {"flow_stage_id": "low-1"},
                "arms": [
                    {
                        "arm": arms[0],
                        "proof_hit": False,
                        "gate_performed": True,
                        "compile_misses": first_compile_misses,
                    },
                    {
                        "arm": arms[1],
                        "proof_hit": False,
                        "gate_performed": True,
                        "compile_misses": 0,
                    },
                    {
                        "arm": arms[2],
                        "proof_hit": True,
                        "gate_performed": False,
                        "compile_misses": 0,
                    },
                ],
                "rng_restored": True,
                "provider_history_reentered": False,
                "vdn_runtime_reentered": False,
                "bsa_pool_reentered": False,
                "output_policy": "discarded",
                "host_wall_s": 0.6,
                "error": None,
            }
        ],
    }

    for arm in arms:
        summary["cuda_diagnostics"]["details"].append(
            {
                "kind": "replay_production_sparse",
                "ordinal": 1,
                "device": "cuda:0",
                "context": {"replay_target": target, "replay_arm": arm},
                "initial_stream_drain_host_wall_s": None,
                "cuda_event_ms": {
                    "prepare": 0.5,
                    "compiled_dispatch": 1.0,
                    "production_call": 1.5,
                },
            }
        )
    for arm in arms[:2]:
        summary["cuda_diagnostics"]["details"].append(
            {
                "kind": "replay_arithmetic_gate",
                "ordinal": 1,
                "device": "cuda:0",
                "context": {"replay_target": target, "replay_arm": arm},
                "initial_stream_drain_host_wall_s": 0.01,
                "cuda_event_ms": {
                    "prepare": 0.5,
                    "compiled_dispatch": 1.0,
                    "all_selected_call": 1.5,
                    "dense_reference": 2.0,
                    "error_reduction": 2.5,
                },
            }
        )
    summary["cuda_diagnostics"]["resolved_samples"] = len(
        summary["cuda_diagnostics"]["details"]
    )
    summary["cuda_diagnostics"]["selected_samples"] = len(
        summary["cuda_diagnostics"]["details"]
    )
    return summary


def test_replay_evidence_distinguishes_executable_and_proof_lifetimes():
    summary = _add_replay(_summary())
    report = validate_cuda_attribution(
        summary,
        required_replay_targets=("partitioned_suffix",),
        require_replay_cold_miss=True,
    )
    replay = report["replay_reports"]["partitioned_suffix"]
    assert replay["first_compile_misses"] == 1
    assert replay["primed_compile_misses"] == 0
    assert replay["retained_proof_hit"] is True


def test_replay_cold_requirement_rejects_already_primed_first_arm():
    summary = _add_replay(_summary(), first_compile_misses=0)
    with pytest.raises(DiagnosticEvidenceError, match="first executable compilation"):
        validate_cuda_attribution(
            summary,
            required_replay_targets=("partitioned_suffix",),
            require_replay_cold_miss=True,
        )


def test_replay_rejects_missing_cuda_component():
    summary = _add_replay(_summary())
    replay_gate = next(
        item
        for item in summary["cuda_diagnostics"]["details"]
        if item["kind"] == "replay_arithmetic_gate"
    )
    del replay_gate["cuda_event_ms"]["dense_reference"]
    with pytest.raises(DiagnosticEvidenceError, match="missing required spans"):
        validate_cuda_attribution(
            summary,
            required_replay_targets=("partitioned_suffix",),
        )
