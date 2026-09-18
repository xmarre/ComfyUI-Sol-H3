#!/usr/bin/env python3
"""Validate one Sol-H3 CUDA-attribution run from a saved ComfyUI log.

This checker validates instrumentation evidence only. It does not certify output
quality, a performance advantage, or equivalence to another run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class DiagnosticEvidenceError(RuntimeError):
    pass


def extract_sol_summaries(text: str) -> list[dict]:
    marker = "Sol-H3 "
    summaries = []
    for line in text.splitlines():
        at = line.find(marker)
        if at < 0:
            continue
        payload = line[at + len(marker) :].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "validation" in value and "runtime_lease" in value:
            summaries.append(value)
    if not summaries:
        raise DiagnosticEvidenceError("no Sol-H3 request summary found in the supplied log")
    return summaries


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticEvidenceError(f"{name} is missing or not numeric")
    return float(value)


def validate_cuda_attribution(
    summary: dict,
    *,
    require_compile_miss: bool = False,
    require_no_compile_miss: bool = False,
    required_replay_targets: tuple[str, ...] = (),
    require_replay_cold_miss: bool = False,
) -> dict:
    diagnostics = summary.get("cuda_diagnostics")
    validation = summary.get("validation")
    lease = summary.get("runtime_lease")
    if not isinstance(diagnostics, dict) or not diagnostics.get("enabled"):
        raise DiagnosticEvidenceError("SOL_H3_CUDA_DIAGNOSTICS was not enabled")
    if diagnostics.get("resolution_error"):
        raise DiagnosticEvidenceError(
            f"CUDA diagnostic resolution failed: {diagnostics['resolution_error']}"
        )
    if not isinstance(validation, dict) or not isinstance(lease, dict):
        raise DiagnosticEvidenceError("validation/runtime lease summary is missing")
    if lease.get("source_verify_count") != 1:
        raise DiagnosticEvidenceError("request did not report exactly one source verification")

    details = diagnostics.get("details")
    if not isinstance(details, list) or not details:
        raise DiagnosticEvidenceError("CUDA diagnostics contain no resolved detail samples")
    if diagnostics.get("resolved_samples") != len(details):
        raise DiagnosticEvidenceError("resolved CUDA sample count disagrees with detail records")
    if len(details) > int(diagnostics.get("max_samples", 0)):
        raise DiagnosticEvidenceError("CUDA diagnostics exceeded their declared detail bound")

    gate_details = [
        item
        for item in details
        if item.get("kind") in {"arithmetic_gate", "partitioned_arithmetic_gate"}
    ]
    production_details = [
        item
        for item in details
        if item.get("kind") in {"production_sparse", "partitioned_production_sparse"}
    ]
    if not gate_details:
        raise DiagnosticEvidenceError("no arithmetic-gate CUDA sample was resolved")
    if not production_details:
        raise DiagnosticEvidenceError("no production sparse CUDA sample was resolved")

    gate_required = {
        "prepare",
        "compiled_dispatch",
        "all_selected_call",
        "dense_reference",
        "error_reduction",
    }
    production_required = {"prepare", "compiled_dispatch", "production_call"}
    for item in gate_details:
        spans = item.get("cuda_event_ms")
        if not isinstance(spans, dict) or not gate_required.issubset(spans):
            missing = sorted(gate_required - set(spans or {}))
            raise DiagnosticEvidenceError(
                f"gate CUDA sample is missing required spans: {missing}"
            )
        if item.get("initial_stream_drain_host_wall_s") is None:
            raise DiagnosticEvidenceError("gate CUDA sample omitted its clean-start drain receipt")
    for item in production_details:
        spans = item.get("cuda_event_ms")
        if not isinstance(spans, dict) or not production_required.issubset(spans):
            missing = sorted(production_required - set(spans or {}))
            raise DiagnosticEvidenceError(
                f"production CUDA sample is missing required spans: {missing}"
            )

    compile_misses = int(validation.get("compile_misses", 0))
    compile_hits = int(validation.get("compile_hits", 0))
    if require_compile_miss and compile_misses < 1:
        raise DiagnosticEvidenceError("run did not report the required compiler miss")
    if require_no_compile_miss and compile_misses != 0:
        raise DiagnosticEvidenceError(
            f"primed run reported {compile_misses} compiler misses"
        )

    cuda_totals = {}
    for item in details:
        for name, value in (item.get("cuda_event_ms") or {}).items():
            cuda_totals[name] = cuda_totals.get(name, 0.0) + _number(
                value, f"cuda_event_ms.{name}"
            )

    replay_reports = {}
    if required_replay_targets:
        replay = summary.get("replay_diagnostics")
        if not isinstance(replay, dict) or not replay.get("enabled"):
            raise DiagnosticEvidenceError("SOL_H3_REPLAY_DIAGNOSTICS was not enabled")
        if replay.get("configuration_error"):
            raise DiagnosticEvidenceError(
                f"replay diagnostics configuration failed: {replay['configuration_error']}"
            )
        reports = replay.get("reports")
        if not isinstance(reports, list):
            raise DiagnosticEvidenceError("replay diagnostics contain no report list")
        by_target = {
            item.get("target"): item
            for item in reports
            if isinstance(item, dict) and isinstance(item.get("target"), str)
        }
        expected_arms = (
            "first_executable_fresh_validation",
            "primed_executable_fresh_validation",
            "primed_executable_retained_validation",
        )
        replay_gate_required = {
            "prepare",
            "compiled_dispatch",
            "all_selected_call",
            "dense_reference",
            "error_reduction",
        }
        replay_production_required = {
            "prepare",
            "compiled_dispatch",
            "production_call",
        }
        replay_cuda = [
            item
            for item in details
            if isinstance(item, dict)
            and item.get("kind") in {"replay_arithmetic_gate", "replay_production_sparse"}
        ]
        for target in required_replay_targets:
            report = by_target.get(target)
            if not isinstance(report, dict):
                raise DiagnosticEvidenceError(f"required replay target {target!r} was not captured")
            if report.get("error"):
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} failed: {report['error']}"
                )
            if report.get("rng_restored") is not True:
                raise DiagnosticEvidenceError(f"replay target {target!r} did not restore RNG state")
            for name in (
                "provider_history_reentered",
                "vdn_runtime_reentered",
                "bsa_pool_reentered",
            ):
                if report.get(name) is not False:
                    raise DiagnosticEvidenceError(
                        f"replay target {target!r} did not preserve {name}"
                    )
            if report.get("output_policy") != "discarded":
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} did not discard diagnostic output"
                )
            arms = report.get("arms")
            if not isinstance(arms, list) or tuple(
                arm.get("arm") for arm in arms if isinstance(arm, dict)
            ) != expected_arms:
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} has incomplete arm ordering"
                )
            first, primed, retained = arms
            if first.get("gate_performed") is not True or first.get("proof_hit") is not False:
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} first fresh validation did not gate"
                )
            if primed.get("gate_performed") is not True or primed.get("proof_hit") is not False:
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} primed fresh validation did not gate"
                )
            if retained.get("gate_performed") is not False or retained.get("proof_hit") is not True:
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} retained validation did not reuse proof"
                )
            if int(primed.get("compile_misses", 0)) != 0:
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} primed fresh arm compiled unexpectedly"
                )
            if int(retained.get("compile_misses", 0)) != 0:
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} retained-proof arm compiled unexpectedly"
                )
            if require_replay_cold_miss and int(first.get("compile_misses", 0)) < 1:
                raise DiagnosticEvidenceError(
                    f"replay target {target!r} did not observe first executable compilation"
                )

            for arm_name in expected_arms:
                matching_production = [
                    item
                    for item in replay_cuda
                    if item.get("kind") == "replay_production_sparse"
                    and (item.get("context") or {}).get("replay_target") == target
                    and (item.get("context") or {}).get("replay_arm") == arm_name
                ]
                if len(matching_production) != 1:
                    raise DiagnosticEvidenceError(
                        f"replay target {target!r} arm {arm_name!r} lacks one production CUDA sample"
                    )
                spans = matching_production[0].get("cuda_event_ms")
                if not isinstance(spans, dict) or not replay_production_required.issubset(spans):
                    missing = sorted(replay_production_required - set(spans or {}))
                    raise DiagnosticEvidenceError(
                        f"replay production sample is missing required spans: {missing}"
                    )

            for arm_name in expected_arms[:2]:
                matching_gate = [
                    item
                    for item in replay_cuda
                    if item.get("kind") == "replay_arithmetic_gate"
                    and (item.get("context") or {}).get("replay_target") == target
                    and (item.get("context") or {}).get("replay_arm") == arm_name
                ]
                if len(matching_gate) != 1:
                    raise DiagnosticEvidenceError(
                        f"replay target {target!r} arm {arm_name!r} lacks one gate CUDA sample"
                    )
                gate_item = matching_gate[0]
                spans = gate_item.get("cuda_event_ms")
                if not isinstance(spans, dict) or not replay_gate_required.issubset(spans):
                    missing = sorted(replay_gate_required - set(spans or {}))
                    raise DiagnosticEvidenceError(
                        f"replay gate sample is missing required spans: {missing}"
                    )
                if gate_item.get("initial_stream_drain_host_wall_s") is None:
                    raise DiagnosticEvidenceError(
                        "replay gate CUDA sample omitted its clean-start drain receipt"
                    )
            replay_reports[target] = {
                "first_compile_misses": int(first.get("compile_misses", 0)),
                "primed_compile_misses": int(primed.get("compile_misses", 0)),
                "retained_proof_hit": bool(retained.get("proof_hit")),
                "host_wall_s": _number(report.get("host_wall_s", 0.0), "replay.host_wall_s"),
            }

    return {
        "success": bool(summary.get("success")),
        "request_id": lease.get("request_id"),
        "source_generation": lease.get("source_generation"),
        "implementation_generation": lease.get("implementation_generation"),
        "validation_generation": validation.get("validation_generation"),
        "validation_hits": int(validation.get("hits", 0)),
        "validation_misses": int(validation.get("misses", 0)),
        "validation_failures": int(validation.get("failures", 0)),
        "compile_hits": compile_hits,
        "compile_misses": compile_misses,
        "gate_total_s": _number(validation.get("gate_total_s", 0.0), "gate_total_s"),
        "production_host_wall_s": _number(
            validation.get("production_host_wall_s", 0.0),
            "production_host_wall_s",
        ),
        "selected_samples": int(diagnostics.get("selected_samples", 0)),
        "resolved_samples": int(diagnostics.get("resolved_samples", 0)),
        "resolve_sync_wall_s": _number(
            diagnostics.get("resolve_sync_wall_s", 0.0),
            "resolve_sync_wall_s",
        ),
        "cuda_event_ms_totals": cuda_totals,
        "replay_reports": replay_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-compile-miss", action="store_true")
    group.add_argument("--require-no-compile-miss", action="store_true")
    parser.add_argument(
        "--require-replay-target",
        action="append",
        default=[],
        choices=("ordinary_low", "ordinary_continuation_high", "partitioned_suffix"),
        help="Require and validate one same-input replay target; may be repeated.",
    )
    parser.add_argument(
        "--require-replay-cold-miss",
        action="store_true",
        help="Require first replay arm to observe executable compilation.",
    )
    parser.add_argument(
        "--request-index",
        type=int,
        default=-1,
        help="Sol-H3 summary index in the log; default is the final request",
    )
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    summaries = extract_sol_summaries(text)
    try:
        summary = summaries[args.request_index]
    except IndexError as exc:
        raise SystemExit(
            f"request index {args.request_index} is outside {len(summaries)} summaries"
        ) from exc
    try:
        report = validate_cuda_attribution(
            summary,
            require_compile_miss=args.require_compile_miss,
            require_no_compile_miss=args.require_no_compile_miss,
            required_replay_targets=tuple(args.require_replay_target),
            require_replay_cold_miss=args.require_replay_cold_miss,
        )
    except DiagnosticEvidenceError as exc:
        raise SystemExit(f"Sol-H3 CUDA attribution: FAIL: {exc}") from exc
    print(json.dumps({"status": "pass", **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
