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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-compile-miss", action="store_true")
    group.add_argument("--require-no-compile-miss", action="store_true")
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
        )
    except DiagnosticEvidenceError as exc:
        raise SystemExit(f"Sol-H3 CUDA attribution: FAIL: {exc}") from exc
    print(json.dumps({"status": "pass", **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
