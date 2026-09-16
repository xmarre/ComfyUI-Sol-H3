"""Run the production mapped-neighbor probe against the exact diagnostic-M oracle.

The candidate and historical M execute in separate fresh subprocesses on the same
GPU/runtime and consume the same frozen E Q/K/V plus M report. The historical
subprocess imports exact diagnostic PR #13 head and receives only geometry already
verified by the production probe against preserved M identities. This runner
persists both raw transcripts and a combined comparison report before pass/fail.

It does not run H3 and does not regenerate R/W/E/M.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PERFORMANCE_BUDGET_FRACTION = 0.05
_RUNTIME_MATCH_FIELDS = (
    "torch",
    "torch_cuda",
    "device_name",
    "compute_capability",
    "cutlass_dsl",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _extract_runner_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("result_path"), str) and "complete" in value:
            return value
    raise RuntimeError("production probe runner did not emit its result-path record")


def _extract_historical_report(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() != "{":
            continue
        candidate = "\n".join(lines[index:]).strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("kind") == "historical_m_same_input_timing_probe_v1":
            return value
    raise RuntimeError("historical M probe did not end with the expected JSON report")


def _candidate_geometry(candidate_report: dict[str, Any]) -> dict[str, Any]:
    geometry = candidate_report.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("matches_preserved_m_geometry") is not True:
        raise RuntimeError("candidate production report lacks preserved-M-matched geometry")
    q_rows = geometry.get("q_rows")
    kv_rows = geometry.get("kv_rows")
    sink_rows = geometry.get("sink_rows")
    if any(type(value) is not int for value in (q_rows, kv_rows, sink_rows)):
        raise RuntimeError("candidate geometry Q/KV/sink rows are not integers")
    if q_rows <= 0 or kv_rows <= 0 or not 0 <= sink_rows <= kv_rows:
        raise RuntimeError("candidate geometry Q/KV/sink rows are invalid")
    query_sha = geometry.get("query_positions_sha256")
    descriptor_sha = geometry.get("diagnostic_descriptor_sha256")
    if not isinstance(query_sha, str) or len(query_sha) != 64:
        raise RuntimeError("candidate geometry query-position digest is invalid")
    if not isinstance(descriptor_sha, str) or len(descriptor_sha) != 64:
        raise RuntimeError("candidate geometry descriptor digest is invalid")
    intervals = geometry.get("mapped_neighbor_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise RuntimeError("candidate geometry is missing mapped-neighbor intervals")
    for item in intervals:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(type(value) is not int for value in item)
            or not 0 <= item[0] < item[1] <= math.ceil(kv_rows / 64)
        ):
            raise RuntimeError(f"candidate geometry contains an invalid mapped-neighbor interval: {item!r}")
    if len(intervals) != math.ceil(q_rows / 64):
        raise RuntimeError("candidate geometry descriptor length does not match its Q64 domain")
    return {
        "q_rows": q_rows,
        "kv_rows": kv_rows,
        "sink_rows": sink_rows,
        "query_positions_sha256": query_sha,
        "mapped_neighbor_intervals": intervals,
        "diagnostic_descriptor_sha256": descriptor_sha,
    }


def _positive_median(timing: Any, label: str) -> float:
    if not isinstance(timing, dict):
        raise RuntimeError(f"{label} timing report is missing")
    value = timing.get("median_cuda_ms")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise RuntimeError(f"{label} median CUDA timing is invalid: {value!r}")
    return float(value)


def evaluate_performance_gate(candidate_timing: dict[str, Any], historical_timing: dict[str, Any]) -> dict[str, Any]:
    candidate = _positive_median(candidate_timing, "candidate mapped")
    historical = _positive_median(historical_timing, "historical M")
    ratio = candidate / historical
    delta_fraction = ratio - 1.0

    candidate_min = float(candidate_timing.get("min_cuda_ms", candidate))
    candidate_max = float(candidate_timing.get("max_cuda_ms", candidate))
    historical_min = float(historical_timing.get("min_cuda_ms", historical))
    historical_max = float(historical_timing.get("max_cuda_ms", historical))
    values = (candidate_min, candidate_max, historical_min, historical_max)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("performance timing range contains invalid values")
    if candidate_min > candidate_max or historical_min > historical_max:
        raise RuntimeError("performance timing range is inverted")

    limit = 1.0 + PERFORMANCE_BUDGET_FRACTION
    if candidate_max / historical_min <= limit:
        observed_range = "clear_pass"
    elif candidate_min / historical_max > limit:
        observed_range = "clear_fail"
    else:
        observed_range = "overlap"
    return {
        "budget_fraction": PERFORMANCE_BUDGET_FRACTION,
        "candidate_median_cuda_ms": candidate,
        "historical_m_median_cuda_ms": historical,
        "candidate_over_historical_ratio": ratio,
        "candidate_over_historical_delta_fraction": delta_fraction,
        "median_budget_pass": bool(ratio <= limit),
        "observed_sample_range_classification": observed_range,
        "candidate_sample_range_cuda_ms": [candidate_min, candidate_max],
        "historical_m_sample_range_cuda_ms": [historical_min, historical_max],
    }


def _persist_process(root: Path, stem: str, completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout_path = root / f"{stem}.stdout.txt"
    stderr_path = root / f"{stem}.stderr.txt"
    _atomic_write(stdout_path, completed.stdout.encode("utf-8"))
    _atomic_write(stderr_path, completed.stderr.encode("utf-8"))
    return {
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256_file(stderr_path),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(report, indent=2, sort_keys=True, default=str).encode("utf-8"))


def _fail_report(report: dict[str, Any], report_path: Path, message: str) -> None:
    report["failure"] = message
    report["complete"] = False
    report["performance_gate_pass"] = False
    _write_report(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-sol-path", type=Path, required=True)
    parser.add_argument("--e-evidence", type=Path, required=True)
    parser.add_argument("--m-report", type=Path, required=True)
    parser.add_argument("--vdn-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=(0, 2, 10), default=10)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error("--warmup and --repeats must be positive")

    root = Path(__file__).resolve().parents[1]
    current_runner = root / "tools" / "run_mapped_neighbor_probe.py"
    historical_probe = root / "tools" / "historical_m_timing_probe.py"
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"mapped-neighbor-performance-{args.block}-{args.group}-{run_stamp}"
    report_path = output_root / f"{stem}.json"
    if report_path.exists():
        raise RuntimeError(f"refusing to overwrite existing performance evidence: {report_path}")

    common = [
        "--e-evidence", str(args.e_evidence.resolve()),
        "--m-report", str(args.m_report.resolve()),
        "--group", str(args.group),
        "--block", str(args.block),
        "--device", args.device,
        "--tau", str(args.tau),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
    ]
    candidate_command = [
        sys.executable,
        str(current_runner),
        "--vdn-path", str(args.vdn_path.resolve()),
        "--output-dir", str(output_root),
        *common,
    ]
    candidate_process = subprocess.run(candidate_command, text=True, capture_output=True, check=False)
    candidate_process_record = _persist_process(output_root, f"{stem}.candidate-runner", candidate_process)

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "production_mapped_neighbor_performance_gate_v1",
        "run_stamp_utc": run_stamp,
        "block_index": args.block,
        "group_index": args.group,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "candidate_command": candidate_command,
        "candidate_process": candidate_process_record,
        "historical_command": None,
        "historical_process": None,
        "candidate_result_path": None,
        "comparison": None,
        "complete": False,
        "performance_gate_pass": False,
    }

    if candidate_process.returncode != 0:
        _fail_report(report, report_path, "candidate production probe failed")
        sys.stdout.write(candidate_process.stdout)
        sys.stderr.write(candidate_process.stderr)
        print(json.dumps({"result_path": str(report_path), "complete": False, "performance_gate_pass": False}))
        raise SystemExit(candidate_process.returncode)

    try:
        candidate_pointer = _extract_runner_result(candidate_process.stdout)
        candidate_result_path = Path(candidate_pointer["result_path"]).resolve()
        candidate_envelope = json.loads(candidate_result_path.read_text(encoding="utf-8"))
        if candidate_pointer.get("complete") is not True or candidate_envelope.get("complete") is not True:
            raise RuntimeError("candidate production probe did not complete its same-input gate")
        candidate_report = candidate_envelope.get("child_report")
        if not isinstance(candidate_report, dict) or candidate_report.get("production_same_input_gate_pass") is not True:
            raise RuntimeError("candidate production probe result is missing its passed child report")
        candidate_geometry = _candidate_geometry(candidate_report)
        report["candidate_result_path"] = str(candidate_result_path)
        report["candidate_result_sha256"] = _sha256_file(candidate_result_path)
        report["candidate_verified_geometry"] = candidate_geometry
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        message = f"candidate production result could not be validated: {exc}"
        _fail_report(report, report_path, message)
        raise RuntimeError(message) from exc

    historical_command = [
        sys.executable,
        str(historical_probe),
        "--historical-sol-path", str(args.historical_sol_path.resolve()),
        "--mapped-neighbor-intervals-json",
        json.dumps(candidate_geometry["mapped_neighbor_intervals"], separators=(",", ":")),
        "--query-positions-sha256", candidate_geometry["query_positions_sha256"],
        "--q-rows", str(candidate_geometry["q_rows"]),
        "--kv-rows", str(candidate_geometry["kv_rows"]),
        "--sink-rows", str(candidate_geometry["sink_rows"]),
        *common,
    ]
    report["historical_command"] = historical_command
    historical_process = subprocess.run(historical_command, text=True, capture_output=True, check=False)
    historical_process_record = _persist_process(output_root, f"{stem}.historical-m", historical_process)
    report["historical_process"] = historical_process_record
    if historical_process.returncode != 0:
        _fail_report(report, report_path, "historical M probe failed")
        sys.stdout.write(historical_process.stdout)
        sys.stderr.write(historical_process.stderr)
        print(json.dumps({"result_path": str(report_path), "complete": False, "performance_gate_pass": False}))
        raise SystemExit(historical_process.returncode)

    try:
        historical_report = _extract_historical_report(historical_process.stdout)
        if historical_report.get("historical_m_same_input_gate_pass") is not True:
            raise RuntimeError("historical M report is not marked passed")

        candidate_runtime = candidate_envelope.get("runtime")
        historical_runtime = historical_report.get("runtime")
        if not isinstance(candidate_runtime, dict) or not isinstance(historical_runtime, dict):
            raise RuntimeError("candidate/historical runtime provenance is incomplete")
        runtime_mismatches = {
            field: [candidate_runtime.get(field), historical_runtime.get(field)]
            for field in _RUNTIME_MATCH_FIELDS
            if candidate_runtime.get(field) != historical_runtime.get(field)
        }
        if runtime_mismatches:
            raise RuntimeError(f"candidate and historical M did not execute on the same runtime/device: {runtime_mismatches}")
        if candidate_report.get("e_evidence_sha256") != historical_report.get("e_evidence_sha256"):
            raise RuntimeError("candidate and historical M did not consume the same E evidence")
        if candidate_report.get("m_report_sha256") != historical_report.get("m_report_sha256"):
            raise RuntimeError("candidate and historical M did not consume the same M report")
        if candidate_geometry["diagnostic_descriptor_sha256"] != historical_report.get("descriptor_sha256"):
            raise RuntimeError("candidate and historical M descriptor identities differ")
        if candidate_geometry["query_positions_sha256"] != historical_report.get("query_positions_sha256"):
            raise RuntimeError("candidate and historical M query-position identities differ")

        candidate_timing = (candidate_report.get("warmed_kernel_timing") or {}).get("mapped")
        historical_timing = historical_report.get("warmed_m_timing")
        comparison = evaluate_performance_gate(candidate_timing, historical_timing)
    except (json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
        message = f"candidate/historical performance comparison could not be validated: {exc}"
        _fail_report(report, report_path, message)
        raise RuntimeError(message) from exc

    report.update(
        {
            "runtime_match_fields": list(_RUNTIME_MATCH_FIELDS),
            "runtime_match": True,
            "e_evidence_sha256": candidate_report.get("e_evidence_sha256"),
            "m_report_sha256": candidate_report.get("m_report_sha256"),
            "candidate_timing": candidate_timing,
            "historical_m_timing": historical_timing,
            "historical_m_checkout": historical_report.get("historical_checkout"),
            "historical_m_contract": historical_report.get("historical_contract"),
            "historical_m_route_counts_match_preserved_m": historical_report.get("route_counts_match_preserved_m"),
            "comparison": comparison,
            "complete": True,
            "performance_gate_pass": comparison["median_budget_pass"],
            "timing_scope": (
                "sequential fresh subprocesses on the same preserved runtime/device; each reports warmed CUDA-event "
                "timing for prepare + mapped SM120 attention on the same frozen E Q/K/V"
            ),
        }
    )
    _write_report(report_path, report)

    sys.stdout.write(candidate_process.stdout)
    sys.stdout.write(historical_process.stdout)
    if candidate_process.stderr:
        sys.stderr.write(candidate_process.stderr)
    if historical_process.stderr:
        sys.stderr.write(historical_process.stderr)
    print(
        json.dumps(
            {
                "result_path": str(report_path),
                "complete": True,
                "performance_gate_pass": report["performance_gate_pass"],
                "candidate_over_historical_delta_fraction": comparison["candidate_over_historical_delta_fraction"],
                "observed_sample_range_classification": comparison["observed_sample_range_classification"],
            },
            sort_keys=True,
        )
    )
    if not report["performance_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
