"""Benchmark the preserved diagnostic-M implementation on saved E Q/K/V.

This helper is intentionally isolated from the production Sol package. It imports
an exact checkout of diagnostic PR #13 head, verifies that checkout and the frozen
E/M artifacts, reconstructs the preserved M descriptor from the durable M route
record, and measures the warmed diagnostic-M path on the saved same-input tensors.

It does not run H3, modify production dispatch, or regenerate R/W/E/M.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

HISTORICAL_M_COMMIT = "b95ad7b3bc7028465547b22fc61300cda53eb110"
HISTORICAL_CONTRACT = "sana-sol-engine-sol-attn-64-rect-sm120-v3"
SANA_REVISION = "2936c47637380842aaa4a4488fac5006cc542b70"
CAPTURE_ID = "234ed062128e43ed8d5ec63e27517b22"
E_FULL_SHA256 = "e25ff53bb3e6f7d6c1780ba9a1bb44288be1b3e095e87166a77b152971e8a6d5"
E_GROUP10_SHA256 = "7abacbb03407a7486329ce05bcb4ed2a2778f9eb090c6f08ec890a658bf3f9ec"
M_JSON_SHA256 = "1914990091be84bf01c382820b1b12e7f73a9886db114ce0f820fced4122b0ba"
EXPECTED_RUNTIME = {
    "torch": "2.10.0+cu130",
    "torch_cuda": "13.0",
    "cutlass_dsl": "4.7.1",
}
WITNESS_GROUPS = (0, 2, 10)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _git_checkout_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        tracked_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"historical M path is not a readable Git checkout: {root}") from exc
    if head != HISTORICAL_M_COMMIT:
        raise RuntimeError(f"historical M checkout is {head}, expected exact PR #13 head {HISTORICAL_M_COMMIT}")
    if tracked_dirty:
        raise RuntimeError("historical M checkout has tracked modifications")
    return {"root": str(root), "head": head, "tracked_dirty": False}


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("historical M timing requires CUDA")
    with torch.cuda.device(device):
        capability = tuple(torch.cuda.get_device_capability())
        if capability != (12, 0):
            raise RuntimeError(f"historical M timing requires SM120, got {capability}")
        identity = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "cutlass_dsl": _package_version("nvidia-cutlass-dsl", "cutlass"),
            "cuda_python": _package_version("cuda-python"),
            "triton": _package_version("triton"),
            "apache_tvm_ffi": _package_version("apache-tvm-ffi"),
        }
    mismatches = [
        f"{name}={identity.get(name)!r} != {expected!r}"
        for name, expected in EXPECTED_RUNTIME.items()
        if identity.get(name) != expected
    ]
    if mismatches:
        raise RuntimeError("historical M runtime differs from the preserved comparison runtime: " + "; ".join(mismatches))
    return identity


def _iter_records(value: Any, kind: str):
    if torch.is_tensor(value):
        return
    if isinstance(value, dict):
        if value.get("kind") == kind:
            yield value
        for item in value.values():
            yield from _iter_records(item, kind)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_records(item, kind)


def _single_record(value: Any, kind: str, block_index: int, group_index: int, label: str) -> dict[str, Any]:
    matches = [
        item
        for item in _iter_records(value, kind)
        if item.get("block_index") == block_index and item.get("group_index") == group_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one {kind} for block {block_index}/group {group_index}; found {len(matches)}"
        )
    return matches[0]


def _operator_witness(payload: Any, block_index: int, group_index: int) -> dict[str, Any]:
    return _single_record(payload, "operator_witness", block_index, group_index, "E evidence")


def _m_route_record(report: dict[str, Any], block_index: int, group_index: int) -> dict[str, Any]:
    return _single_record(report, "mapped_neighbor_route", block_index, group_index, "M durable JSON")


def _m_summary_record(report: dict[str, Any], block_index: int, group_index: int) -> dict[str, Any]:
    witnesses = report.get("operator_witnesses")
    records = witnesses.get("reports") if isinstance(witnesses, dict) else None
    if not isinstance(records, list):
        raise RuntimeError("M JSON is missing operator_witnesses.reports")
    matches = [
        item
        for item in records
        if isinstance(item, dict)
        and item.get("block_index") == block_index
        and item.get("group_index") == group_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"M JSON must contain exactly one summary for block {block_index}/group {group_index}; found {len(matches)}"
        )
    return matches[0]


def _descriptor_sha256(intervals: tuple[tuple[int, int], ...]) -> str:
    return hashlib.sha256(json.dumps(intervals, separators=(",", ":")).encode("ascii")).hexdigest()


def _load_evidence(path: Path) -> tuple[Any, str]:
    actual = _sha256_file(path)
    if actual not in {E_FULL_SHA256, E_GROUP10_SHA256}:
        raise RuntimeError(f"E evidence SHA-256 is not an approved preserved artifact: {actual}")
    return torch.load(path, map_location="cpu", weights_only=False), actual


def _load_m_report(path: Path) -> tuple[dict[str, Any], str]:
    actual = _sha256_file(path)
    if actual != M_JSON_SHA256:
        raise RuntimeError(f"M JSON SHA-256 mismatch: {actual} != {M_JSON_SHA256}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("capture_id") != CAPTURE_ID:
        raise RuntimeError("M JSON is not the preserved controlled capture")
    return value, actual


def _timings(function, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {
        "median_cuda_ms": statistics.median(values),
        "min_cuda_ms": min(values),
        "max_cuda_ms": max(values),
        "samples_cuda_ms": values,
    }


def _validate_preserved_records(route_record: dict[str, Any], summary_record: dict[str, Any]) -> None:
    if summary_record.get("valid") is not True:
        raise RuntimeError("preserved M summary is not marked valid")
    for name in (
        "block_index",
        "group_index",
        "descriptor_sha256",
        "query_positions_sha256",
        "original_selected_pairs",
        "added_selected_pairs",
        "effective_selected_pairs",
        "total_block_pairs",
        "exact_work_increase_fraction",
    ):
        if route_record.get(name) != summary_record.get(name):
            raise RuntimeError(
                f"preserved M durable route and summary disagree for {name}: "
                f"{route_record.get(name)!r} != {summary_record.get(name)!r}"
            )
    if route_record.get("additive_only") is not True or route_record.get("restricted_domain_unchanged") is not True:
        raise RuntimeError("preserved M durable route does not assert additive-only restricted-domain semantics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-sol-path", type=Path, required=True)
    parser.add_argument("--e-evidence", type=Path, required=True)
    parser.add_argument("--m-report", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=WITNESS_GROUPS, default=10)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error("--warmup and --repeats must be positive")
    if not math.isfinite(args.tau) or args.tau <= 0.0:
        parser.error("--tau must be a finite positive value")

    historical_root = args.historical_sol_path.resolve()
    checkout = _git_checkout_identity(historical_root)
    device = torch.device(args.device)
    runtime = _runtime_identity(device)

    # Import only after the exact historical checkout and runtime have passed.
    sys.path.insert(0, str(historical_root))
    try:
        from sol_h3 import first_high_mapped_neighbor_diagnostic as historical_m
        from sol_h3.provenance import CONTRACT, REVISION, verify_source
    finally:
        sys.path.pop(0)
    module_path = Path(historical_m.__file__).resolve()
    if historical_root not in module_path.parents:
        raise RuntimeError(f"historical M module resolved outside --historical-sol-path: {module_path}")
    if CONTRACT != HISTORICAL_CONTRACT or REVISION != SANA_REVISION:
        raise RuntimeError(
            f"historical M source contract differs from preserved M: contract={CONTRACT!r}, revision={REVISION!r}"
        )
    vendor = verify_source()
    if vendor.get("contract") != HISTORICAL_CONTRACT or vendor.get("revision") != SANA_REVISION:
        raise RuntimeError("historical M packaged Sana manifest differs from the preserved source contract")

    e_payload, e_sha = _load_evidence(args.e_evidence)
    m_report, m_sha = _load_m_report(args.m_report)
    witness = _operator_witness(e_payload, args.block, args.group)
    route_record = _m_route_record(m_report, args.block, args.group)
    summary_record = _m_summary_record(m_report, args.block, args.group)
    _validate_preserved_records(route_record, summary_record)

    intervals_raw = route_record.get("mapped_neighbor_intervals")
    if not isinstance(intervals_raw, list):
        raise RuntimeError("preserved M durable route is missing mapped_neighbor_intervals")
    intervals = tuple((int(item[0]), int(item[1])) for item in intervals_raw)
    descriptor_sha = _descriptor_sha256(intervals)
    if route_record.get("descriptor_sha256") != descriptor_sha:
        raise RuntimeError("preserved M descriptor digest does not match its interval payload")

    q_cpu, k_cpu, v_cpu = (witness.get(name) for name in ("q", "k", "v"))
    if not all(torch.is_tensor(value) and value.device.type == "cpu" for value in (q_cpu, k_cpu, v_cpu)):
        raise RuntimeError("E operator witness does not contain preserved CPU Q/K/V")
    if q_cpu.ndim != 3 or k_cpu.ndim != 3 or v_cpu.ndim != 3 or k_cpu.shape != v_cpu.shape:
        raise RuntimeError("E operator witness Q/K/V geometry is invalid")
    q_rows = int(q_cpu.shape[0])
    kv_rows = int(k_cpu.shape[0])
    if q_rows != int(route_record.get("q_rows", -1)) or kv_rows != int(route_record.get("kv_rows", -1)):
        raise RuntimeError("E Q/KV rows differ from the preserved M durable route")
    sink_rows = int(witness.get("original_sink_rows", -1))
    if sink_rows != int(route_record.get("original_sink_rows", -2)):
        raise RuntimeError("E sink rows differ from the preserved M durable route")
    query_sha = route_record.get("query_positions_sha256")
    if not isinstance(query_sha, str) or len(query_sha) != 64:
        raise RuntimeError("preserved M query-position digest is invalid")

    with torch.cuda.device(device):
        q = q_cpu.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous()
        k = k_cpu.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous()
        v = v_cpu.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous()
        scale = float(q.shape[-1] ** -0.5)

        torch.cuda.synchronize()
        cold_started = time.perf_counter()
        output, _lse, kc, _vc, threshold, qbar = historical_m._mapped_launch(
            q,
            k,
            v,
            tau=args.tau,
            scale=scale,
            sink_rows=sink_rows,
            intervals=intervals,
        )
        torch.cuda.synchronize()
        cold_wall_ms = (time.perf_counter() - cold_started) * 1000.0
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("historical M produced non-finite output")

        group = historical_m.MappedGroup(
            block_index=args.block,
            group_index=args.group,
            q_rows=q_rows,
            kv_rows=kv_rows,
            original_sink_rows=sink_rows,
            scale=scale,
            intervals=intervals,
            query_positions_sha256=query_sha,
            options={},
        )
        route = historical_m._route_evidence(group, qbar, kc, threshold)
        for name in (
            "original_selected_pairs",
            "mapped_candidate_pairs",
            "added_selected_pairs",
            "effective_selected_pairs",
            "total_block_pairs",
            "descriptor_sha256",
            "query_positions_sha256",
        ):
            expected = route_record.get(name)
            if route.get(name) != expected:
                raise RuntimeError(f"historical M same-input route mismatch for {name}: {route.get(name)!r} != {expected!r}")
        fraction = route.get("exact_work_increase_fraction")
        expected_fraction = route_record.get("exact_work_increase_fraction")
        if (
            type(fraction) is not float
            or type(expected_fraction) is not float
            or not math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1.0e-15)
        ):
            raise RuntimeError(
                f"historical M exact-work fraction differs from preserved M: {fraction} != {expected_fraction}"
            )

        def historical_call():
            return historical_m._mapped_launch(
                q,
                k,
                v,
                tau=args.tau,
                scale=scale,
                sink_rows=sink_rows,
                intervals=intervals,
            )[0]

        timing = _timings(historical_call, args.warmup, args.repeats)

    report = {
        "kind": "historical_m_same_input_timing_probe_v1",
        "capture_id": CAPTURE_ID,
        "block_index": args.block,
        "group_index": args.group,
        "historical_checkout": checkout,
        "historical_contract": HISTORICAL_CONTRACT,
        "sana_revision": SANA_REVISION,
        "runtime": runtime,
        "e_evidence_sha256": e_sha,
        "m_report_sha256": m_sha,
        "q_shape": list(q.shape),
        "kv_shape": list(k.shape),
        "tau": args.tau,
        "scale": scale,
        "sink_rows": sink_rows,
        "descriptor_sha256": descriptor_sha,
        "query_positions_sha256": query_sha,
        "route_counts_match_preserved_m": True,
        "route": route,
        "cold_compile_and_first_call_wall_ms": cold_wall_ms,
        "warmed_m_timing": timing,
        "output_sha256": _tensor_sha256(output),
        "timing_scope": (
            "historical diagnostic-M prepare + compile-time-selector SM120 attention on preserved E Q/K/V; "
            "warmed samples exclude first compile"
        ),
        "historical_m_same_input_gate_pass": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
