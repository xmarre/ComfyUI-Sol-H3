"""Run the mapped-neighbor same-input SM120 probe with durable provenance.

This is the authoritative workstation entry point for the bounded production
operator gate. It fails before CUDA execution if the installed Sol/VDN source
bytes or the preserved comparison runtime differ from the reviewed contract,
then runs ``mapped_neighbor_probe.py`` in a fresh subprocess and persists the
complete stdout/stderr plus one machine-readable result envelope atomically.

It does not execute H3 and does not regenerate R/W/E/M.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sol_h3.provenance import CONTRACT, REVISION, verify_source  # noqa: E402

CAPTURE_ID = "234ed062128e43ed8d5ec63e27517b22"
PRESERVED_TAU = 1.0
EXPECTED_RUNTIME = {
    "torch": "2.10.0+cu130",
    "torch_cuda": "13.0",
    "cutlass_dsl": "4.7.1",
}
EXPECTED_SOL_BLOBS = {
    "sol_h3/mapped_neighbors.py": "6a527beb80b4072cb2b611019655ae50be0195d5",
    "sol_h3/sparse.py": "b37892cab6ed75736f70b23dcd2690a6649b060f",
    "sol_h3/provenance.py": "e3d0e3341d18e21080c33bf0f6fae4b8ebe0e237",
    "sol_h3/sol_manifest.json": "1dfd622bd102aa9f66db00f5dc49aa627560173e",
    "tools/mapped_neighbor_probe.py": "445055a66699e66d33da52242749248b2abcae4d",
}
EXPECTED_VDN_BLOBS = {
    "vdn_h3/query_positions.py": "cadb2ad93da16cdc11365f2c4008527d2ccdbb4d",
    "vdn_h3/softmax_provider.py": "c73caec58f1df9e7338992f40e9f6bffb104358b",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _source_gate(root: Path, expected: dict[str, str], owner: str) -> list[dict[str, Any]]:
    root = root.resolve()
    reports = []
    for relative, expected_blob in sorted(expected.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"{owner} source escaped its reviewed root: {path}") from exc
        if not path.is_file():
            raise RuntimeError(f"{owner} reviewed source is missing: {path}")
        actual_blob = _git_blob_sha(path)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"{owner} source differs from reviewed production bytes: {relative}: "
                f"{actual_blob} != {expected_blob}"
            )
        reports.append(
            {
                "owner": owner,
                "relative_path": relative,
                "path": str(path),
                "git_blob_sha": actual_blob,
                "sha256": _sha256_file(path),
            }
        )
    return reports


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production mapped-neighbor validation requires CUDA")
    with torch.cuda.device(device):
        capability = tuple(torch.cuda.get_device_capability())
        if capability != (12, 0):
            raise RuntimeError(f"production mapped-neighbor validation requires SM120, got {capability}")
        return {
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


def _require_preserved_runtime(identity: dict[str, Any]) -> None:
    mismatches = []
    for name, expected in EXPECTED_RUNTIME.items():
        actual = identity.get(name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} != {expected!r}")
    if mismatches:
        raise RuntimeError(
            "production same-input comparison runtime differs from the preserved M runtime: " + "; ".join(mismatches)
        )


def _require_preserved_tau(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) != PRESERVED_TAU:
        raise ValueError(f"production same-input comparison requires tau={PRESERVED_TAU}, got {value!r}")
    return PRESERVED_TAU


def _extract_final_json(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() != "{":
            continue
        candidate = "\n".join(lines[index:]).strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("kind") == "production_mapped_neighbor_same_input_probe_v1":
            return value
    raise RuntimeError("mapped-neighbor probe stdout did not end with the expected JSON report")


def _child_report_matches_request(report: Any, args: Any) -> bool:
    return bool(
        isinstance(report, dict)
        and report.get("production_same_input_gate_pass") is True
        and report.get("capture_id") == CAPTURE_ID
        and report.get("block_index") == args.block
        and report.get("group_index") == args.group
        and report.get("tau") == args.tau
    )


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


def _command(args, probe: Path) -> list[str]:
    return [
        sys.executable,
        str(probe),
        "--e-evidence",
        str(args.e_evidence.resolve()),
        "--m-report",
        str(args.m_report.resolve()),
        "--vdn-path",
        str(args.vdn_path.resolve()),
        "--group",
        str(args.group),
        "--block",
        str(args.block),
        "--device",
        args.device,
        "--tau",
        str(args.tau),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e-evidence", type=Path, required=True)
    parser.add_argument("--m-report", type=Path, required=True)
    parser.add_argument("--vdn-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=(0, 2, 10), default=10)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tau", type=float, default=PRESERVED_TAU)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error("--warmup and --repeats must be positive")
    try:
        args.tau = _require_preserved_tau(args.tau)
    except ValueError as exc:
        parser.error(str(exc))

    sol_root = Path(__file__).resolve().parents[1]
    probe = sol_root / "tools" / "mapped_neighbor_probe.py"
    source_reports = [
        *_source_gate(sol_root, EXPECTED_SOL_BLOBS, "sol"),
        *_source_gate(args.vdn_path, EXPECTED_VDN_BLOBS, "vdn"),
    ]
    vendor = verify_source()
    if vendor.get("contract") != CONTRACT or vendor.get("revision") != REVISION:
        raise RuntimeError("packaged Sol-Attn provenance differs from the production contract")

    device = torch.device(args.device)
    runtime = _runtime_identity(device)
    _require_preserved_runtime(runtime)

    command = _command(args, probe)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)

    output_root = args.output_dir.resolve()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{CAPTURE_ID}-block{args.block}-group{args.group}-{run_stamp}-production-mapped-neighbor"
    stdout_path = output_root / f"{stem}.stdout.txt"
    stderr_path = output_root / f"{stem}.stderr.txt"
    result_path = output_root / f"{stem}.json"
    for path in (stdout_path, stderr_path, result_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing production probe evidence: {path}")
    _atomic_write(stdout_path, completed.stdout.encode("utf-8"))
    _atomic_write(stderr_path, completed.stderr.encode("utf-8"))

    child_report = None
    parse_error = None
    try:
        child_report = _extract_final_json(completed.stdout)
    except RuntimeError as exc:
        parse_error = str(exc)

    manifest_path = sol_root / "sol_h3" / "sol_manifest.json"
    envelope = {
        "schema_version": 1,
        "kind": "production_mapped_neighbor_probe_run_v1",
        "capture_id": CAPTURE_ID,
        "run_stamp_utc": run_stamp,
        "source_gate_complete": True,
        "sources": source_reports,
        "vendor": {
            "source": vendor.get("source"),
            "revision": vendor.get("revision"),
            "contract": vendor.get("contract"),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "runtime": runtime,
        "preserved_runtime_exact": True,
        "command": command,
        "child_returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256_file(stderr_path),
        "child_report_parse_error": parse_error,
        "child_report": child_report,
        "child_report_matches_request": _child_report_matches_request(child_report, args),
        "complete": bool(
            completed.returncode == 0
            and parse_error is None
            and _child_report_matches_request(child_report, args)
        ),
    }
    encoded = json.dumps(envelope, indent=2, sort_keys=True, default=str).encode("utf-8")
    _atomic_write(result_path, encoded)

    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    print(json.dumps({"result_path": str(result_path), "complete": envelope["complete"]}, sort_keys=True))

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not envelope["complete"]:
        raise RuntimeError(f"production mapped-neighbor probe did not produce a complete matched report: {parse_error}")


if __name__ == "__main__":
    main()
