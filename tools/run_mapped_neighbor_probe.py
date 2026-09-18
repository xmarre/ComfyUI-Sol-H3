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
    "sol_h3/mapped_neighbors.py": "834323e2bbafa5f39467b82327e344cf3368a21e",
    "sol_h3/sparse.py": "8a5ab1902ec88c12d122bb8936a9e5d96349a78d",
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
    raise RuntimeError("production mapped-neighbor probe did not emit the expected JSON report")


def _require_child_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    if not _child_report_matches_request(report, args):
        raise RuntimeError("child probe report does not match the requested preserved production gate")


def _child_report_matches_request(report: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        report.get("production_same_input_gate_pass") is True
        and report.get("capture_id") == CAPTURE_ID
        and report.get("block_index") == int(args.block)
        and report.get("group_index") == int(args.group)
        and report.get("tau") == PRESERVED_TAU
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vdn-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--group", type=int, default=10)
    parser.add_argument("--tau", type=float, default=PRESERVED_TAU)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)

    tau = _require_preserved_tau(args.tau)
    sol_root = Path(__file__).resolve().parents[1]
    vdn_root = args.vdn_root.resolve()
    sol_sources = _source_gate(sol_root, EXPECTED_SOL_BLOBS, "sol")
    vdn_sources = _source_gate(vdn_root, EXPECTED_VDN_BLOBS, "vdn")
    device = torch.device(args.device)
    runtime = _runtime_identity(device)
    _require_preserved_runtime(runtime)
    verify_source()

    command = [
        sys.executable,
        str(sol_root / "tools" / "mapped_neighbor_probe.py"),
        "--device",
        str(device),
        "--block",
        str(args.block),
        "--group",
        str(args.group),
        "--tau",
        str(tau),
        "--repeats",
        str(args.repeats),
    ]
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=sol_root, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc)
    report = _extract_final_json(completed.stdout)
    _require_child_report(report, args)

    envelope = {
        "kind": "production_mapped_neighbor_probe_runner_v1",
        "capture_id": CAPTURE_ID,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "returncode": completed.returncode,
        "command": command,
        "sol_contract": CONTRACT,
        "sol_revision": REVISION,
        "sol_sources": sol_sources,
        "vdn_sources": vdn_sources,
        "runtime": runtime,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": report,
    }
    timestamp = finished.strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"production_mapped_neighbor_probe_{timestamp}.json"
    _write_json_atomic(path, envelope)
    print(path)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
