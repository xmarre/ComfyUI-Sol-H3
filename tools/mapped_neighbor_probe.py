"""Bounded real-SM120 production validation for mapped-neighbor routing.

This tool does not run H3 and does not regenerate R/W/E/M. It consumes the
preserved all-selected E operator witness plus the preserved M JSON report,
reconstructs the current provider-v4 map with the production VDN geometry code,
compiles the current production Sol descriptor, and executes the current packaged
SM120 kernel on the saved same-input Q/K/V.

The primary gate is exact route equality:

    current debug ballots == saved E ordinary route OR current mapped interval

The resulting original/added/effective block-pair counts must also equal the
preserved M record for the same block/group. Output arithmetic is checked against
a bounded two-pass mixed exact/approx reference with the route frozen. Descriptor
values are then changed at fixed shapes/strides to verify that they do not create a
new production CuTe cache specialization.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sol_h3 import sparse  # noqa: E402
from sol_h3.mapped_neighbors import (  # noqa: E402
    compile_descriptor,
    device_descriptor,
    validate_wire_map,
)
from sol_h3.provenance import CONTRACT, REVISION, verify_source  # noqa: E402

CAPTURE_ID = "234ed062128e43ed8d5ec63e27517b22"
E_FULL_SHA256 = "e25ff53bb3e6f7d6c1780ba9a1bb44288be1b3e095e87166a77b152971e8a6d5"
E_GROUP10_SHA256 = "7abacbb03407a7486329ce05bcb4ed2a2778f9eb090c6f08ec890a658bf3f9ec"
M_JSON_SHA256 = "1914990091be84bf01c382820b1b12e7f73a9886db114ce0f820fced4122b0ba"
CONTROL_VIDEO_START = 3101
CONTROL_VIDEO_END = 56349
CONTROL_SEQ_LEN = 56349
CONTROL_FRAMES = 52
CONTROL_TOKENS_PER_FRAME = 1024
CONTROL_RADIUS = 1
CONTROL_CHUNK = 5
CONTROL_ANCHOR_MODE = "both"
WITNESS_GROUPS = (0, 2, 10)
_REFERENCE_KEY_CHUNK = 1024

# These are geometry identities, not model-output expectations. They are stable
# across blocks because VDN's first-high grouped topology is stable across blocks.
CONTROL_DESCRIPTOR_SHA256 = {
    0: "52516cf33458f14e497136273005f2eb1e933a886f7c7222373c2981ae55c3f3",
    1: "897eb1b2cece53b9b75ac6d2809e9591bc7a3b1b707daf0450e886a704f58944",
    **{index: "c040bf1f97ee82405df2c6cf3a39792727f4498d5bcddd9f8faceec4dfa0fd73" for index in range(2, 10)},
    10: "ff5dd9c09ca76caf37ad0146ed3b59f4cd4b3295f2acec146746449c27ea0664",
}
CONTROL_QUERY_POSITION_SHA256 = {
    0: "62dd2d24e1e6f6ca82f8ea2a62b4cdce48db78b47d6456c5fed33a6892bab7c5",
    1: "399da858f762f7174af2f4b0acce9ad38b11a229d6ef8908dc7f0dca39c5242f",
    **{index: "2394f31705e44a458014030988f984822b00b2a73ca7889997a49895ba66f53f" for index in range(2, 10)},
    10: "db8c618ab8a5682af3ac4799b0c2c7a9d73f4585bcd402df9debbeee5eed2bcf",
}
CONTROL_Q_ROWS = {0: 4096, **{index: 5120 for index in range(1, 10)}, 10: 1024}
CONTROL_KV_ROWS = {
    0: 14365,
    1: 19485,
    **{index: 20509 for index in range(2, 9)},
    9: 16413,
    10: 11293,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_sha256(intervals: tuple[tuple[int, int], ...]) -> str:
    # Diagnostic M used the tuple JSON representation with compact separators.
    return hashlib.sha256(json.dumps(intervals, separators=(",", ":")).encode("ascii")).hexdigest()


def _query_positions_from_runs(runs: tuple[tuple[int, int, int], ...], q_rows: int) -> torch.Tensor:
    result = torch.empty(q_rows, dtype=torch.long)
    for q_begin, q_end, kv_begin in runs:
        result[q_begin:q_end] = torch.arange(kv_begin, kv_begin + q_end - q_begin, dtype=torch.long)
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _window_bounds(num_frames: int, radius: int, chunk: int) -> tuple[tuple[int, int], ...]:
    if chunk <= 0:
        return tuple((frame - radius, frame + radius) for frame in range(num_frames))
    return tuple(
        (((frame // chunk) - radius) * chunk, ((frame // chunk) + radius + 1) * chunk - 1)
        for frame in range(num_frames)
    )


def controlled_descriptor(vdn_path: Path, group_index: int):
    """Build the production v4 map/descriptor for the preserved first-high geometry."""
    vdn_path = vdn_path.resolve()
    sys.path.insert(0, str(vdn_path))
    try:
        from vdn_h3 import query_positions
        from vdn_h3.softmax_provider import PROVIDER_API_VERSION
    finally:
        sys.path.pop(0)
    module_path = Path(query_positions.__file__).resolve()
    if vdn_path not in module_path.parents:
        raise RuntimeError(f"vdn_h3.query_positions resolved outside --vdn-path: {module_path}")
    if PROVIDER_API_VERSION != 4:
        raise RuntimeError(f"production probe requires VDN provider API 4, got {PROVIDER_API_VERSION}")
    bounds = _window_bounds(CONTROL_FRAMES, CONTROL_RADIUS, CONTROL_CHUNK)
    geometry = query_positions.describe_window_geometry(
        CONTROL_VIDEO_START,
        CONTROL_VIDEO_END,
        CONTROL_FRAMES,
        CONTROL_TOKENS_PER_FRAME,
        bounds,
        CONTROL_ANCHOR_MODE,
        CONTROL_SEQ_LEN,
    )
    wire = query_positions.bind_query_map(geometry, group_index, "production-mapped-neighbor-probe")
    validated = validate_wire_map(
        wire,
        q_rows=wire[5],
        kv_rows=wire[6],
        sink_rows=wire[7],
    )
    descriptor = compile_descriptor(validated)
    if descriptor is None:
        raise RuntimeError("controlled VDN group unexpectedly collapsed to identity-aligned square routing")
    return geometry, validated, descriptor


def controlled_geometry_report(vdn_path: Path, group_index: int) -> dict[str, Any]:
    _geometry, validated, descriptor = controlled_descriptor(vdn_path, group_index)
    positions = _query_positions_from_runs(validated.runs, validated.q_rows)
    descriptor_sha = _descriptor_sha256(descriptor.intervals)
    positions_sha = _tensor_sha256(positions)
    return {
        "group_index": group_index,
        "q_rows": validated.q_rows,
        "kv_rows": validated.kv_rows,
        "sink_rows": validated.sink_rows,
        "query_position_runs": [list(run) for run in validated.runs],
        "mapped_neighbor_intervals": [list(pair) for pair in descriptor.intervals],
        "diagnostic_descriptor_sha256": descriptor_sha,
        "query_positions_sha256": positions_sha,
        "descriptor_digest": descriptor.descriptor_digest,
        "matches_preserved_m_geometry": bool(
            validated.q_rows == CONTROL_Q_ROWS[group_index]
            and validated.kv_rows == CONTROL_KV_ROWS[group_index]
            and validated.sink_rows == CONTROL_VIDEO_START
            and descriptor_sha == CONTROL_DESCRIPTOR_SHA256[group_index]
            and positions_sha == CONTROL_QUERY_POSITION_SHA256[group_index]
        ),
    }


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


def _operator_witness(payload: Any, block_index: int, group_index: int) -> dict[str, Any]:
    matches = [
        item
        for item in _iter_records(payload, "operator_witness")
        if item.get("block_index") == block_index and item.get("group_index") == group_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"E evidence must contain exactly one operator_witness for block {block_index}/group {group_index}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _m_report_record(report: dict[str, Any], block_index: int, group_index: int) -> dict[str, Any]:
    witnesses = report.get("operator_witnesses")
    records = witnesses.get("reports") if isinstance(witnesses, dict) else None
    if not isinstance(records, list):
        raise RuntimeError("M JSON is missing operator_witnesses.reports")
    matches = [
        item for item in records
        if isinstance(item, dict)
        and item.get("block_index") == block_index
        and item.get("group_index") == group_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"M JSON must contain exactly one report for block {block_index}/group {group_index}; found {len(matches)}"
        )
    return matches[0]


def _decode_route_trace(trace: torch.Tensor, k_blocks: int) -> torch.Tensor:
    if trace.dtype != torch.int32 or trace.ndim != 5 or trace.shape[-1] != 2:
        raise RuntimeError(f"unexpected route-trace layout: shape={tuple(trace.shape)} dtype={trace.dtype}")
    _, q_blocks, heads, groups, _ = trace.shape
    result = torch.zeros((q_blocks, heads, k_blocks), dtype=torch.bool, device=trace.device)
    for group in range(groups):
        for word in range(2):
            raw = trace[0, :, :, group, word].to(torch.int64) & 0xFFFFFFFF
            for bit in range(32):
                block = group * 64 + word * 32 + bit
                if block >= k_blocks:
                    break
                result[:, :, block] = ((raw >> bit) & 1).bool()
    return result


def _mapped_mask(intervals: tuple[tuple[int, int], ...], heads: int, k_blocks: int, device) -> torch.Tensor:
    mask = torch.zeros((len(intervals), heads, k_blocks), dtype=torch.bool, device=device)
    for q_block, (start, end) in enumerate(intervals):
        mask[q_block, :, start:end] = True
    return mask


def _route_counts(old: torch.Tensor, mapped: torch.Tensor) -> dict[str, int]:
    if old.shape != mapped.shape:
        raise RuntimeError("old and mapped route masks have different shapes")
    added = mapped & ~old
    effective = old | mapped
    return {
        "original_selected_pairs": int(old.sum().item()),
        "mapped_candidate_pairs": int(mapped.sum().item()),
        "added_selected_pairs": int(added.sum().item()),
        "effective_selected_pairs": int(effective.sum().item()),
        "total_block_pairs": int(old.numel()),
    }


def _debug_mapped_launch(q, k, v, mapped, *, tau: float, sink_rows: int):
    import cutlass.cute as cute

    from sol_h3._vendor.sol_attn.interface import _sink_block_range, _stream, _to_cute_tensors
    from sol_h3._vendor.sol_attn.preprocess import prepare
    from sol_h3._vendor.sol_attn.sm120 import make_kernel

    scale = q.shape[-1] ** -0.5
    kc, vc, threshold = prepare(
        q,
        k,
        v,
        tau=tau,
        scale=scale,
        thresh_type="diag",
        valid_tokens=int(q.shape[1]),
        valid_kv_tokens=int(k.shape[1]),
    )
    output = torch.empty_like(q)
    q_blocks = (int(q.shape[1]) + 63) // 64
    k_blocks = (int(k.shape[1]) + 63) // 64
    route_groups = (k_blocks + 63) // 64
    trace = torch.zeros(
        (int(q.shape[0]), q_blocks, int(q.shape[2]), route_groups, 2),
        device=q.device,
        dtype=torch.int32,
    )
    sink_start_block, sink_end_block = _sink_block_range(int(k.shape[1]), 0, int(sink_rows))
    operator = make_kernel(debug_route_trace=True, key_bias_enabled=False, mapped_neighbors_enabled=True)
    tensors = [q, k, v, output, kc, vc, threshold, threshold, mapped, trace]
    args = _to_cute_tensors(tensors)
    stream = _stream(q.device)
    compiled = cute.compile(
        operator,
        *args,
        float(scale),
        sink_start_block,
        sink_end_block,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    compiled(
        *args,
        float(scale),
        sink_start_block,
        sink_end_block,
        stream=stream,
    )
    return output, trace, kc, vc


def _mixed_route_reference(q, k, v, kc, vc, routes, *, scale: float) -> torch.Tensor:
    """Bounded two-pass FP32 exact/zeroth-order reference for one frozen route."""
    q_blocks, heads, _k_blocks = routes.shape
    tkv = int(k.shape[1])
    result = torch.empty_like(q, dtype=torch.float32)
    for q_block in range(q_blocks):
        q_start = q_block * 64
        q_stop = min(int(q.shape[1]), q_start + 64)
        for head in range(heads):
            qh = q[0, q_start:q_stop, head].float()
            exact_blocks = routes[q_block, head]
            exact_rows = exact_blocks.repeat_interleave(64)[:tkv].nonzero(as_tuple=False).flatten()
            approx_blocks = (~exact_blocks).nonzero(as_tuple=False).flatten()
            row_max = torch.full((qh.shape[0],), -torch.inf, device=q.device, dtype=torch.float32)
            for offset in range(0, int(exact_rows.numel()), _REFERENCE_KEY_CHUNK):
                rows = exact_rows[offset : offset + _REFERENCE_KEY_CHUNK]
                scores = (qh @ k[0, rows, head].float().T) * float(scale)
                row_max = torch.maximum(row_max, scores.max(dim=1).values)
            for offset in range(0, int(approx_blocks.numel()), _REFERENCE_KEY_CHUNK):
                blocks = approx_blocks[offset : offset + _REFERENCE_KEY_CHUNK]
                scores = (qh @ kc[0, blocks, head].float().T) * float(scale)
                row_max = torch.maximum(row_max, scores.max(dim=1).values)
            denominator = torch.zeros_like(row_max)
            numerator = torch.zeros((qh.shape[0], qh.shape[1]), device=q.device, dtype=torch.float32)
            for offset in range(0, int(exact_rows.numel()), _REFERENCE_KEY_CHUNK):
                rows = exact_rows[offset : offset + _REFERENCE_KEY_CHUNK]
                probabilities = torch.exp((qh @ k[0, rows, head].float().T) * float(scale) - row_max[:, None])
                denominator.add_(probabilities.sum(dim=1))
                numerator.add_(probabilities @ v[0, rows, head].float())
            for offset in range(0, int(approx_blocks.numel()), _REFERENCE_KEY_CHUNK):
                blocks = approx_blocks[offset : offset + _REFERENCE_KEY_CHUNK]
                probabilities = torch.exp((qh @ kc[0, blocks, head].float().T) * float(scale) - row_max[:, None])
                lengths = (tkv - blocks * 64).clamp(min=0, max=64).to(torch.float32)
                denominator.add_((probabilities * lengths[None, :]).sum(dim=1))
                numerator.add_(probabilities @ vc[0, blocks, head].float())
            result[0, q_start:q_stop, head] = numerator / denominator[:, None]
    return result


def _timings(functions: dict[str, Any], warmup: int, repeats: int) -> dict[str, Any]:
    for function in functions.values():
        for _ in range(warmup):
            function()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    names = list(functions)
    for index in range(repeats):
        order = names if index % 2 == 0 else list(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return {
        name: {
            "median_cuda_ms": statistics.median(values),
            "min_cuda_ms": min(values),
            "max_cuda_ms": max(values),
            "samples_cuda_ms": values,
        }
        for name, values in samples.items()
    }


def _git_identity(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _load_evidence(path: Path, expected_hashes: set[str]) -> tuple[Any, str]:
    actual = _sha256_file(path)
    if actual not in expected_hashes:
        raise RuntimeError(f"E evidence SHA-256 is not an approved preserved artifact: {actual}")
    return torch.load(path, map_location="cpu", weights_only=False), actual


def _load_m_report(path: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    actual = _sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"M JSON SHA-256 mismatch: {actual} != {expected_sha}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("capture_id") != CAPTURE_ID:
        raise RuntimeError("M JSON is not the preserved controlled capture")
    return value, actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e-evidence", type=Path, required=True)
    parser.add_argument("--m-report", type=Path, required=True)
    parser.add_argument("--vdn-path", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=WITNESS_GROUPS, default=10)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error("--warmup and --repeats must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production mapped-neighbor validation requires CUDA")
    with torch.cuda.device(device):
        if tuple(torch.cuda.get_device_capability()) != (12, 0):
            raise RuntimeError("production mapped-neighbor validation requires SM120")

        e_payload, e_sha = _load_evidence(args.e_evidence, {E_FULL_SHA256, E_GROUP10_SHA256})
        m_report, m_sha = _load_m_report(args.m_report, M_JSON_SHA256)
        witness = _operator_witness(e_payload, args.block, args.group)
        m_record = _m_report_record(m_report, args.block, args.group)
        geometry_report = controlled_geometry_report(args.vdn_path, args.group)
        if not geometry_report["matches_preserved_m_geometry"]:
            raise RuntimeError(f"current production geometry differs from preserved M: {geometry_report}")

        _geometry, validated, descriptor = controlled_descriptor(args.vdn_path, args.group)
        if m_record.get("descriptor_sha256") != _descriptor_sha256(descriptor.intervals):
            raise RuntimeError("current production descriptor differs from the preserved M descriptor")
        positions = _query_positions_from_runs(validated.runs, validated.q_rows)
        if m_record.get("query_positions_sha256") != _tensor_sha256(positions):
            raise RuntimeError("current production query positions differ from the preserved M mapping")

        q_cpu, k_cpu, v_cpu = (witness.get(name) for name in ("q", "k", "v"))
        if not all(torch.is_tensor(value) and value.device.type == "cpu" for value in (q_cpu, k_cpu, v_cpu)):
            raise RuntimeError("E operator witness does not contain preserved CPU Q/K/V")
        if q_cpu.ndim != 3 or k_cpu.ndim != 3 or v_cpu.ndim != 3 or k_cpu.shape != v_cpu.shape:
            raise RuntimeError("E operator witness Q/K/V geometry is invalid")
        if int(q_cpu.shape[0]) != validated.q_rows or int(k_cpu.shape[0]) != validated.kv_rows:
            raise RuntimeError("E operator witness Q/KV rows differ from the current v4 map")
        if int(witness.get("original_sink_rows", -1)) != validated.sink_rows:
            raise RuntimeError("E witness sink rows differ from the current v4 map")
        saved_trace = witness.get("route_trace")
        if not torch.is_tensor(saved_trace) or saved_trace.device.type != "cpu":
            raise RuntimeError("E operator witness is missing its saved ordinary route trace")

        q = q_cpu.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous()
        k = k_cpu.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous()
        v = v_cpu.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous()
        state = SimpleNamespace(mapped_descriptor_cache=OrderedDict(), mapped_descriptor_bytes=0)
        torch.cuda.synchronize()
        descriptor_started = time.perf_counter()
        mapped = device_descriptor(state, descriptor, device)
        torch.cuda.synchronize()
        descriptor_cold_ms = (time.perf_counter() - descriptor_started) * 1000.0
        warm_started = time.perf_counter()
        mapped_again = device_descriptor(state, descriptor, device)
        descriptor_warm_host_us = (time.perf_counter() - warm_started) * 1_000_000.0
        if mapped_again is not mapped:
            raise RuntimeError("request-local mapped descriptor cache did not reuse its CUDA tensor")
        expected_bytes = 8 * len(descriptor.intervals)
        if state.mapped_descriptor_bytes != expected_bytes:
            raise RuntimeError(
                f"mapped descriptor cache size {state.mapped_descriptor_bytes} != exact {expected_bytes} bytes"
            )

        kernel = sparse.load_kernel(device)
        from sol_h3._vendor.sol_attn import interface

        before_keys = set(interface._compiled)
        torch.cuda.synchronize()
        cold_started = time.perf_counter()
        candidate = kernel(
            q,
            k,
            v,
            tau=args.tau,
            sink_start=0,
            sink_tokens=validated.sink_rows,
            mapped_neighbor_intervals=mapped,
        )
        torch.cuda.synchronize()
        cold_candidate_ms = (time.perf_counter() - cold_started) * 1000.0
        after_first_keys = set(interface._compiled)

        alternate = mapped.clone()
        alternate[0, 0] = 0
        alternate[0, 1] = 1
        kernel(
            q,
            k,
            v,
            tau=args.tau,
            sink_start=0,
            sink_tokens=validated.sink_rows,
            mapped_neighbor_intervals=alternate,
        )
        torch.cuda.synchronize()
        after_second_keys = set(interface._compiled)
        if after_second_keys != after_first_keys:
            raise RuntimeError("changing descriptor values created a new CuTe production specialization")

        debug_output, current_trace, kc, vc = _debug_mapped_launch(
            q,
            k,
            v,
            mapped,
            tau=args.tau,
            sink_rows=validated.sink_rows,
        )
        torch.cuda.synchronize()
        production_debug_metrics = sparse.error_metrics(debug_output, candidate)
        if not sparse.arithmetic_gate_passes(production_debug_metrics):
            raise RuntimeError(
                f"production and debug mapped specializations diverged beyond the arithmetic gate: "
                f"{production_debug_metrics}"
            )

        k_blocks = (validated.kv_rows + 63) // 64
        old_routes = _decode_route_trace(saved_trace.to(device), k_blocks)
        mapped_routes = _mapped_mask(descriptor.intervals, int(q.shape[2]), k_blocks, device)
        expected_routes = old_routes | mapped_routes
        current_routes = _decode_route_trace(current_trace, k_blocks)
        mismatch_count = int((current_routes ^ expected_routes).sum().item())
        if mismatch_count:
            raise RuntimeError(f"current production route differs from saved-E OR mapped contract at {mismatch_count} pairs")

        counts = _route_counts(old_routes, mapped_routes)
        for name in (
            "original_selected_pairs",
            "mapped_candidate_pairs",
            "added_selected_pairs",
            "effective_selected_pairs",
            "total_block_pairs",
        ):
            expected = m_record.get(name)
            if type(expected) is not int or counts[name] != expected:
                raise RuntimeError(f"same-input route count mismatch for {name}: {counts[name]} != {expected}")

        reference = _mixed_route_reference(
            q,
            k,
            v,
            kc,
            vc,
            expected_routes,
            scale=q.shape[-1] ** -0.5,
        )
        arithmetic = sparse.error_metrics(candidate, reference)
        if not sparse.arithmetic_gate_passes(arithmetic):
            raise RuntimeError(f"mapped same-route arithmetic gate failed: {arithmetic}")

        def ordinary():
            return kernel(q, k, v, tau=args.tau, sink_start=0, sink_tokens=validated.sink_rows)

        def mapped_call():
            return kernel(
                q,
                k,
                v,
                tau=args.tau,
                sink_start=0,
                sink_tokens=validated.sink_rows,
                mapped_neighbor_intervals=mapped,
            )

        timing = _timings({"ordinary": ordinary, "mapped": mapped_call}, args.warmup, args.repeats)
        provenance = verify_source()
        report = {
            "kind": "production_mapped_neighbor_same_input_probe_v1",
            "capture_id": CAPTURE_ID,
            "block_index": args.block,
            "group_index": args.group,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sol_repository": _git_identity(Path(__file__).resolve().parents[1]),
            "vdn_repository": _git_identity(args.vdn_path.resolve()),
            "kernel_contract": CONTRACT,
            "sana_revision": REVISION,
            "packaged_manifest_contract": provenance.get("contract"),
            "e_evidence_sha256": e_sha,
            "m_report_sha256": m_sha,
            "geometry": geometry_report,
            "q_shape": list(q.shape),
            "kv_shape": list(k.shape),
            "descriptor_cache_bytes": state.mapped_descriptor_bytes,
            "descriptor_cold_ms": descriptor_cold_ms,
            "descriptor_warm_host_us": descriptor_warm_host_us,
            "production_cold_first_mapped_ms": cold_candidate_ms,
            "compile_cache_new_keys_first_descriptor": len(after_first_keys - before_keys),
            "compile_cache_new_keys_value_change": len(after_second_keys - after_first_keys),
            "route_mismatch_count": mismatch_count,
            "route_counts": counts,
            "route_counts_match_preserved_m": True,
            "production_debug_arithmetic": production_debug_metrics,
            "production_debug_arithmetic_gate_pass": True,
            "same_route_arithmetic": arithmetic,
            "same_route_arithmetic_gate_pass": True,
            "warmed_kernel_timing": timing,
            "timing_scope": (
                "packaged public Sol-Attn preprocessing + SM120 attention on preserved E Q/K/V; "
                "excludes H3/VDN gather and model execution"
            ),
            "production_same_input_gate_pass": True,
        }
        print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
