"""Diagnostic-only mapped-neighbor (M) Sol-Attn arm.

This module is inert without the first-high diagnostic request in mode
``mapped_neighbor_m``.  It keeps the production diagonal threshold, sink, VDN
restricted K/V support, and every ordinary route decision, then ORs in only the
VDN-owned mapped query-neighbor K blocks.  No production API is repurposed.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from . import first_high_sol_local_diagnostic as e
from . import sparse

MODE = "mapped_neighbor_m"
ROUTE = "vdn_local_sol_mapped_neighbor_m"
_COMPILED: dict[tuple[Any, ...], Any] = {}
_COMPILE_LOCK = threading.RLock()
_INSTALLED = False

# Under the M overlay, the existing one-call E harness carries the same immutable
# request contract but with a separately identified mode.  E itself remains
# unchanged on its underlying PR.
e.MODE = MODE


@dataclass(frozen=True, slots=True)
class MappedGroup:
    block_index: int
    group_index: int
    q_rows: int
    kv_rows: int
    original_sink_rows: int
    scale: float
    intervals: tuple[tuple[int, int], ...]
    query_positions_sha256: str
    options: dict[str, Any]


_GROUP: contextvars.ContextVar[MappedGroup | None] = contextvars.ContextVar(
    "h3_first_high_mapped_neighbor_group", default=None
)


def _descriptor_sha256(intervals: tuple[tuple[int, int], ...]) -> str:
    payload = json.dumps(intervals, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_intervals(q_rows: int, kv_rows: int, intervals: Any) -> tuple[tuple[int, int], ...]:
    q_blocks = (int(q_rows) + 63) // 64
    k_blocks = (int(kv_rows) + 63) // 64
    if not isinstance(intervals, (tuple, list)) or len(intervals) != q_blocks:
        raise RuntimeError("mapped-neighbor M requires one bounded K-block interval per Q64 tile")
    result = []
    for index, raw in enumerate(intervals):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2 or any(type(v) is not int for v in raw):
            raise RuntimeError(f"mapped-neighbor M interval {index} is not an integer [start,end) pair")
        start, end = int(raw[0]), int(raw[1])
        if not 0 <= start < end <= k_blocks:
            raise RuntimeError(f"mapped-neighbor M interval {index} is outside restricted K/V")
        # One contiguous Q64 tile can touch at most two K64 blocks; adding +/-1
        # therefore yields at most four K blocks. Anything wider is not the
        # reviewed M intervention.
        if end - start > 4:
            raise RuntimeError(f"mapped-neighbor M interval {index} exceeds the reviewed +/-1 bound")
        result.append((start, end))
    return tuple(result)


def enter_mapped_group(options: dict[str, Any], metadata: dict[str, Any]):
    request = e.parse_request(options)
    if request is None or request.get("mode") != MODE:
        raise RuntimeError("mapped-neighbor M group requires the exact M request")
    required = (
        "block_index",
        "group_index",
        "q_rows",
        "kv_rows",
        "original_sink_rows",
        "scale",
        "mapped_neighbor_intervals",
        "query_positions_sha256",
    )
    if any(name not in metadata for name in required):
        raise RuntimeError("mapped-neighbor M group metadata is incomplete")
    if _GROUP.get() is not None:
        raise RuntimeError("nested mapped-neighbor M groups are unsupported")
    block_index = int(metadata["block_index"])
    group_index = int(metadata["group_index"])
    q_rows = int(metadata["q_rows"])
    kv_rows = int(metadata["kv_rows"])
    sink_rows = int(metadata["original_sink_rows"])
    scale = float(metadata["scale"])
    if not 2 <= block_index < 50 or not 0 <= group_index < 11:
        raise RuntimeError("mapped-neighbor M is restricted to Sol-local blocks 2..49 and groups 0..10")
    if q_rows <= 0 or kv_rows <= 0 or not 0 <= sink_rows <= kv_rows:
        raise RuntimeError("mapped-neighbor M group geometry is invalid")
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("mapped-neighbor M attention scale is invalid")
    query_sha = metadata["query_positions_sha256"]
    if not isinstance(query_sha, str) or len(query_sha) != 64:
        raise RuntimeError("mapped-neighbor M query-position digest is invalid")
    intervals = _validate_intervals(q_rows, kv_rows, metadata["mapped_neighbor_intervals"])
    group = MappedGroup(
        block_index=block_index,
        group_index=group_index,
        q_rows=q_rows,
        kv_rows=kv_rows,
        original_sink_rows=sink_rows,
        scale=scale,
        intervals=intervals,
        query_positions_sha256=query_sha,
        options=options,
    )
    return _GROUP.set(group)


def exit_mapped_group(token) -> None:
    _GROUP.reset(token)


def _mapped_selector(intervals: tuple[tuple[int, int], ...]):
    """Build one compile-time additive selector for a bounded Q64 descriptor."""
    import cutlass
    import cutlass.cute as cute

    from ._vendor.sol_attn.common.selector import sol_attn_route_is_exact as base_selector

    starts = tuple(int(item[0]) for item in intervals)
    ends = tuple(int(item[1]) for item in intervals)

    @cute.jit
    def selector(q_block, kv_block, column_mean, threshold, valid):
        exact = base_selector(q_block, kv_block, column_mean, threshold, valid)
        mapped = False
        for index in cutlass.range_constexpr(len(starts)):
            if q_block == cutlass.Int32(index):
                mapped = (kv_block >= cutlass.Int32(starts[index])) and (kv_block < cutlass.Int32(ends[index]))
        return (exact or mapped) and valid

    return selector


def _mapped_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    sink_rows: int,
    intervals: tuple[tuple[int, int], ...],
):
    """Execute the real packaged SM120 mainloop with only the selector augmented."""
    import cutlass.cute as cute

    from ._vendor.sol_attn.interface import _sink_block_range, _stream, _to_cute_tensors
    from ._vendor.sol_attn.preprocess import prepare
    from ._vendor.sol_attn.sm120 import make_kernel
    from ._vendor.sol_attn.sm120 import mainloop

    if tuple(torch.cuda.get_device_capability(q.device)) != (12, 0):
        raise RuntimeError("mapped-neighbor M requires SM120")
    kc, vc, threshold, qbar = prepare(
        q,
        k,
        v,
        tau=float(tau),
        scale=float(scale),
        thresh_type="diag",
        valid_tokens=int(q.shape[1]),
        valid_kv_tokens=int(k.shape[1]),
        return_q_bar=True,
    )
    output = torch.empty_like(q)
    lse = torch.empty((int(q.shape[0]), int(q.shape[1]), int(q.shape[2])), device=q.device, dtype=torch.float32)
    key_bias_arg = threshold
    tensors = [q, k, v, output, kc, vc, threshold, key_bias_arg, lse]
    args = _to_cute_tensors(tensors)
    stream = _stream(q.device)
    sink_start_block, sink_end_block = _sink_block_range(int(k.shape[1]), 0, int(sink_rows))
    layout_key = tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype)) for x in tensors)
    descriptor = tuple((int(a), int(b)) for a, b in intervals)
    key = (str(q.device), layout_key, float(scale), float(tau), sink_start_block, sink_end_block, descriptor)

    compiled = _COMPILED.get(key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                original_selector = mainloop.sol_attn_route_is_exact
                if original_selector is not __import__(
                    f"{__package__}._vendor.sol_attn.common.selector", fromlist=["sol_attn_route_is_exact"]
                ).sol_attn_route_is_exact:
                    raise RuntimeError("mapped-neighbor M found an unexpected pre-existing SM120 selector override")
                try:
                    mainloop.sol_attn_route_is_exact = _mapped_selector(descriptor)
                    operator = make_kernel(key_bias_enabled=False)
                    compiled = cute.compile(
                        operator,
                        *args,
                        float(scale),
                        sink_start_block,
                        sink_end_block,
                        stream=stream,
                        options="--enable-tvm-ffi",
                    )
                finally:
                    mainloop.sol_attn_route_is_exact = original_selector
                _COMPILED[key] = compiled
    compiled(
        *args,
        float(scale),
        sink_start_block,
        sink_end_block,
        stream=stream,
    )
    return output, lse, kc, vc, threshold, qbar


def _route_evidence(
    group: MappedGroup,
    qbar: torch.Tensor,
    kc: torch.Tensor,
    threshold: torch.Tensor,
) -> dict[str, Any]:
    """Recompute the exact ordinary selector and additive M delta for telemetry."""
    q_blocks = len(group.intervals)
    k_blocks = int(kc.shape[1])
    if int(qbar.shape[1]) != q_blocks:
        raise RuntimeError("mapped-neighbor M Q summary count differs from descriptor")
    log2_scale = float(group.scale) * 1.4426950408889634
    means = torch.einsum("bqhd,bkhd->qhk", qbar.float(), kc.float()) * log2_scale
    old = means > threshold[0, :q_blocks, :, None]
    ordinal = torch.arange(k_blocks, device=old.device, dtype=torch.int64)
    q_ordinal = torch.arange(q_blocks, device=old.device, dtype=torch.int64)[:, None]
    ordinal_neighbor = (ordinal[None, :] - q_ordinal).abs() <= 1
    sink_end = (group.original_sink_rows + 63) // 64
    sink = ordinal < sink_end
    old = old | ordinal_neighbor[:, None, :] | sink[None, None, :]
    mapped = torch.zeros((q_blocks, k_blocks), device=old.device, dtype=torch.bool)
    for q_block, (start, end) in enumerate(group.intervals):
        mapped[q_block, start:end] = True
    mapped_heads = mapped[:, None, :].expand(-1, int(old.shape[1]), -1)
    missing = mapped_heads & ~old
    effective = old | mapped_heads
    original_count = int(old.sum().item())
    added_count = int(missing.sum().item())
    effective_count = int(effective.sum().item())
    total_pairs = int(old.numel())
    mapped_pairs = int(mapped_heads.sum().item())
    if effective_count != original_count + added_count or bool((old & ~effective).any().item()):
        raise RuntimeError("mapped-neighbor M additive route accounting is inconsistent")
    return {
        "kind": "mapped_neighbor_route",
        "block_index": group.block_index,
        "group_index": group.group_index,
        "reason": "mapped_neighbor",
        "q_rows": group.q_rows,
        "kv_rows": group.kv_rows,
        "q_blocks": q_blocks,
        "k_blocks": k_blocks,
        "original_sink_rows": group.original_sink_rows,
        "mapped_neighbor_intervals": [list(item) for item in group.intervals],
        "descriptor_sha256": _descriptor_sha256(group.intervals),
        "query_positions_sha256": group.query_positions_sha256,
        "original_selected_pairs": original_count,
        "mapped_candidate_pairs": mapped_pairs,
        "added_selected_pairs": added_count,
        "effective_selected_pairs": effective_count,
        "total_block_pairs": total_pairs,
        "exact_work_increase_fraction": (float(added_count) / float(original_count)) if original_count else math.inf,
        "additive_only": True,
        "restricted_domain_unchanged": True,
    }


def _ensure_kernel_and_calibration(q, k, v, config, state):
    if state.kernel is None:
        failure = getattr(state, "kernel_failure", None)
        if failure:
            raise sparse.KernelUnavailable(failure)
        started = time.perf_counter()
        try:
            state.kernel = sparse.load_kernel(q.device)
        except RuntimeError as exc:
            state.kernel_failure = str(exc)
            raise sparse.KernelUnavailable(str(exc)) from exc
        kernel_loader_s = time.perf_counter() - started
        state.kernel_device = q.device
    else:
        kernel_loader_s = 0.0
        if q.device != state.kernel_device:
            raise RuntimeError("mapped-neighbor M compute device changed within the sampling request")

    qb, kb, vb = (x.transpose(1, 2) for x in (q, k, v))
    layout_key = sparse._bthd_layout_key(qb, kb, vb)
    calibration_key = (q.device, q.dtype, layout_key, None)
    if calibration_key not in state.sparse_verified:
        started = time.perf_counter()
        got = state.kernel(
            qb,
            kb,
            vb,
            tau=config.tau,
            thresh_type="diag",
            kv_splits=1,
            sink_start=0,
            sink_tokens=kb.shape[1],
        )
        want = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        metrics = sparse.error_metrics(got, want)
        gate_wall_s = time.perf_counter() - started
        if not sparse.arithmetic_gate_passes(metrics):
            raise RuntimeError(f"mapped-neighbor M all-selected arithmetic gate failed: {metrics}")
        state.sparse_verified.add(calibration_key)
        state.gates.append(
            {
                "shape": list(q.shape),
                "kv_shape": list(k.shape),
                "backend": getattr(state.kernel, "backend_name", "test_substitute"),
                "kernel_loader_s": kernel_loader_s,
                "gate_wall_s": gate_wall_s,
                "materialized_qkv_bytes": 0,
                "bthd_qkv_bytes": sum(x.numel() * x.element_size() for x in (qb, kb, vb)),
                "bthd_strides": [list(x.stride()) for x in (qb, kb, vb)],
                **metrics,
            }
        )
    return qb, kb, vb


def _mapped_attention(q, k, v, prefix, config, state, *args, **kwargs):
    group = _GROUP.get()
    if group is None:
        return _CURRENT_ATTENTION(q, k, v, prefix, config, state, *args, **kwargs)
    if args or kwargs.get("key_bias") is not None or kwargs.get("exact_k_blocks") is not None:
        raise RuntimeError("mapped-neighbor M forbids weighted/external routing metadata")
    if kwargs.get("recompute_prefix_queries", False):
        raise RuntimeError("mapped-neighbor M requires VDN local queries without prefix-query recomputation")
    if int(prefix) != group.original_sink_rows or int(k.shape[2]) != group.kv_rows or int(q.shape[2]) != group.q_rows:
        raise RuntimeError("mapped-neighbor M final Sol geometry differs from the VDN-owned descriptor")
    if float(group.scale) != float(q.shape[-1] ** -0.5):
        raise RuntimeError("mapped-neighbor M attention scale differs from VDN")
    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
        raise RuntimeError("mapped-neighbor M requires BF16 QKV on one device")

    qb, kb, vb = _ensure_kernel_and_calibration(q, k, v, config, state)
    output, _lse, kc, _vc, threshold, qbar = _mapped_launch(
        qb,
        kb,
        vb,
        tau=float(config.tau),
        scale=group.scale,
        sink_rows=group.original_sink_rows,
        intervals=group.intervals,
    )
    evidence = _route_evidence(group, qbar, kc, threshold)
    sink = group.options.get(e.EVIDENCE_KEY)
    append = getattr(sink, "append", None)
    if not callable(append):
        raise RuntimeError("mapped-neighbor M evidence sink is missing")
    append(evidence)
    e._append_provenance_once(group.options)
    # Match E's counter-isolation contract: the M arm changes only the returned
    # local operator, not ordinary Sol production accounting.
    return output.reshape(1, q.shape[2], -1)


def _install_attention() -> None:
    global _CURRENT_ATTENTION
    _CURRENT_ATTENTION = sparse.attention
    if getattr(_CURRENT_ATTENTION, "_first_high_mapped_neighbor_m_v1", False):
        return
    _mapped_attention._first_high_mapped_neighbor_m_v1 = True
    _mapped_attention._first_high_mapped_neighbor_inner = _CURRENT_ATTENTION
    sparse.attention = _mapped_attention


def _install_receipts() -> None:
    from . import runtime

    current = runtime.receipt
    if getattr(current, "_first_high_mapped_neighbor_m_v1", False):
        return

    def receipt(options, block_index, route, *args, **kwargs):
        group = _GROUP.get()
        request = e.parse_request(options) if isinstance(options, dict) else None
        if group is not None and request is not None and request.get("mode") == MODE and route == "vdn_local_sol":
            route = ROUTE
        return current(options, block_index, route, *args, **kwargs)

    receipt._first_high_mapped_neighbor_m_v1 = True
    receipt._first_high_mapped_neighbor_inner = current
    runtime.receipt = receipt


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_attention()
    _install_receipts()
    _INSTALLED = True


install()

__all__ = ["MODE", "ROUTE", "enter_mapped_group", "exit_mapped_group", "install"]
