"""Request-owned single-union Sol route for exact-prefix progressive attention.

The first partition prototype evaluated physical K/V domains independently and
merged their LSEs. That is algebraically exact for dense attention, but it changes
Sol's sparse routing because each partition receives an independently computed
routing threshold. The production-shaped route in this module instead keeps one
physical K/V union, applies the target-prefix key measure as an additive natural-
log bias, and combines that with VDN-owned mapped-neighbor metadata in one SM120
kernel invocation.

This module is opt-in. Released Sol-H3 routing is unchanged unless the Flow/VDN
partitioned development path imports and calls ``partitioned_request_attention``.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from typing import Any

import torch
import torch.nn.functional as F

from .mapped_neighbors import MappingUnavailable, compile_descriptor, device_descriptor, validate_wire_map
from .sparse import arithmetic_gate_passes, error_metrics

PARTITIONED_REQUEST_ABI = "sol-h3-partitioned-single-union-v1"
PARTITIONED_RECEIPT_TAG = "sol_h3_partitioned_exact_prefix_v1"
MAX_BIAS_CACHE_ENTRIES = 64
MAX_BIAS_CACHE_BYTES = 4 * 1024 * 1024
BLOCK_SIZE = 64


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_state():
    from .runtime import _REQUEST

    state = _REQUEST.get()
    if state is None:
        raise RuntimeError("partitioned Sol must execute inside the native Sol OUTER_SAMPLE lifecycle")
    return state


def _forward_evaluation() -> int:
    from .runtime import _FORWARD

    active = _FORWARD.get()
    if active is None or len(active) != 5:
        raise RuntimeError("partitioned Sol must execute inside the native Sol diffusion lifecycle")
    return int(active[2])


def _validate_thd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if (
        any(t.ndim != 3 for t in (q, k, v))
        or k.shape != v.shape
        or q.shape[1:] != k.shape[1:]
        or q.shape[0] <= 0
        or k.shape[0] <= 0
        or q.shape[1] <= 0
        or q.shape[-1] != 128
    ):
        raise RuntimeError("partitioned Sol requires Q [Tq,H,128] and KV [Tkv,H,128]")
    if any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)):
        raise RuntimeError("partitioned Sol requires BF16 QKV on one device")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("partitioned Sol currently requires SM120")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise RuntimeError("partitioned Sol requires a contiguous head dimension")
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("partitioned Sol is unavailable under autograd, torch.compile, or CUDA graph capture")


def _bias_cache(state) -> OrderedDict:
    cache = getattr(state, "partitioned_measure_bias_cache", None)
    if cache is None:
        cache = OrderedDict()
        state.partitioned_measure_bias_cache = cache
        state.partitioned_measure_bias_bytes = 0
    if not isinstance(cache, OrderedDict):
        raise RuntimeError("partitioned Sol bias-cache ownership was replaced")
    return cache


def _key_bias(
    state,
    *,
    kv_rows: int,
    prefix_range: tuple[int, int] | None,
    log_measure: float,
    device: torch.device,
    semantic_digest: str,
) -> torch.Tensor | None:
    if prefix_range is None:
        if log_measure != 0.0:
            raise RuntimeError("partitioned Sol received a measure without a target-prefix K/V range")
        return None
    if (
        not isinstance(prefix_range, tuple)
        or len(prefix_range) != 2
        or any(type(value) is not int for value in prefix_range)
    ):
        raise RuntimeError("partitioned Sol prefix K/V range must be an integer pair")
    start, end = prefix_range
    if not 0 <= start < end <= kv_rows:
        raise RuntimeError("partitioned Sol prefix K/V range is outside the current domain")
    log_measure = float(log_measure)
    if not math.isfinite(log_measure) or log_measure > 0.0:
        raise RuntimeError("partitioned Sol target-prefix log measure must be finite and non-positive")
    if not isinstance(semantic_digest, str) or len(semantic_digest) != 64:
        raise RuntimeError("partitioned Sol semantic digest is invalid")

    cache = _bias_cache(state)
    key = (
        PARTITIONED_REQUEST_ABI,
        semantic_digest,
        kv_rows,
        start,
        end,
        log_measure,
        str(device),
    )
    hit = cache.get(key)
    if hit is not None:
        cache.move_to_end(key)
        return hit
    bias = torch.zeros(kv_rows, dtype=torch.float32, device=device)
    bias[start:end] = log_measure
    cache[key] = bias
    cache.move_to_end(key)
    size = bias.numel() * bias.element_size()
    state.partitioned_measure_bias_bytes += size
    while len(cache) > MAX_BIAS_CACHE_ENTRIES or state.partitioned_measure_bias_bytes > MAX_BIAS_CACHE_BYTES:
        _, evicted = cache.popitem(last=False)
        state.partitioned_measure_bias_bytes -= evicted.numel() * evicted.element_size()
    return bias


def _weighted_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_bias: torch.Tensor | None,
    *,
    scale: float,
) -> torch.Tensor:
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    mask = None if key_bias is None else key_bias.to(q.dtype).view(1, 1, 1, -1)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=float(scale))
    return out.squeeze(0).transpose(0, 1).contiguous()


def _sm120_union(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    sink_rows: int,
    key_bias: torch.Tensor | None,
    mapped_neighbor_intervals: torch.Tensor | None,
) -> torch.Tensor:
    """Execute one SM120 sparse attention union with optional bias + mapped metadata."""
    from .provenance import verify_source
    from ._vendor.sol_attn import interface
    from ._vendor.sol_attn.preprocess import prepare

    verify_source()
    qb = q.unsqueeze(0)
    kb = k.unsqueeze(0)
    vb = v.unsqueeze(0)
    arch = tuple(torch.cuda.get_device_capability(q.device))
    if interface._backend_for_arch(arch) != "cute_sm120":
        raise RuntimeError("partitioned Sol requires the packaged cute_sm120 backend")
    interface._validate_cute(arch, kb.shape[1], 1)
    if type(sink_rows) is not int or not 0 <= sink_rows <= kb.shape[1]:
        raise RuntimeError("partitioned Sol exact K prefix is outside the K/V domain")

    if key_bias is not None and (
        key_bias.ndim != 1
        or key_bias.shape[0] != kb.shape[1]
        or key_bias.dtype != torch.float32
        or key_bias.device != q.device
        or not key_bias.is_contiguous()
    ):
        raise RuntimeError("partitioned Sol key bias must be contiguous FP32 [Tkv]")

    batch, q_rows, heads, _ = qb.shape
    kv_rows = kb.shape[1]
    with torch.cuda.device(q.device):
        kc, vc, threshold = prepare(
            qb,
            kb,
            vb,
            scale=float(scale),
            tau=float(tau),
            thresh_type="diag",
            valid_tokens=q_rows,
            valid_kv_tokens=kv_rows,
        )
        output = torch.empty_like(qb)
        lse = torch.empty((batch, q_rows, heads), device=q.device, dtype=torch.float32)
        stream = interface._stream(q.device)
        sink_start_block, sink_end_block = interface._sink_block_range(kv_rows, 0, sink_rows)
        key_bias_arg = key_bias if key_bias is not None else threshold
        mapped_arg = mapped_neighbor_intervals if mapped_neighbor_intervals is not None else threshold
        tensors = [qb, kb, vb, output, kc, vc, threshold, key_bias_arg, mapped_arg, lse]
        layout_key = tuple(tuple(int(value) for value in tensor.stride()) for tensor in (qb, kb, vb))
        key = (
            PARTITIONED_REQUEST_ABI,
            q.device.index,
            arch,
            batch,
            q_rows,
            kv_rows,
            heads,
            layout_key,
            key_bias is not None,
            mapped_neighbor_intervals is not None,
        )
        compiled = interface._compiled.get(key)
        if compiled is None:
            with interface._compile_lock:
                compiled = interface._compiled.get(key)
                if compiled is None:
                    compiled, args = interface._compile_sm120(
                        key,
                        tensors,
                        float(scale),
                        sink_start_block,
                        sink_end_block,
                        stream,
                        key_bias is not None,
                        mapped_neighbor_intervals is not None,
                    )
                else:
                    args = interface._to_cute_tensors(tensors)
        else:
            args = interface._to_cute_tensors(tensors)
        compiled(*args, float(scale), sink_start_block, sink_end_block, stream=stream)
    return output[0]


def _descriptor_for_wire(state, wire, *, q_rows: int, kv_rows: int, sink_rows: int, device):
    if wire is None:
        return None, None
    try:
        validated = validate_wire_map(wire, q_rows=q_rows, kv_rows=kv_rows, sink_rows=sink_rows)
        descriptor = compile_descriptor(validated)
        if descriptor is None:
            return validated, None
        return validated, device_descriptor(state, descriptor, device)
    except MappingUnavailable as exc:
        raise RuntimeError(f"partitioned Sol mapped-neighbor contract is unavailable: {exc.reason}") from exc


def partitioned_request_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    transformer_options: dict,
    block_index: int,
    kind: str,
    scale: float,
    sink_rows: int,
    prefix_k_range: tuple[int, int] | None,
    prefix_log_key_measure: float,
    semantic_digest: str,
    query_position_map=None,
    force_dense: bool = False,
) -> torch.Tensor:
    """Run one request-owned partitioned attention subcall over a single K/V union."""
    _validate_thd(q, k, v)
    if kind not in {"global", "local", "anchor", "full"}:
        raise RuntimeError("partitioned Sol attention kind is unsupported")
    if type(block_index) is not int or block_index < 0:
        raise RuntimeError("partitioned Sol block index is invalid")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0 or scale != q.shape[-1] ** -0.5:
        raise RuntimeError("partitioned Sol requires the native H3 attention scale")
    if type(sink_rows) is not int or not 0 <= sink_rows <= k.shape[0]:
        raise RuntimeError("partitioned Sol sink-row count is invalid")

    state = _request_state()
    config = state.config
    evaluation = _forward_evaluation()
    key_bias = _key_bias(
        state,
        kv_rows=int(k.shape[0]),
        prefix_range=prefix_k_range,
        log_measure=float(prefix_log_key_measure),
        device=q.device,
        semantic_digest=semantic_digest,
    )
    exact_end = sink_rows
    if prefix_k_range is not None:
        if prefix_k_range[0] < sink_rows:
            raise RuntimeError("partitioned Sol target-prefix K/V range overlaps the global sink")
        exact_end = max(exact_end, prefix_k_range[1])
    validated, mapped = _descriptor_for_wire(
        state,
        query_position_map,
        q_rows=int(q.shape[0]),
        kv_rows=int(k.shape[0]),
        sink_rows=sink_rows,
        device=q.device,
    )

    from .interop import dense_evaluation_warmup

    warmup = dense_evaluation_warmup(config, evaluation, transformer_options)
    if force_dense or warmup:
        result = _weighted_dense(q, k, v, key_bias, scale=scale)
        state.partitioned_dense_calls = getattr(state, "partitioned_dense_calls", 0) + 1
        return result

    calibration_identity = _sha256_json(
        {
            "abi": PARTITIONED_REQUEST_ABI,
            "semantic_digest": semantic_digest,
            "kind": kind,
            "q_rows": int(q.shape[0]),
            "kv_rows": int(k.shape[0]),
            "sink_rows": sink_rows,
            "exact_end": exact_end,
            "mapped": mapped is not None,
            "map_digest": None if validated is None else validated.map_digest,
        }
    )
    verified = getattr(state, "partitioned_sparse_verified", None)
    if verified is None:
        verified = set()
        state.partitioned_sparse_verified = verified
    layout_key = (
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.shape),
        tuple(k.stride()),
        tuple(v.shape),
        tuple(v.stride()),
        calibration_identity,
    )
    if layout_key not in verified:
        got = _sm120_union(
            q,
            k,
            v,
            tau=config.tau,
            scale=scale,
            sink_rows=int(k.shape[0]),
            key_bias=key_bias,
            mapped_neighbor_intervals=mapped,
        )
        want = _weighted_dense(q, k, v, key_bias, scale=scale)
        gate = error_metrics(
            got.transpose(0, 1).unsqueeze(0),
            want.transpose(0, 1).unsqueeze(0),
        )
        if not arithmetic_gate_passes(gate):
            raise RuntimeError(f"partitioned Sol all-selected arithmetic gate failed: {gate}")
        verified.add(layout_key)
        state.gates.append(
            {
                "route": PARTITIONED_REQUEST_ABI,
                "kind": kind,
                "shape": [1, q.shape[1], q.shape[0], q.shape[2]],
                "kv_shape": [1, k.shape[1], k.shape[0], k.shape[2]],
                "mapped_neighbor_abi": mapped is not None,
                "key_measure_bias": key_bias is not None,
                **gate,
            }
        )

    result = _sm120_union(
        q,
        k,
        v,
        tau=config.tau,
        scale=scale,
        sink_rows=exact_end,
        key_bias=key_bias,
        mapped_neighbor_intervals=mapped,
    )
    state.partitioned_sparse_calls = getattr(state, "partitioned_sparse_calls", 0) + 1
    state.partitioned_requested_q_rows = getattr(state, "partitioned_requested_q_rows", 0) + int(q.shape[0])
    state.partitioned_kernel_q_rows = getattr(state, "partitioned_kernel_q_rows", 0) + int(q.shape[0])
    return result


__all__ = ["PARTITIONED_RECEIPT_TAG", "PARTITIONED_REQUEST_ABI", "partitioned_request_attention"]
