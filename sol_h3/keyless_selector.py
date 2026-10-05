"""K3 Keyless Sol selector isolation on top of the released SM120 routing policy.

This module is diagnostic-only.  It lets a probe feed K1-produced routed
centroids/value sums into the existing Sol threshold and CTA routing policy while
keeping the exact-block K tensor materialized.  That deliberately isolates K3
selector behavior from the later K4 task of deriving selected routed K tiles from
raw V inside the CuTe mainloop.

Nothing here is installed into production dispatch.
"""
from __future__ import annotations

import math
import threading

import torch


CONTRACT = "sol-h3-keyless-selector-k3-calibration-v1"
BLOCK_SIZE = 64
ROUTE_GROUP_BLOCKS = 64
TRACE_WORD_BITS = 32
TRACE_WORDS = 2

_COMPILED: dict[tuple[object, ...], object] = {}
_COMPILE_LOCK = threading.Lock()


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def route_trace_shape(
    *,
    batch: int,
    q_rows: int,
    kv_rows: int,
    heads: int,
) -> tuple[int, int, int, int, int]:
    if min(batch, q_rows, kv_rows, heads) <= 0:
        raise ValueError("route trace geometry must be positive")
    q_tiles = _ceil_div(q_rows, BLOCK_SIZE)
    kv_blocks = _ceil_div(kv_rows, BLOCK_SIZE)
    route_groups = _ceil_div(kv_blocks, ROUTE_GROUP_BLOCKS)
    return (batch, q_tiles, heads, route_groups, TRACE_WORDS)


def sink_block_range(
    *,
    kv_rows: int,
    sink_start: int,
    sink_tokens: int,
) -> tuple[int, int]:
    if type(kv_rows) is not int or kv_rows <= 0:
        raise ValueError("kv_rows must be a positive integer")
    if type(sink_start) is not int or type(sink_tokens) is not int:
        raise ValueError("sink geometry must use integers")
    if sink_start < 0 or sink_tokens < 0 or sink_start > kv_rows:
        raise ValueError("sink geometry is outside the KV rows")
    if sink_start + sink_tokens > kv_rows:
        raise ValueError("sink_start + sink_tokens exceeds KV rows")
    blocks = _ceil_div(kv_rows, BLOCK_SIZE)
    if sink_tokens == 0:
        return blocks, blocks
    return (
        sink_start // BLOCK_SIZE,
        _ceil_div(sink_start + sink_tokens, BLOCK_SIZE),
    )


def threshold_from_route_centroids(
    q: torch.Tensor,
    route_centroid: torch.Tensor,
    *,
    kv_rows: int,
    tau: float,
    scale: float,
) -> torch.Tensor:
    """Apply the released Sol diagonal-threshold policy to K1 routed centroids."""
    if (
        q.ndim != 4
        or route_centroid.ndim != 4
        or q.shape[0] != route_centroid.shape[0]
        or q.shape[2:] != route_centroid.shape[2:]
        or q.shape[-1] != 128
    ):
        raise ValueError(
            "K3 threshold requires Q [B,Tq,H,128] and RC [B,ceil(Tv/64),H,128]"
        )
    if type(kv_rows) is not int or kv_rows <= 0:
        raise ValueError("K3 kv_rows must be a positive integer")
    expected_blocks = _ceil_div(kv_rows, BLOCK_SIZE)
    if int(route_centroid.shape[1]) != expected_blocks:
        raise ValueError(
            "K3 routed-centroid block count does not match the physical KV row count: "
            f"{route_centroid.shape[1]} vs {expected_blocks}"
        )
    if q.dtype is not torch.bfloat16 or route_centroid.dtype is not torch.bfloat16:
        raise TypeError("K3 threshold requires BF16 Q and routed centroids")
    if q.device != route_centroid.device or q.device.type != "cuda":
        raise ValueError("K3 threshold requires Q/RC on one CUDA device")
    if not math.isfinite(float(tau)) or not 0.0 <= float(tau) <= 3.0:
        raise ValueError("K3 tau must be finite and in [0, 3]")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("K3 attention scale must be finite and positive")

    from ._vendor.sol_attn.preprocess import _compute_diag_threshold

    # Bind the exact physical row count even though the current diagonal
    # threshold helper consumes it only through ceil(Tv/64).  This keeps K3
    # fail-closed if later threshold arithmetic starts using the partial-block
    # length directly.
    valid_kv_tokens = kv_rows
    return _compute_diag_threshold(
        q,
        route_centroid,
        tau=float(tau),
        scale=float(scale),
        valid_tokens=int(q.shape[1]),
        valid_kv_tokens=valid_kv_tokens,
    )


def _validate_executor_inputs(
    q: torch.Tensor,
    route: torch.Tensor,
    raw_v: torch.Tensor,
    route_centroid: torch.Tensor,
    value_sum: torch.Tensor,
    threshold: torch.Tensor,
) -> None:
    if (
        q.ndim != 4
        or route.ndim != 4
        or raw_v.ndim != 4
        or route.shape != raw_v.shape
        or q.shape[0] != route.shape[0]
        or q.shape[2:] != route.shape[2:]
        or q.shape[-1] != 128
    ):
        raise ValueError(
            "K3 executor requires Q [B,Tq,H,128], route/raw-V [B,Tv,H,128]"
        )
    if any(x.dtype is not torch.bfloat16 for x in (q, route, raw_v)):
        raise TypeError("K3 executor requires BF16 Q/route/raw-V")
    if any(x.device != q.device for x in (route, raw_v)):
        raise ValueError("K3 executor tensors must share one device")
    if q.device.type != "cuda":
        raise ValueError("K3 executor requires CUDA")
    blocks = _ceil_div(int(raw_v.shape[1]), BLOCK_SIZE)
    expected_summary = (int(q.shape[0]), blocks, int(q.shape[2]), 128)
    if tuple(route_centroid.shape) != expected_summary or tuple(value_sum.shape) != expected_summary:
        raise ValueError(
            f"K3 summaries must have shape {expected_summary}, got "
            f"{tuple(route_centroid.shape)} and {tuple(value_sum.shape)}"
        )
    if route_centroid.dtype is not torch.bfloat16 or value_sum.dtype is not torch.bfloat16:
        raise TypeError("K3 summaries must be BF16")
    if route_centroid.device != q.device or value_sum.device != q.device:
        raise ValueError("K3 summaries must share the Q device")
    q_tiles = _ceil_div(int(q.shape[1]), BLOCK_SIZE)
    expected_threshold = (int(q.shape[0]), q_tiles, int(q.shape[2]))
    if tuple(threshold.shape) != expected_threshold:
        raise ValueError(
            f"K3 threshold must have shape {expected_threshold}, got {tuple(threshold.shape)}"
        )
    if threshold.dtype is not torch.float32 or threshold.device != q.device:
        raise TypeError("K3 threshold must be FP32 on the Q device")


def run_materialized_exact_selector_isolation(
    q: torch.Tensor,
    route: torch.Tensor,
    raw_v: torch.Tensor,
    route_centroid: torch.Tensor,
    value_sum: torch.Tensor,
    threshold: torch.Tensor,
    *,
    scale: float,
    sink_start: int = 0,
    sink_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the existing SM120 selector with supplied RC/VC and trace its exact blocks.

    The exact K tensor is intentionally the materialized route.  This function
    therefore validates K3 selector/approximate-summary semantics only; it is not
    the K4 no-global-route production executor.
    """
    _validate_executor_inputs(
        q,
        route,
        raw_v,
        route_centroid,
        value_sum,
        threshold,
    )
    if tuple(torch.cuda.get_device_capability(q.device)) != (12, 0):
        raise RuntimeError("K3 selector isolation requires SM120")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("K3 attention scale must be finite and positive")

    sink_start_block, sink_end_block = sink_block_range(
        kv_rows=int(raw_v.shape[1]),
        sink_start=int(sink_start),
        sink_tokens=int(sink_tokens),
    )
    trace = torch.zeros(
        route_trace_shape(
            batch=int(q.shape[0]),
            q_rows=int(q.shape[1]),
            kv_rows=int(raw_v.shape[1]),
            heads=int(q.shape[2]),
        ),
        dtype=torch.int32,
        device=q.device,
    )
    output = torch.empty_like(q)

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from ._vendor.sol_attn.common import to_cute_tensor
    from ._vendor.sol_attn.sm120 import make_kernel

    # Disabled optional key-bias/mapped arguments reuse an already-live FP32
    # threshold tensor, matching the public SM120 interface without extra buffers.
    tensors = (
        q,
        route,
        raw_v,
        output,
        route_centroid,
        value_sum,
        threshold,
        threshold,
        threshold,
        trace,
    )
    layout_key = tuple(
        (tuple(int(v) for v in tensor.shape), tuple(int(v) for v in tensor.stride()))
        for tensor in tensors
    )
    key = (
        int(q.device.index or 0),
        str(q.dtype),
        layout_key,
        float(scale),
        int(sink_start_block),
        int(sink_end_block),
        CONTRACT,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = [to_cute_tensor(tensor) for tensor in tensors]
    compiled = _COMPILED.get(key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                operator = make_kernel(
                    debug_route_trace=True,
                    key_bias_enabled=False,
                    mapped_neighbors_enabled=False,
                )
                compiled = cute.compile(
                    operator,
                    *args,
                    float(scale),
                    int(sink_start_block),
                    int(sink_end_block),
                    stream=stream,
                    options="--enable-tvm-ffi",
                )
                _COMPILED[key] = compiled
    compiled(
        *args,
        float(scale),
        int(sink_start_block),
        int(sink_end_block),
        stream=stream,
    )
    return output, trace


def route_trace_metrics(
    got: torch.Tensor,
    want: torch.Tensor,
    *,
    kv_rows: int,
) -> dict[str, int | float | bool]:
    """Compare only valid physical KV-block bits from two debug-route traces."""
    if got.shape != want.shape:
        raise ValueError(
            f"route traces differ in shape: {tuple(got.shape)} vs {tuple(want.shape)}"
        )
    if got.ndim != 5 or got.shape[-1] != TRACE_WORDS:
        raise ValueError("route trace must have shape [B,Qtiles,H,groups,2]")
    if got.dtype is not torch.int32 or want.dtype is not torch.int32:
        raise TypeError("route traces must be int32")
    if type(kv_rows) is not int or kv_rows <= 0:
        raise ValueError("kv_rows must be a positive integer")

    kv_blocks = _ceil_div(kv_rows, BLOCK_SIZE)
    expected_groups = _ceil_div(kv_blocks, ROUTE_GROUP_BLOCKS)
    if int(got.shape[3]) != expected_groups:
        raise ValueError(
            f"route trace has {got.shape[3]} groups, expected {expected_groups}"
        )

    got_cpu = got.detach().to(device="cpu")
    want_cpu = want.detach().to(device="cpu")
    differing_bits = 0
    got_exact = 0
    want_exact = 0
    total = int(got.shape[0] * got.shape[1] * got.shape[2] * kv_blocks)

    for batch in range(int(got.shape[0])):
        for q_tile in range(int(got.shape[1])):
            for head in range(int(got.shape[2])):
                for kv_block in range(kv_blocks):
                    group = kv_block // ROUTE_GROUP_BLOCKS
                    offset = kv_block % ROUTE_GROUP_BLOCKS
                    word = offset // TRACE_WORD_BITS
                    bit = offset % TRACE_WORD_BITS
                    got_word = int(got_cpu[batch, q_tile, head, group, word].item())
                    want_word = int(want_cpu[batch, q_tile, head, group, word].item())
                    got_bit = ((got_word & 0xFFFFFFFF) >> bit) & 1
                    want_bit = ((want_word & 0xFFFFFFFF) >> bit) & 1
                    got_exact += got_bit
                    want_exact += want_bit
                    differing_bits += got_bit ^ want_bit

    return {
        "equal": differing_bits == 0,
        "differing_bits": differing_bits,
        "total_routing_bits": total,
        "differing_bit_fraction": differing_bits / total if total else 0.0,
        "candidate_exact_bits": got_exact,
        "reference_exact_bits": want_exact,
    }


__all__ = [
    "BLOCK_SIZE",
    "CONTRACT",
    "ROUTE_GROUP_BLOCKS",
    "TRACE_WORDS",
    "route_trace_metrics",
    "route_trace_shape",
    "run_materialized_exact_selector_isolation",
    "sink_block_range",
    "threshold_from_route_centroids",
]
