# Modified by ComfyUI-Sol-H3: rectangular SM120 Q/KV geometry and exact-path key measure; see tools/rectangular_sm120.patch.
"""Public Sol-Attn interface."""

from __future__ import annotations

import functools
import math

import torch

BLOCK_SIZE = 64
_CUTE_BACKENDS = {
    (9, 0): "cute_sm90",
    (10, 0): "cute_sm100",
    # B300 / Blackwell Ultra reports SM103.  The H3 BF16 attention kernel uses
    # the common Blackwell tcgen05 path (not SM100-only block-scaled MMA), and
    # CUTLASS DSL >=4.3.1 can lower that path for SM103.  Keeping SM103 out of
    # this table silently selected the much slower Triton reference backend.
    (10, 3): "cute_sm100",
    (12, 0): "cute_sm120",
}
_compiled = {}


def _validate_inputs(
    q,
    k,
    v,
    thresh_type,
    sink_tokens=0,
    sink_start=None,
):
    if (any(x.ndim != 4 for x in (q, k, v)) or k.shape != v.shape
            or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]):
        raise ValueError("q must be [B, Tq, H, 128]; k/v must share [B, Tkv, H, 128]")
    if q.shape[1] == 0 or k.shape[1] == 0 or q.shape[3] != 128:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    # The Ulysses fast path receives one packed [..., Q | K | V] allocation.  Each component has
    # a contiguous head dimension but a 3*D row/head stride.  CuTe's tensor descriptors carry
    # those strides into TMA, so materialising three contiguous tensors before attention is not a
    # kernel requirement.  Requiring the innermost dimension to be contiguous retains vector/TMA
    # alignment while allowing the packed zero-copy view.
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("q, k, and v must have a contiguous head dimension")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    if not isinstance(sink_tokens, int):
        raise TypeError("sink_tokens must be an integer")
    if not 0 <= sink_tokens <= k.shape[1]:
        raise ValueError("sink_tokens must be in [0, Tkv]")
    if sink_start is not None:
        if not isinstance(sink_start, int):
            raise TypeError("sink_start must be an integer or None")
        if not 0 <= sink_start <= k.shape[1]:
            raise ValueError("sink_start must be in [0, Tkv]")
        if sink_start + sink_tokens > k.shape[1]:
            raise ValueError("sink_start + sink_tokens must be <= Tkv")
    return tuple(torch.cuda.get_device_capability(q.device))


def _validate_key_bias(k, key_bias, key_bias_range, sink_start, sink_tokens):
    if key_bias is None:
        if key_bias_range is not None:
            raise ValueError("key_bias_range requires key_bias")
        return None, None
    if not torch.is_tensor(key_bias) or key_bias.ndim != 1 or key_bias.shape[0] != k.shape[1]:
        raise ValueError("key_bias must have shape [Tkv]")
    if not key_bias.dtype.is_floating_point:
        raise TypeError("key_bias must be floating point natural-log measure")
    if key_bias.device != k.device:
        raise ValueError("key_bias must be on the K/V CUDA device")
    if not isinstance(key_bias_range, (tuple, list)) or len(key_bias_range) != 2:
        raise ValueError("weighted Sol-Attn requires key_bias_range=(start, stop)")
    start, stop = key_bias_range
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= k.shape[1]:
        raise ValueError("key_bias_range must be a nonempty interval within Tkv")
    sink_blocks = _sink_block_range(k.shape[1], sink_start, sink_tokens)
    bias_blocks = (start // BLOCK_SIZE, (stop + BLOCK_SIZE - 1) // BLOCK_SIZE)
    if sink_blocks[0] > bias_blocks[0] or sink_blocks[1] < bias_blocks[1]:
        raise ValueError("every active key-bias block must be exact-covered by the sink range")
    return key_bias.to(dtype=torch.float32).contiguous(), (start, stop)


@functools.lru_cache(maxsize=1)
def _cute_runtime_available() -> bool:
    """Whether the optional CuTe DSL runtime can be imported."""

    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _backend_for_arch(
    arch: tuple[int, int],
    *,
    cute_available: bool | None = None,
) -> str:
    """Select CuTe when specialized and available, otherwise Triton."""

    if arch[0] < 8:
        raise RuntimeError(
            "Sol-Attn requires an NVIDIA GPU with compute capability >= 8.0; "
            f"got SM{arch[0]}{arch[1]}"
        )
    cute_backend = _CUTE_BACKENDS.get(arch)
    if cute_backend is not None:
        available = (
            _cute_runtime_available()
            if cute_available is None
            else cute_available
        )
        if available:
            return cute_backend
    return "triton"


def get_sol_attn_backend(device: torch.device | str | int | None = None) -> str:
    """Return the backend selected for ``device`` without compiling it."""

    if device is None:
        device = torch.cuda.current_device()
    return _backend_for_arch(tuple(torch.cuda.get_device_capability(device)))


def _validate_cute(arch, tokens, kv_splits):
    if arch != (9, 0) and kv_splits != 1:
        raise ValueError("kv_splits=2/4 is currently available on SM90 only")
    route_groups = ((tokens + 63) // 64 + 63) // 64
    if kv_splits > route_groups:
        raise ValueError("each KV split must contain at least one N64 route group")


def _stream(device):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _to_cute_tensors(tensors):
    from .common import to_cute_tensor

    return [to_cute_tensor(x) for x in tensors]


def _sink_block_range(tokens, sink_start, sink_tokens):
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if not sink_tokens:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else sink_start
    return (
        start // BLOCK_SIZE,
        (start + sink_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )


def _compile_sm90(
    key,
    tensors,
    scale,
    tokens,
    kv_splits,
    sink_range,
    stream,
):
    import cutlass.cute as cute

    from .sm90 import make_kernel

    operator = make_kernel(tokens, kv_splits)
    args = _to_cute_tensors(tensors)
    compiled = cute.compile(
        operator,
        *args,
        scale,
        sink_range,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _compile_sm100(
    key,
    tensors,
    valid_tokens,
    valid_blocks,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
):
    import cutlass.cute as cute

    from .sm100 import forward

    args = _to_cute_tensors(tensors)
    compiled = cute.compile(
        forward,
        *args,
        valid_tokens,
        valid_blocks,
        scale,
        sink_start_block,
        sink_end_block,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _compile_sm120(
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    key_bias_start,
    key_bias_stop,
    use_key_bias,
    stream,
):
    import cutlass.cute as cute

    from .sm120 import make_kernel

    operator = make_kernel(use_key_bias=use_key_bias)
    args = _to_cute_tensors(tensors)
    compiled = cute.compile(
        operator,
        *args,
        scale,
        sink_start_block,
        sink_end_block,
        key_bias_start,
        key_bias_stop,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _sol_attn_cute(
    q,
    k,
    v,
    *,
    arch,
    scale,
    tau,
    thresh_type,
    kv_splits,
    sink_tokens,
    sink_start,
    key_bias=None,
    key_bias_range=None,
    valid_tokens=None,
):
    from .preprocess import prepare

    batch, capacity_tokens, heads, _ = q.shape
    tokens = capacity_tokens if valid_tokens is None else int(valid_tokens)
    kv_tokens = k.shape[1] if arch == (12, 0) else tokens
    valid_blocks = (kv_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    with torch.cuda.device(q.device):
        kc, vc, threshold = prepare(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            valid_tokens=tokens,
            valid_kv_tokens=kv_tokens,
        )
        output = torch.empty_like(q)
        lse = torch.empty(
            (batch, capacity_tokens, heads),
            device=q.device,
            dtype=torch.float32,
        )
        stream = _stream(q.device)
        # CuTe specializes tensor layouts. Shape alone is insufficient now that a caller may
        # supply views into an interleaved packed QKV buffer; never reuse a compiled contiguous
        # descriptor for a strided view (or vice versa).
        layout_key = tuple(tuple(int(s) for s in x.stride()) for x in (q, k, v))
        use_key_bias = key_bias is not None
        bias_start, bias_stop = key_bias_range if use_key_bias else (0, 0)
        key = (
            q.device.index,
            arch,
            batch,
            capacity_tokens,
            k.shape[1],
            heads,
            kv_splits,
            layout_key,
            use_key_bias,
            (tuple(key_bias.shape), tuple(key_bias.stride())) if use_key_bias else None,
            bias_start,
            bias_stop,
        )

        if arch == (9, 0):
            if sink_tokens:
                sink_start_block, sink_end_block = _sink_block_range(
                    kv_tokens,
                    sink_start,
                    sink_tokens,
                )
                sink_range = sink_start_block | (sink_end_block << 16)
            else:
                sink_range = 0
            tensors = [q, k, v, output, kc, vc, threshold, lse]
            if kv_splits > 1:
                tensors.extend(
                    [
                        torch.empty(
                            (batch, capacity_tokens, kv_splits * heads, 128),
                            device=q.device,
                            dtype=torch.bfloat16,
                        ),
                        torch.empty(
                            (batch, capacity_tokens, kv_splits * heads),
                            device=q.device,
                            dtype=torch.float32,
                        ),
                    ]
                )
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm90(
                    key,
                    tensors,
                    scale,
                    tokens,
                    kv_splits,
                    sink_range,
                    stream,
                )
            else:
                args = _to_cute_tensors(tensors)
            compiled(
                *args,
                scale,
                sink_range,
                stream=stream,
            )
        elif arch in ((10, 0), (10, 3)):
            sink_start_block, sink_end_block = _sink_block_range(
                kv_tokens,
                sink_start,
                sink_tokens,
            )
            tensors = [q, k, v, output, kc, vc, threshold, lse]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm100(
                    key,
                    tensors,
                    tokens,
                    valid_blocks,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    stream,
                )
            else:
                args = _to_cute_tensors(tensors)
            compiled(
                *args,
                tokens,
                valid_blocks,
                scale,
                sink_start_block,
                sink_end_block,
                stream=stream,
            )
        else:
            sink_start_block, sink_end_block = _sink_block_range(
                kv_tokens,
                sink_start,
                sink_tokens,
            )
            # Keep the no-measure arithmetic path identical: the extra tensor is
            # compile-time dead when use_key_bias=False and is never read.
            bias_tensor = (
                key_bias
                if use_key_bias
                else torch.empty((1,), device=q.device, dtype=torch.float32)
            )
            tensors = [q, k, v, output, kc, vc, threshold, lse, bias_tensor]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm120(
                    key,
                    tensors,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    bias_start,
                    bias_stop,
                    use_key_bias,
                    stream,
                )
            else:
                args = _to_cute_tensors(tensors)
            compiled(
                *args,
                scale,
                sink_start_block,
                sink_end_block,
                bias_start,
                bias_stop,
                stream=stream,
            )
    return output[:, :tokens]


def _pad_to_bucket(q, k, v, bucket_size: int):
    """Pad BTHD inputs to a stable compile shape while preserving a logical length."""
    tokens = q.shape[1]
    if bucket_size <= 0:
        raise ValueError("compile_bucket_size must be positive")
    capacity = ((tokens + bucket_size - 1) // bucket_size) * bucket_size
    if capacity == tokens:
        return q, k, v

    # H3's Ulysses path deliberately presents Q/K/V as three views of one packed
    # [..., Q | K | V] allocation. Keep that interleaved stride after padding:
    # the SM100/SM103 TMA path is validated on this layout, while three unrelated
    # contiguous allocations produced invalid long-sequence results.
    packed = torch.empty(
        (q.shape[0], capacity, q.shape[2], 3 * q.shape[3]),
        dtype=q.dtype,
        device=q.device,
    )
    padded = packed.split(q.shape[3], dim=-1)
    # Ulysses produces Q/K/V by splitting one contiguous [..., Q | K | V]
    # allocation. Recover that packed view and copy it in one launch. The
    # fallback retains support for callers that supply independent tensors.
    head_dim = q.shape[3]
    same_storage = (
        q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
        and q.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    )
    packed_offsets = (
        k.storage_offset() == q.storage_offset() + head_dim
        and v.storage_offset() == q.storage_offset() + 2 * head_dim
    )
    packed_strides = (
        q.stride() == k.stride() == v.stride()
        and q.stride(-1) == 1
        and q.stride(-2) == 3 * head_dim
    )
    if same_storage and packed_offsets and packed_strides:
        source = torch.as_strided(
            q,
            size=(q.shape[0], tokens, q.shape[2], 3 * head_dim),
            stride=q.stride(),
            storage_offset=q.storage_offset(),
        )
        packed[:, :tokens].copy_(source)
    else:
        for destination, source in zip(padded, (q, k, v)):
            destination[:, :tokens].copy_(source)
    # Only the bucket tail is unwritten. Clearing it after copying avoids the
    # old full-capacity memset while preserving exactly the same padded values.
    packed[:, tokens:].zero_()
    return padded


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    kv_splits: int = 1,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    compile_bucket_size: int | None = None,
    key_bias: torch.Tensor | None = None,
    key_bias_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Compute noncausal Sol-Attn for innermost-contiguous BF16 BTHD tensors.

    ``sink_start`` and ``sink_tokens`` keep every KV block overlapping the
    corresponding contiguous token range exact for all queries. Omitting
    ``sink_start`` places the range at the token suffix.

    ``key_bias`` is natural-log measure over K/V rows. Weighted execution is
    implemented only by the SM120 CuTe exact path. ``key_bias_range`` declares
    the only active interval; values outside it are never read, and every block
    overlapping that interval must be covered by the exact sink range.
    """

    arch = _validate_inputs(
        q,
        k,
        v,
        thresh_type,
        sink_tokens,
        sink_start,
    )
    key_bias, key_bias_range = _validate_key_bias(
        k, key_bias, key_bias_range, sink_start, sink_tokens
    )
    if kv_splits not in (1, 2, 4):
        raise ValueError("kv_splits must be 1, 2, or 4")
    backend = _backend_for_arch(arch)
    if key_bias is not None and backend != "cute_sm120":
        raise ValueError("key-log-measure Sol-Attn currently requires cute_sm120")
    if q.shape[1] != k.shape[1] and backend != "cute_sm120":
        raise ValueError("Rectangular attention currently requires cute_sm120")
    valid_tokens = q.shape[1]
    if compile_bucket_size is not None:
        if backend != "cute_sm100":
            raise ValueError("compile_bucket_size is currently supported on SM100/SM103 only")
        compile_bucket_size = int(compile_bucket_size)
        if compile_bucket_size % BLOCK_SIZE:
            raise ValueError(f"compile_bucket_size must be a multiple of {BLOCK_SIZE}")
        q, k, v = _pad_to_bucket(q, k, v, compile_bucket_size)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    tau = float(tau)

    if backend == "triton":
        if kv_splits != 1:
            raise ValueError("kv_splits=2/4 is currently available on SM90 only")
        from .triton_ref import sol_attn as triton_sol_attn

        return triton_sol_attn(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            sink_tokens=sink_tokens,
            sink_start=sink_start,
        )

    _validate_cute(arch, k.shape[1], kv_splits)
    return _sol_attn_cute(
        q,
        k,
        v,
        arch=arch,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        kv_splits=kv_splits,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
        key_bias=key_bias,
        key_bias_range=key_bias_range,
        valid_tokens=valid_tokens,
    )


__all__ = ["get_sol_attn_backend", "sol_attn"]
