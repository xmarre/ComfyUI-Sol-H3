"""Native attention bridge to the packaged Sana Sol-H3 implementation."""
import math
import sys
import time

import torch
import torch.nn.functional as F


BLOCK_SIZE = 64
ARITH_MEAN_ABS_LIMIT = 0.002
ARITH_REL_L2_LIMIT = 0.005
# The all-selected calibration is a numerical sanity check for an approximate
# attention kernel, not a bitwise-parity test. Aggregate error is the primary
# contract. A scale-aware peak guard remains only to catch gross finite
# corruption that can be diluted by a very large tensor norm.
ARITH_CATASTROPHIC_MAX_FLOOR = 0.5
ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER = 4.0


class KernelUnavailable(RuntimeError):
    """The native provider can execute safely without this optional sparse kernel."""


def _sink_blocks(start, tokens, rows):
    """Validate an exact interval and describe its overlapping 64-row blocks."""
    if type(start) is not int or type(tokens) is not int or type(rows) is not int:
        raise RuntimeError("SOL sink geometry must use integer row counts")
    if start < 0 or tokens < 0 or start + tokens > rows:
        raise RuntimeError("SOL sink row interval is outside the current sequence")
    if tokens == 0:
        blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
        return [blocks, blocks]
    return [start // BLOCK_SIZE, (start + tokens + BLOCK_SIZE - 1) // BLOCK_SIZE]


def _validate_key_bias(key_bias, key_bias_range, rows):
    if key_bias is None:
        if key_bias_range is not None:
            raise RuntimeError("SOL key_bias_range requires key_bias")
        return None
    if (
        not torch.is_tensor(key_bias)
        or key_bias.ndim != 1
        or key_bias.shape[0] != rows
        or not key_bias.dtype.is_floating_point
    ):
        raise RuntimeError("SOL key bias must be a floating vector with one value per K/V row")
    if not isinstance(key_bias_range, (tuple, list)) or len(key_bias_range) != 2:
        raise RuntimeError("SOL weighted attention requires an explicit key-bias interval")
    start, stop = key_bias_range
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= rows:
        raise RuntimeError("SOL key-bias interval is invalid")
    return int(start), int(stop)


def load_kernel(device):
    """Load the verified node-local public API; SM120 requires its CuTe backend."""
    from .provenance import verify_source
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("This experimental SOL integration currently targets single-GPU SM120 only")
    try:
        verify_source()
        from ._vendor.sol_attn import get_sol_attn_backend, sol_attn
        backend = get_sol_attn_backend(device)
        if backend != "cute_sm120":
            if sys.platform == "win32":
                raise RuntimeError(
                    "native Windows cannot execute the required NVIDIA CUTLASS CuTe DSL backend; "
                    "SM120 SOL attention currently requires Linux/WSL2"
                )
            raise RuntimeError(
                f"Sana selected {backend}; SM120 requires CuTe (cutlass.cute and cuda.bindings.driver)"
            )
        # Import lazy dependencies before returning a usable kernel. No compilation here.
        from ._vendor.sol_attn import preprocess  # noqa: F401
        from ._vendor.sol_attn.sm120 import make_kernel  # noqa: F401
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Sana Sol-Attn initialization failed: {exc}") from exc

    def kernel(q, k, v, *, tau, thresh_type="diag", kv_splits=1,
               sink_start=0, sink_tokens=0, key_bias=None, key_bias_range=None):
        sink = _sink_blocks(sink_start, sink_tokens, k.shape[1])
        bias_interval = _validate_key_bias(key_bias, key_bias_range, k.shape[1])
        if key_bias is not None:
            if key_bias.device != q.device:
                raise RuntimeError("SOL key bias must share the Q/K/V device")
            bias_blocks = _sink_blocks(
                bias_interval[0], bias_interval[1] - bias_interval[0], k.shape[1]
            )
            if not (sink[0] <= bias_blocks[0] and sink[1] >= bias_blocks[1]):
                raise RuntimeError("SOL nonzero key-bias interval must be covered by exact sink blocks")
        return sol_attn(
            q,
            k,
            v,
            scale=q.shape[-1] ** -0.5,
            tau=float(tau),
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
            key_bias=key_bias,
            key_bias_range=bias_interval,
        )

    kernel.backend_name = backend
    kernel.source_tree_verified = True
    kernel.block_size = BLOCK_SIZE
    return kernel


def error_metrics(got, want):
    """Return aggregate and scale-aware arithmetic calibration metrics.

    Sana's SM120 implementation is explicitly a mixed approximate/exact
    attention mainloop. In all-selected calibration mode, isolated BF16/CuTe
    peaks can therefore be much larger than the tensor-wide error without
    indicating a bad route or broken kernel. Mean absolute and relative L2
    error are the primary contract; max error remains telemetry plus a broad
    scale-aware corruption guard.
    """
    got_f = got.float()
    want_f = want.float()
    delta = got_f - want_f
    abs_delta = delta.abs()
    finite = bool(torch.isfinite(got_f).all().item() and
                  torch.isfinite(want_f).all().item() and
                  torch.isfinite(delta).all().item())
    if not finite:
        return {
            "finite": False,
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "rel_l2": math.inf,
            "reference_peak_abs": math.nan,
            "catastrophic_max_abs_limit": math.nan,
        }

    reference_peak_abs = float(want_f.abs().max().item())
    max_abs = float(abs_delta.max().item())
    mean_abs = float(abs_delta.mean().item())
    rel_l2 = float((torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(want_f).clamp_min(1e-12)).item())
    catastrophic_limit = max(
        ARITH_CATASTROPHIC_MAX_FLOOR,
        ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER * reference_peak_abs,
    )
    return {
        "finite": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "reference_peak_abs": reference_peak_abs,
        "catastrophic_max_abs_limit": catastrophic_limit,
    }


def arithmetic_gate_passes(metrics):
    """Accept close aggregate arithmetic while rejecting broad/gross corruption."""
    return bool(
        metrics.get("finite")
        and metrics["mean_abs"] <= ARITH_MEAN_ABS_LIMIT
        and metrics["rel_l2"] <= ARITH_REL_L2_LIMIT
        and metrics["max_abs"] <= metrics["catastrophic_max_abs_limit"]
    )


def _dense_reference(q, k, v, dense_attention, key_bias=None):
    if key_bias is not None:
        try:
            from comfy import attention_measure as core_measure
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("weighted SOL reference requires ComfyUI attention-measure support") from exc
        out = core_measure.weighted_dense(
            q,
            k,
            v,
            q.shape[1],
            key_bias=key_bias,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=True,
        )
        return out.transpose(1, 2)
    if dense_attention is not None:
        out = dense_attention(q, k, v)
        expected = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        if out.shape != expected:
            raise RuntimeError(f"Dense SOL reference returned {tuple(out.shape)}, expected {expected}")
        return out
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def _bthd_layout_key(*tensors):
    """Describe the exact BTHD layouts consumed by CuTe for calibration identity."""
    return tuple((tuple(x.shape), tuple(x.stride())) for x in tensors)


def attention(
    q,
    k,
    v,
    prefix,
    config,
    state,
    dense_attention=None,
    recompute_prefix_queries=True,
    *,
    key_bias=None,
    key_bias_range=None,
    measure_identity=None,
):
    if (any(x.ndim != 4 for x in (q, k, v)) or k.shape != v.shape
            or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]
            or q.shape[0] != 1 or q.shape[-1] != 128
            or q.shape[2] == 0 or k.shape[2] == 0):
        raise RuntimeError("SOL requires Q [1, heads, Tq, 128], KV [1, heads, Tkv, 128]")
    _sink_blocks(0, prefix, k.shape[2])
    bias_interval = _validate_key_bias(key_bias, key_bias_range, k.shape[2])
    if key_bias is not None and key_bias.device != q.device:
        raise RuntimeError("SOL key bias must share the Q/K/V device")
    if key_bias is None and measure_identity is not None:
        raise RuntimeError("SOL measure identity requires a key bias")
    if recompute_prefix_queries and prefix > q.shape[2]:
        raise RuntimeError("Prefix query recomputation exceeds the available Q rows")
    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
        raise RuntimeError("SOL requires BF16 QKV on the same device")
    kernel_loader_s = 0.0
    if state.kernel is None:
        failure = getattr(state, "kernel_failure", None)
        if failure:
            raise KernelUnavailable(failure)
        started = time.perf_counter()
        try:
            state.kernel = load_kernel(q.device)
        except RuntimeError as exc:
            state.kernel_failure = str(exc)
            raise KernelUnavailable(str(exc)) from exc
        kernel_loader_s = time.perf_counter() - started
        state.kernel_device = q.device
    elif q.device != state.kernel_device:
        raise RuntimeError("SOL compute device changed within a sampling request")

    # CuTe accepts innermost-contiguous strided BTHD tensors. Preserve the exact
    # transposed views instead of materialising Q/K/V copies on every SOL call.
    qb, kb, vb = (x.transpose(1, 2) for x in (q, k, v))
    if any(x.stride(-1) != 1 for x in (qb, kb, vb)):
        raise RuntimeError("SOL BTHD bridge requires a contiguous head dimension")

    layout_key = _bthd_layout_key(qb, kb, vb)
    key = (q.device, q.dtype, layout_key, measure_identity, bias_interval)
    calibrating = key not in state.sparse_verified
    gate_started = time.perf_counter() if calibrating else None
    if calibrating:
        # The all-selected gate validates the exact strided layout and, in
        # weighted mode, the exact same key-log-measure used by the hot path.
        got = state.kernel(
            qb,
            kb,
            vb,
            tau=config.tau,
            thresh_type="diag",
            kv_splits=1,
            sink_start=0,
            sink_tokens=kb.shape[1],
            key_bias=key_bias,
            key_bias_range=bias_interval,
        )
        want = _dense_reference(q, k, v, None, key_bias=key_bias)
        metrics = error_metrics(got, want)
        gate_wall_s = time.perf_counter() - gate_started
        if not arithmetic_gate_passes(metrics):
            raise RuntimeError(f"SOL all-selected arithmetic gate failed: {metrics}")
        state.sparse_verified.add(key)
        state.gates.append({
            "shape": list(q.shape),
            "kv_shape": list(k.shape),
            "backend": getattr(state.kernel, "backend_name", "test_substitute"),
            "kernel_loader_s": kernel_loader_s,
            "gate_wall_s": gate_wall_s,
            "materialized_qkv_bytes": 0,
            "bthd_qkv_bytes": sum(x.numel() * x.element_size() for x in (qb, kb, vb)),
            "bthd_strides": [list(x.stride()) for x in (qb, kb, vb)],
            "attention_measure_identity": measure_identity,
            "key_bias_range": list(bias_interval) if bias_interval is not None else None,
            **metrics,
        })
        del got, want
    out = state.kernel(
        qb,
        kb,
        vb,
        tau=config.tau,
        thresh_type="diag",
        kv_splits=1,
        sink_start=0,
        sink_tokens=prefix,
        key_bias=key_bias,
        key_bias_range=bias_interval,
    )
    # Normal H3 replaces non-video query rows with exact dense attention.
    # In weighted mode that replacement must use the same key measure.
    if prefix and recompute_prefix_queries:
        out[:, :prefix] = _dense_reference(
            q[:, :, :prefix],
            k,
            v,
            dense_attention,
            key_bias=key_bias,
        )
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], -1)
