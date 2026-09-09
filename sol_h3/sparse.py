"""Native attention bridge to the packaged Sana Sol-H3 implementation."""
import math
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
    """Validate an exact prefix interval and describe its overlapping 64-row blocks.

    Sol-H3 only requests prefix sinks beginning at row zero. The final partial
    block is intentionally rounded outward: this makes a few extra keys exact,
    which is semantics-preserving and only slightly more expensive.
    """
    if type(start) is not int or type(tokens) is not int or type(rows) is not int:
        raise RuntimeError("SOL sink geometry must use integer row counts")
    if start != 0:
        raise RuntimeError("Sol-H3 currently supports only a prefix sink beginning at row zero")
    if tokens < 0 or tokens > rows:
        raise RuntimeError("SOL sink row count is outside the current sequence")
    if tokens == 0:
        return [0, 0]
    return [0, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE]


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
            raise RuntimeError(
                f"Sana selected {backend}; SM120 requires CuTe (cutlass.cute and cuda.bindings.driver)"
            )
        # Import lazy dependencies before returning a usable kernel. No compilation here.
        from ._vendor.sol_attn import preprocess  # noqa: F401
        from ._vendor.sol_attn.sm120 import make_kernel  # noqa: F401
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Sana Sol-Attn initialization failed: {exc}") from exc

    def kernel(q, k, v, *, tau, thresh_type="diag", kv_splits=1,
               sink_start=0, sink_tokens=0):
        _sink_blocks(sink_start, sink_tokens, k.shape[1])  # validate prefix geometry
        return sol_attn(q, k, v, scale=q.shape[-1] ** -0.5, tau=float(tau),
                        thresh_type=thresh_type, kv_splits=kv_splits,
                        sink_start=sink_start, sink_tokens=sink_tokens)

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


def _dense_reference(q, k, v, dense_attention):
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


def attention(q, k, v, prefix, config, state, dense_attention=None,
              recompute_prefix_queries=True):
    if (any(x.ndim != 4 for x in (q, k, v)) or k.shape != v.shape
            or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]
            or q.shape[0] != 1 or q.shape[-1] != 128
            or q.shape[2] == 0 or k.shape[2] == 0):
        raise RuntimeError("SOL requires Q [1, heads, Tq, 128], KV [1, heads, Tkv, 128]")
    _sink_blocks(0, prefix, k.shape[2])
    if recompute_prefix_queries and q.shape != k.shape:
        raise RuntimeError("Prefix query recomputation requires the original square packed sequence")
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
    key = (q.device, q.dtype, layout_key)
    calibrating = key not in state.sparse_verified
    gate_started = time.perf_counter() if calibrating else None
    if calibrating:
        # The all-selected gate now validates the exact strided layout that the
        # hot path will reuse. error_metrics() already performs scalar reads, so
        # this adds no new steady-state synchronization point.
        got = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                           sink_start=0, sink_tokens=kb.shape[1])
        want = _dense_reference(q, k, v, None)
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
            **metrics,
        })
        del got, want
    out = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                       sink_start=0, sink_tokens=prefix)
    # Normal H3 replaces non-video query rows with the inherited dense provider.
    # VDN local Q contains only requested rows; sink_rows describes K/V only.
    if prefix and recompute_prefix_queries:
        out[:, :prefix] = _dense_reference(q[:, :, :prefix], k, v, dense_attention)
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], -1)
