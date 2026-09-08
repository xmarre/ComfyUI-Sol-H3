"""Native attention bridge to the packaged Sana Sol-H3 implementation."""
import torch
import torch.nn.functional as F


BLOCK_SIZE = 64


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
        _sink_blocks(sink_start, sink_tokens, q.shape[1])  # validate prefix geometry
        return sol_attn(q, k, v, scale=q.shape[-1] ** -0.5, tau=float(tau),
                        thresh_type=thresh_type, kv_splits=kv_splits,
                        sink_start=sink_start, sink_tokens=sink_tokens)

    kernel.backend_name = backend
    kernel.source_tree_verified = True
    kernel.block_size = BLOCK_SIZE
    return kernel


def error_metrics(got, want):
    delta = got.float() - want.float()
    return {"max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean()),
            "rel_l2": float(torch.linalg.vector_norm(delta) /
                            torch.linalg.vector_norm(want.float()).clamp_min(1e-12))}


def _dense_reference(q, k, v, dense_attention):
    if dense_attention is not None:
        out = dense_attention(q, k, v)
        expected = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        if out.shape != expected:
            raise RuntimeError(f"Dense SOL reference returned {tuple(out.shape)}, expected {expected}")
        return out
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def attention(q, k, v, prefix, config, state, dense_attention=None,
              recompute_prefix_queries=True):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[-1] != 128:
        raise RuntimeError("SOL requires matching QKV [1, heads, packed rows, 128]")
    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
        raise RuntimeError("SOL requires BF16 QKV on the same device")
    if state.kernel is None:
        failure = getattr(state, "kernel_failure", None)
        if failure:
            raise KernelUnavailable(failure)
        try:
            state.kernel = load_kernel(q.device)
        except RuntimeError as exc:
            state.kernel_failure = str(exc)
            raise KernelUnavailable(str(exc)) from exc
        state.kernel_device = q.device
    elif q.device != state.kernel_device:
        raise RuntimeError("SOL compute device changed within a sampling request")
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    key = (q.device, q.dtype, tuple(q.shape))
    if key not in state.sparse_verified:
        got = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                           sink_start=0, sink_tokens=qb.shape[1])
        want = _dense_reference(q, k, v, None)
        metrics = error_metrics(got, want)
        limits = {"max_abs": 0.15 if qb.shape[1] >= 32768 else 0.08,
                  "mean_abs": 0.002, "rel_l2": 0.005}
        if not all(metrics[n] <= limits[n] for n in limits):
            raise RuntimeError(f"SOL all-selected arithmetic gate failed: {metrics}")
        state.sparse_verified.add(key)
        state.gates.append({"shape": list(q.shape), "backend": getattr(state.kernel, "backend_name", "test_substitute"), **metrics})
        del got, want
    out = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                       sink_start=0, sink_tokens=prefix)
    # Normal H3 replaces non-video query rows with the inherited dense provider.
    # A VDN v2 expanded square domain discards those extra query outputs entirely,
    # so it can retain exact sink keys without paying for unused dense query rows.
    if prefix and recompute_prefix_queries:
        out[:, :prefix] = _dense_reference(q[:, :, :prefix], k, v, dense_attention)
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], -1)
