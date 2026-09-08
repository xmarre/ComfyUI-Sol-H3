"""Native attention bridge to ComfyUI's maintained comfy-kitchen Sol-Attn kernel."""
import torch
import torch.nn.functional as F


BLOCK_SIZE = 64


class KernelUnavailable(RuntimeError):
    """The native provider can execute safely without this optional sparse kernel."""


def _sink_blocks(start, tokens, rows):
    """Convert an exact row interval to comfy-kitchen's exact 64-row block interval.

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
    """Return the ComfyUI-installed Sol-Attn kernel with the Sol-H3 call contract.

    Current ComfyUI ships the compiled Sol-Attn implementation in
    ``comfy-kitchen``. No external Sana checkout, PYTHONPATH mutation, runtime
    download or duplicate kernel installation is required.
    """
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("This experimental SOL integration currently targets single-GPU SM120 only")
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise RuntimeError(
            "ComfyUI's comfy-kitchen package is unavailable; update the current ComfyUI environment"
        ) from exc
    available = getattr(ck, "sol_attn_is_available", None)
    sol_attn = getattr(ck, "sol_attn", None)
    if not callable(available) or not callable(sol_attn):
        raise RuntimeError(
            "The installed comfy-kitchen does not expose the required sol_attn API; update ComfyUI/comfy-kitchen"
        )
    if not available(device):
        raise RuntimeError("The installed comfy-kitchen has no compiled sol_attn kernel for this GPU")

    def kernel(q, k, v, *, tau, thresh_type="diag", kv_splits=1,
               sink_start=0, sink_tokens=0):
        if thresh_type != "diag":
            raise RuntimeError(f"Unsupported comfy-kitchen SOL threshold policy {thresh_type!r}")
        if kv_splits != 1:
            raise RuntimeError("Sol-H3 currently requires one logical KV partition")
        sink_blocks = _sink_blocks(sink_start, sink_tokens, q.shape[1])
        return sol_attn(
            q, k, v,
            tau=float(tau),
            scale=None,
            sink_blocks=sink_blocks,
            sink_q=[0, 0],
            topk_ratio=0.0,
            tail=True,
            token_aug=0,
        )

    kernel.backend_name = "comfy_kitchen.sol_attn"
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
        state.gates.append({"shape": list(q.shape), "backend": "comfy_kitchen.sol_attn", **metrics})
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
