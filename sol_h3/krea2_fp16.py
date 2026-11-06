"""Native FP16 Sol-Attn route for Krea 2, through ComfyUI Kitchen.

The packaged Sana CuTe SM120 kernel accepts BF16 only. Krea 2's native FP16
inference must not silently narrow its Q/K/V to BF16; Kitchen implements
the same Sol block-routing family with native FP16 inputs. Its explicitly
protected text K/V blocks and dense text query blocks are mandatory.
"""
import torch
import torch.nn.functional as F

from .sparse import KernelUnavailable, error_metrics

BLOCK = 64
GATE_ROWS = 512
# Kitchen upstream tests bound fully routed Sol relative L2 below 0.016.
# Keep this backend-specific margin separate from Sana CuTe's 0.005 gate.
FP16_PILOT_REL_L2_LIMIT = 0.02


def available(device):
    """A capability probe, not a claim that sparse inference was executed."""
    if device.type != "cuda":
        return False
    try:
        import comfy_kitchen as ck
    except (ImportError, OSError):
        return False
    check = getattr(ck, "sol_attn_is_available", None)
    return callable(check) and bool(check(device))


def attention(q, k, v, prefix, settings, state):
    """Return [1, tokens, heads*128] from native FP16 B/H/T/D Q/K/V."""
    if any(t.dtype != torch.float16 or t.shape != q.shape or t.device != q.device
           or t.ndim != 4 or t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("Krea FP16 Sol requires matching FP16 BHND Q/K/V with contiguous head dimension")
    if q.shape[0] != 1 or q.shape[-1] != 128 or not 0 < prefix <= q.shape[2]:
        raise ValueError("Krea FP16 Sol requires single-batch head_dim 128 and a text prefix")
    if state.kernel_failure:
        raise KernelUnavailable(state.kernel_failure)
    if not available(q.device):
        state.kernel_failure = "ComfyUI Kitchen native FP16 Sol kernel is unavailable on this device"
        raise KernelUnavailable(state.kernel_failure)

    import comfy_kitchen as ck
    blocks = (prefix + BLOCK - 1) // BLOCK
    exact = [0, blocks]
    qb, kb, vb = (t.transpose(1, 2) for t in (q, k, v))
    # Fail closed on a broken FP16 attention kernel. This small all-selected
    # arithmetic pilot checks the actual FP16 inputs against native FP16 SDPA.
    # It does NOT certify the approximation quality of the sparse route.
    key = ("kitchen_fp16_gate", q.device, q.dtype, tuple(q.shape),
           tuple(t.stride() for t in (q, k, v)))
    if key not in state.sparse_verified:
        size = min(GATE_ROWS, q.shape[2])
        trial = [x[:, :size] for x in (qb, kb, vb)]
        all_blocks = [0, (size + BLOCK - 1) // BLOCK]
        selected = ck.sol_attn(
            *trial, tau=settings.tau, sink_blocks=all_blocks,
            sink_q=all_blocks, token_aug=0,
        )
        expected = F.scaled_dot_product_attention(
            *(x.transpose(1, 2) for x in trial),
        ).transpose(1, 2)
        if (selected.shape != expected.shape or selected.dtype != expected.dtype or
                selected.device != expected.device):
            raise RuntimeError("ComfyUI Kitchen FP16 Sol all-selected output contract changed")
        metrics = error_metrics(selected, expected)
        if not (metrics["finite"] and metrics["rel_l2"] <= FP16_PILOT_REL_L2_LIMIT
                and metrics["max_abs"] <= metrics["catastrophic_max_abs_limit"]):
            state.kernel_failure = "FP16 Kitchen arithmetic pilot failed"
            raise KernelUnavailable(f"Krea FP16 Sol all-selected arithmetic pilot failed: {metrics}")
        state.sparse_verified.add(key)
        state.gates.append({
            "backend": "comfy_kitchen_fp16", "gate": "all_selected_fp16_pilot",
            "gate_tokens": size, "full_sequence_tokens": q.shape[2], **metrics,
        })

    out = ck.sol_attn(
        qb, kb, vb, tau=settings.tau, sink_blocks=exact,
        sink_q=exact, token_aug=settings.extra_tokens,
    )
    if out.shape != qb.shape or out.dtype != q.dtype or out.device != q.device:
        raise RuntimeError("ComfyUI Kitchen FP16 Sol returned unexpected shape/dtype/device")
    # Kitchen's sink_q makes these rows full-attention, but its CUDA math
    # is INT8-quantized. Recompute only authored text queries with native
    # FP16 SDPA over ALL original K/V to keep text evolution numerically dense.
    out[:, :prefix] = F.scaled_dot_product_attention(
        q[:, :, :prefix], k, v,
    ).transpose(1, 2)
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], q.shape[1] * q.shape[3])
