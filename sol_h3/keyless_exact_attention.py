"""Experimental no-global-route Keyless all-selected attention primitive.

Phase K2 of the native Keyless SM120 design. The CUDA path reads raw projected V,
derives the RMSNorm + H3 split-half-RoPE routing view one bounded N64 tile at a
time, scores Q against that routing view, and accumulates the original raw V.

The primitive is intentionally not wired into production Sol dispatch. It exists
to establish all-selected arithmetic and allocation behavior before sparse routing
or vendored CuTe mainloop changes are promoted.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .keyless_route_summary import (
    HEAD_DIM,
    NORM_EPS,
    ROPE_HALF_DIM,
    ROPE_ROT_DIM,
    materialized_route_reference,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


BLOCK_M = 64
BLOCK_N = 64
CONTRACT = "sol-h3-keyless-exact-allselected-v2"
CANONICAL_SCALE = HEAD_DIM ** -0.5


if triton is not None:

    @triton.jit
    def _keyless_exact_allselected_kernel(
        q_ptr,
        v_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        q_rows,
        v_rows,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ct: tl.constexpr,
        stride_cd: tl.constexpr,
        stride_st: tl.constexpr,
        stride_sd: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        scale,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        head_dim: tl.constexpr,
        rope_half: tl.constexpr,
        rope_rot: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        head = tl.program_id(1)

        m = q_block * block_m + tl.arange(0, block_m)
        d = tl.arange(0, head_dim)
        valid_m = m < q_rows

        q_offsets = (
            m[:, None] * stride_qt
            + head * stride_qh
            + d[None, :] * stride_qd
        )
        q = tl.load(
            q_ptr + q_offsets,
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        weight = tl.load(weight_ptr + d).to(tl.float32)
        acc = tl.zeros((block_m, head_dim), dtype=tl.float32)
        row_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((block_m,), dtype=tl.float32)

        for start_n in tl.range(0, v_rows, block_n):
            n = start_n + tl.arange(0, block_n)
            valid_n = n < v_rows

            v_offsets = (
                n[:, None] * stride_vt
                + head * stride_vh
                + d[None, :] * stride_vd
            )
            raw_bf16 = tl.load(
                v_ptr + v_offsets,
                mask=valid_n[:, None],
                other=0.0,
            )
            raw = raw_bf16.to(tl.float32)

            mean_square = tl.sum(raw * raw, axis=1) / head_dim
            inv_rms = tl.rsqrt(mean_square + eps)
            norm = (
                raw * inv_rms[:, None] * weight[None, :]
            ).to(tl.bfloat16)

            rotating = d < rope_rot
            first_half = d < rope_half
            partner_d = tl.where(
                first_half,
                d + rope_half,
                d - rope_half,
            )
            partner_offsets = (
                n[:, None] * stride_vt
                + head * stride_vh
                + partner_d[None, :] * stride_vd
            )
            partner_raw = tl.load(
                v_ptr + partner_offsets,
                mask=valid_n[:, None] & rotating[None, :],
                other=0.0,
            ).to(tl.float32)
            partner_weight = tl.load(
                weight_ptr + partner_d,
                mask=rotating,
                other=0.0,
            ).to(tl.float32)
            partner_norm = (
                partner_raw * inv_rms[:, None] * partner_weight[None, :]
            ).to(tl.bfloat16)

            pair_d = tl.where(
                first_half,
                d,
                d - rope_half,
            )
            # Match comfy-kitchen CUDA fused RMS+RoPE arithmetic: the
            # normalized/scaled values are materialized to BF16 first, then the
            # 2x2 rotation executes in FP32 and only the routed result is cast
            # back to BF16.
            cos_values = tl.load(
                cos_ptr
                + n[:, None] * stride_ct
                + pair_d[None, :] * stride_cd,
                mask=valid_n[:, None] & rotating[None, :],
                other=1.0,
            ).to(tl.float32)
            sin_values = tl.load(
                sin_ptr
                + n[:, None] * stride_st
                + pair_d[None, :] * stride_sd,
                mask=valid_n[:, None] & rotating[None, :],
                other=0.0,
            ).to(tl.float32)
            norm_math = norm.to(tl.float32)
            partner_math = partner_norm.to(tl.float32)

            first = norm_math * cos_values - partner_math * sin_values
            second = partner_math * sin_values + norm_math * cos_values
            route = tl.where(
                first_half[None, :],
                first,
                tl.where(rotating[None, :], second, norm),
            ).to(tl.bfloat16)

            scores = tl.dot(q, tl.trans(route), out_dtype=tl.float32) * scale
            scores = tl.where(
                valid_m[:, None] & valid_n[None, :],
                scores,
                -float("inf"),
            )

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            alpha = tl.exp(row_max - new_max)
            alpha = tl.where(valid_m, alpha, 0.0)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(
                valid_m[:, None] & valid_n[None, :],
                probabilities,
                0.0,
            )

            acc = (
                acc * alpha[:, None]
                + tl.dot(
                    probabilities.to(tl.bfloat16),
                    raw_bf16.to(tl.bfloat16),
                    out_dtype=tl.float32,
                )
            )
            row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
            row_max = new_max

        denom = tl.where(valid_m & (row_sum > 0.0), row_sum, 1.0)
        result = acc / denom[:, None]
        out_offsets = (
            m[:, None] * stride_ot
            + head * stride_oh
            + d[None, :] * stride_od
        )
        tl.store(
            out_ptr + out_offsets,
            result.to(tl.bfloat16),
            mask=valid_m[:, None],
        )

else:
    _keyless_exact_allselected_kernel = None


def _validate_common(
    q: torch.Tensor,
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    scale: float,
) -> None:
    if not torch.is_tensor(q) or q.ndim != 3 or q.shape[-1] != HEAD_DIM:
        raise ValueError("Keyless exact attention requires Q [Tq,H,128]")
    if not torch.is_tensor(v) or v.ndim != 3 or v.shape[-1] != HEAD_DIM:
        raise ValueError("Keyless exact attention requires V [Tv,H,128]")
    if q.shape[0] <= 0 or v.shape[0] <= 0 or q.shape[1] <= 0:
        raise ValueError("Keyless exact attention requires nonempty Q/V rows and heads")
    if q.shape[1] != v.shape[1]:
        raise ValueError("Keyless exact attention requires matching Q/V head counts")
    if not torch.is_tensor(norm_weight) or norm_weight.shape != (HEAD_DIM,):
        raise ValueError("Keyless exact attention requires route_norm.weight [128]")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("Keyless exact attention epsilon must be finite and positive")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("Keyless exact attention scale must be finite and positive")
    expected_rope = (1, int(v.shape[0]), 1, ROPE_HALF_DIM, 2, 2)
    if not torch.is_tensor(rope_freqs) or tuple(rope_freqs.shape) != expected_rope:
        raise ValueError(
            f"Keyless exact attention requires rope_freqs {expected_rope}, got "
            f"{tuple(getattr(rope_freqs, 'shape', ()))}"
        )
    if (
        q.device != v.device
        or norm_weight.device != v.device
        or rope_freqs.device != v.device
    ):
        raise ValueError("Keyless exact attention tensors must share one device")
    if q.stride(-1) != 1 or v.stride(-1) != 1 or norm_weight.stride(0) != 1:
        raise ValueError("Keyless exact attention requires contiguous head channels")


def materialized_exact_reference(
    q: torch.Tensor,
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    *,
    scale: float = CANONICAL_SCALE,
) -> torch.Tensor:
    """Dense materialized oracle: score with route(V), retrieve original raw V."""
    _validate_common(q, v, norm_weight, eps, rope_freqs, scale)
    route = materialized_route_reference(v, norm_weight, eps, rope_freqs)
    q4 = q.transpose(0, 1).unsqueeze(0)
    route4 = route.transpose(0, 1).unsqueeze(0)
    v4 = v.transpose(0, 1).unsqueeze(0)
    result = F.scaled_dot_product_attention(
        q4,
        route4,
        v4,
        dropout_p=0.0,
        scale=float(scale),
    )
    return result.squeeze(0).transpose(0, 1)


def exact_attention(
    q: torch.Tensor,
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    *,
    scale: float = CANONICAL_SCALE,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the experimental all-selected CUDA path without global route storage."""
    _validate_common(q, v, norm_weight, eps, rope_freqs, scale)
    if _keyless_exact_allselected_kernel is None:
        raise RuntimeError("Keyless exact attention requires Triton")
    if q.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("native Keyless exact attention requires BF16 Q and V")
    if norm_weight.dtype != torch.bfloat16:
        raise TypeError("native Keyless exact attention requires BF16 route norm weight")
    if q.device.type != "cuda":
        raise RuntimeError("native Keyless exact attention requires CUDA")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("native Keyless exact attention K2 currently targets SM120 only")
    if float(eps) != NORM_EPS:
        raise ValueError(
            f"native Keyless exact attention currently requires eps={NORM_EPS}"
        )
    if float(scale) != CANONICAL_SCALE:
        raise ValueError(
            f"native Keyless exact attention currently requires scale={CANONICAL_SCALE}"
        )

    if out is None:
        out = torch.empty_like(q)
    elif (
        not torch.is_tensor(out)
        or out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or out.stride(-1) != 1
    ):
        raise ValueError(
            "Keyless exact attention out must match Q shape/dtype/device/head layout"
        )

    q_rows, heads, _ = q.shape
    v_rows = int(v.shape[0])
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]
    grid = (triton.cdiv(q_rows, BLOCK_M), heads)

    _keyless_exact_allselected_kernel[grid](
        q,
        v,
        norm_weight,
        cos,
        sin,
        out,
        q_rows,
        v_rows,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        float(scale),
        eps=float(eps),
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        head_dim=HEAD_DIM,
        rope_half=ROPE_HALF_DIM,
        rope_rot=ROPE_ROT_DIM,
        num_warps=8,
        num_stages=2,
    )
    return out


__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "CANONICAL_SCALE",
    "CONTRACT",
    "exact_attention",
    "materialized_exact_reference",
]
