"""Experimental Keyless raw-V route-summary primitive.

This is phase K1 of the native Keyless SM120 design.  It produces only one routed
centroid and one raw-value sum per V64 block.  It never materializes the complete
[T,H,128] routing tensor on the native path.

The primitive is deliberately not wired into production Sol dispatch yet.  CUDA
arithmetic must be compared with the materialized Keyless oracle on real SM120
before the provider may select it.
"""
from __future__ import annotations

import math

import torch

try:  # Optional until this experimental primitive is executed.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency
    triton = None
    tl = None


BLOCK_SIZE = 64
HEAD_DIM = 128
ROPE_HALF_DIM = 48
ROPE_ROT_DIM = 96
NORM_EPS = 1e-5
CONTRACT = "sol-h3-keyless-route-summary-v2"


if triton is not None:

    @triton.jit
    def _reduce_route_vc_kernel(
        v_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        rc_ptr,
        vc_ptr,
        rows,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ct: tl.constexpr,
        stride_cd: tl.constexpr,
        stride_st: tl.constexpr,
        stride_sd: tl.constexpr,
        stride_rcb: tl.constexpr,
        stride_rch: tl.constexpr,
        stride_rcd: tl.constexpr,
        stride_vcb: tl.constexpr,
        stride_vch: tl.constexpr,
        stride_vcd: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        rope_half: tl.constexpr,
        rope_rot: tl.constexpr,
    ):
        block = tl.program_id(0)
        head = tl.program_id(1)
        row_offsets = block * block_size + tl.arange(0, block_size)
        d_offsets = tl.arange(0, head_dim)
        valid_rows = row_offsets < rows

        v_offsets = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + d_offsets[None, :] * stride_vd
        )
        raw = tl.load(
            v_ptr + v_offsets,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)

        # Match Keyless rms_norm: FP32 variance and scale, then cast back to V
        # dtype before positional rotation.
        mean_square = tl.sum(raw * raw, axis=1) / head_dim
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(weight_ptr + d_offsets).to(tl.float32)
        norm = (raw * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotated = d_offsets < rope_rot
        first_half = d_offsets < rope_half
        partner_d = tl.where(first_half, d_offsets + rope_half, d_offsets - rope_half)
        partner_offsets = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + partner_d[None, :] * stride_vd
        )
        partner_raw = tl.load(
            v_ptr + partner_offsets,
            mask=valid_rows[:, None] & rotated[None, :],
            other=0.0,
        ).to(tl.float32)
        partner_weight = tl.load(
            weight_ptr + partner_d,
            mask=rotated,
            other=0.0,
        ).to(tl.float32)
        partner_norm = (
            partner_raw * inv_rms[:, None] * partner_weight[None, :]
        ).to(tl.bfloat16)

        pair_d = tl.where(first_half, d_offsets, d_offsets - rope_half)
        # Match comfy-kitchen CUDA fused RMS+RoPE semantics. HasRms selects
        # ComputeType=float: normalized/scaled values are first materialized to
        # BF16, then both those values and the rotation matrix are promoted to
        # FP32 for the 2x2 rotation before the route is cast back to BF16.
        cos_values = tl.load(
            cos_ptr
            + row_offsets[:, None] * stride_ct
            + pair_d[None, :] * stride_cd,
            mask=valid_rows[:, None] & rotated[None, :],
            other=1.0,
        ).to(tl.float32)
        sin_values = tl.load(
            sin_ptr
            + row_offsets[:, None] * stride_st
            + pair_d[None, :] * stride_sd,
            mask=valid_rows[:, None] & rotated[None, :],
            other=0.0,
        ).to(tl.float32)
        norm_math = norm.to(tl.float32)
        partner_math = partner_norm.to(tl.float32)

        first = norm_math * cos_values - partner_math * sin_values
        second = partner_math * sin_values + norm_math * cos_values
        route = tl.where(
            first_half[None, :],
            first,
            tl.where(rotated[None, :], second, norm),
        ).to(tl.bfloat16)
        route = tl.where(valid_rows[:, None], route, 0.0)

        block_rows = tl.minimum(block_size, rows - block * block_size)
        rc = tl.sum(route.to(tl.float32), axis=0) / block_rows
        vc = tl.sum(raw, axis=0)

        rc_offsets = block * stride_rcb + head * stride_rch + d_offsets * stride_rcd
        vc_offsets = block * stride_vcb + head * stride_vch + d_offsets * stride_vcd
        tl.store(rc_ptr + rc_offsets, rc.to(tl.bfloat16))
        tl.store(vc_ptr + vc_offsets, vc.to(tl.bfloat16))

else:
    _reduce_route_vc_kernel = None


def _validate_common(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> None:
    if not torch.is_tensor(v) or v.ndim != 3 or v.shape[-1] != HEAD_DIM:
        raise ValueError("Keyless route summary requires V [T,H,128]")
    if v.shape[0] <= 0 or v.shape[1] <= 0:
        raise ValueError("Keyless route summary requires nonempty rows and heads")
    if not torch.is_tensor(norm_weight) or norm_weight.shape != (HEAD_DIM,):
        raise ValueError("Keyless route summary requires route_norm.weight [128]")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("Keyless route summary epsilon must be finite and positive")
    expected_rope = (1, int(v.shape[0]), 1, ROPE_HALF_DIM, 2, 2)
    if not torch.is_tensor(rope_freqs) or tuple(rope_freqs.shape) != expected_rope:
        raise ValueError(
            f"Keyless route summary requires rope_freqs {expected_rope}, got "
            f"{tuple(getattr(rope_freqs, 'shape', ())) }"
        )
    if v.stride(-1) != 1:
        raise ValueError("Keyless route summary requires contiguous V head channels")
    if norm_weight.stride(0) != 1:
        raise ValueError("Keyless route summary requires contiguous route_norm.weight")


def materialized_route_reference(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Torch oracle matching Comfy H3 fused RMSNorm + split-half RoPE semantics."""
    _validate_common(v, norm_weight, eps, rope_freqs)
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("Keyless route reference tensors must share one device")

    work = v.float()
    inv = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + float(eps))
    norm = (work * inv * norm_weight.float()).to(dtype=v.dtype)
    rot = rope_freqs.to(device=v.device, dtype=torch.float32)
    c = rot[0, :, 0, :, 0, 0].unsqueeze(1)
    s = rot[0, :, 0, :, 1, 0].unsqueeze(1)
    a = norm[..., :ROPE_HALF_DIM].float()
    b = norm[..., ROPE_HALF_DIM:ROPE_ROT_DIM].float()
    rotated = torch.cat(
        (
            a * c - b * s,
            a * s + b * c,
        ),
        dim=-1,
    ).to(dtype=v.dtype)
    return torch.cat((rotated, norm[..., ROPE_ROT_DIM:]), dim=-1)


def route_summary_reference(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialized oracle for block routed-centroids and raw-V sums.

    This helper exists only for tests/calibration. Production native execution must
    call :func:`route_summary` and may not retain the full route tensor.
    """
    route = materialized_route_reference(v, norm_weight, eps, rope_freqs)
    rows, heads, dim = route.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_rows = blocks * BLOCK_SIZE
    if padded_rows != rows:
        route_padded = torch.zeros(
            padded_rows, heads, dim, device=route.device, dtype=route.dtype
        )
        value_padded = torch.zeros_like(route_padded)
        route_padded[:rows].copy_(route)
        value_padded[:rows].copy_(v)
    else:
        route_padded = route
        value_padded = v

    routed = route_padded.view(blocks, BLOCK_SIZE, heads, dim)
    values = value_padded.view(blocks, BLOCK_SIZE, heads, dim)
    counts = torch.full(
        (blocks, 1, 1),
        BLOCK_SIZE,
        device=v.device,
        dtype=torch.float32,
    )
    counts[-1] = rows - (blocks - 1) * BLOCK_SIZE
    rc = (routed.float().sum(dim=1) / counts).to(dtype=v.dtype)
    vc = values.float().sum(dim=1).to(dtype=v.dtype)
    return rc, vc


def route_summary(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bounded Keyless route/value block summaries directly from raw V.

    No `[T,H,128]` route output is allocated. Returned tensors are `[ceil(T/64),H,128]`.
    The kernel is experimental and intentionally not connected to Sol dispatch until
    real-SM120 arithmetic validation is recorded.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _reduce_route_vc_kernel is None:
        raise RuntimeError("Keyless route summary requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("native Keyless route summary requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("native Keyless route summary requires CUDA")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("Keyless route summary tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(f"native Keyless route summary currently requires eps={NORM_EPS}")

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    rc = torch.empty((blocks, heads, HEAD_DIM), device=v.device, dtype=torch.bfloat16)
    vc = torch.empty_like(rc)
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]

    _reduce_route_vc_kernel[(blocks, heads)](
        v,
        norm_weight,
        cos,
        sin,
        rc,
        vc,
        rows,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        rc.stride(0),
        rc.stride(1),
        rc.stride(2),
        vc.stride(0),
        vc.stride(1),
        vc.stride(2),
        eps=float(eps),
        block_size=BLOCK_SIZE,
        head_dim=HEAD_DIM,
        rope_half=ROPE_HALF_DIM,
        rope_rot=ROPE_ROT_DIM,
        num_warps=8,
        num_stages=2,
    )
    return rc, vc


__all__ = [
    "BLOCK_SIZE",
    "CONTRACT",
    "HEAD_DIM",
    "NORM_EPS",
    "ROPE_HALF_DIM",
    "ROPE_ROT_DIM",
    "materialized_route_reference",
    "route_summary",
    "route_summary_reference",
]
