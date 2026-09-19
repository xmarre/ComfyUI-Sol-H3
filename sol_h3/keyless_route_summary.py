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
SOL_REDUCTION_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
SOL_REDUCTION_CHUNK_BLOCKS = 8
FUSED_V5_CONTROL_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v5-rejected-control-v1"
BOUNDED_SPILL_CONTRACT = "sol-h3-keyless-k1-bounded-spill-v1"


if triton is not None:

    @triton.jit
    def _reduce_route_rc_sol_kernel(
        v_desc,
        v_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        rc_ptr,
        rows,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ct: tl.constexpr,
        stride_cd: tl.constexpr,
        stride_st: tl.constexpr,
        stride_sd: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
        TILE_D: tl.constexpr,
        rope_half: tl.constexpr,
        rope_rot: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Route-centroid candidate aligned to the released Sol reduction layout.

        The raw V tile is loaded with the same TensorDescriptor geometry and
        program-id ordering as Sana's released _reduce_kv_kernel.  Routing is
        derived per physical row before the final reduction, so no global route
        tensor is written.
        """
        d_tile, block, batch_head = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
        batch, head = batch_head // H, batch_head % H
        row_offsets = block * BLOCK + tl.arange(0, BLOCK)
        d_offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
        valid_rows = row_offsets < rows
        valid_d = d_offsets < D
        block_len = tl.minimum(BLOCK, rows - block * BLOCK)

        raw = v_desc.load(
            [batch, block * BLOCK, head, d_tile * TILE_D]
        ).reshape([BLOCK, TILE_D]).to(tl.float32)
        raw = tl.where(valid_rows[:, None] & valid_d[None, :], raw, 0.0)

        # Match comfy-kitchen's CUDA rms_sum ordering for D=128: each warp
        # lane accumulates d, d+32, d+64, d+96 with sequential FP32 FMAs,
        # then the 32 lane partials are reduced. The earlier tl.sum over all
        # 128 channels produced tiny row-wise norm differences that become
        # selector-significant after block-centroid cancellation.
        lane_offsets = tl.arange(0, 32)
        lane_base = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + lane_offsets[None, :] * stride_vd
        )
        lane0 = tl.load(
            v_ptr + lane_base,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane1 = tl.load(
            v_ptr + lane_base + 32 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane2 = tl.load(
            v_ptr + lane_base + 64 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane3 = tl.load(
            v_ptr + lane_base + 96 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane_sum = tl.zeros((BLOCK, 32), dtype=tl.float32)
        lane_sum = tl.fma(lane0, lane0, lane_sum)
        lane_sum = tl.fma(lane1, lane1, lane_sum)
        lane_sum = tl.fma(lane2, lane2, lane_sum)
        lane_sum = tl.fma(lane3, lane3, lane_sum)
        mean_square = tl.sum(lane_sum, axis=1) / D
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(
            weight_ptr + d_offsets,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        norm = (raw * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotating = d_offsets < rope_rot
        first_half = d_offsets < rope_half
        partner_d = tl.where(
            first_half,
            d_offsets + rope_half,
            d_offsets - rope_half,
        )
        partner_offsets = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + partner_d[None, :] * stride_vd
        )
        partner_raw = tl.load(
            v_ptr + partner_offsets,
            mask=valid_rows[:, None] & rotating[None, :],
            other=0.0,
        ).to(tl.float32)
        partner_weight = tl.load(
            weight_ptr + partner_d,
            mask=rotating & (partner_d < D),
            other=0.0,
        ).to(tl.float32)
        partner_norm = (
            partner_raw * inv_rms[:, None] * partner_weight[None, :]
        ).to(tl.bfloat16)

        pair_d = tl.where(
            first_half,
            d_offsets,
            d_offsets - rope_half,
        )
        cos_values = tl.load(
            cos_ptr
            + row_offsets[:, None] * stride_ct
            + pair_d[None, :] * stride_cd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=1.0,
        ).to(tl.float32)
        sin_values = tl.load(
            sin_ptr
            + row_offsets[:, None] * stride_st
            + pair_d[None, :] * stride_sd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=0.0,
        ).to(tl.float32)

        norm_math = norm.to(tl.float32)
        partner_math = partner_norm.to(tl.float32)
        first = norm_math * cos_values - partner_math * sin_values
        second = partner_math * sin_values + norm_math * cos_values
        route = tl.where(
            first_half[None, :],
            first,
            tl.where(rotating[None, :], second, norm_math),
        ).to(tl.bfloat16)
        # Keep the masked route in BF16 before reduction. This matches the
        # released Sol source expression and dtype, but real-SM120 evidence
        # shows that source/dtype parity alone does not reproduce the released
        # reducer: launch/distributed-layout effects remain under diagnosis.
        route = tl.where(
            valid_rows[:, None] & valid_d[None, :],
            route,
            0.0,
        ).to(tl.bfloat16)

        summary = tl.sum(route, axis=0) / block_len
        output_offsets = ((batch * N + block) * H + head) * D + d_offsets
        tl.store(
            rc_ptr + output_offsets,
            summary,
            mask=valid_d,
        )


    @triton.jit
    def _compute_route_inv_rms_kernel(
        v_ptr,
        inv_rms_ptr,
        rows,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_it: tl.constexpr,
        stride_ih: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Diagnostic split point after the exact current D=128 RMS reduction."""
        block = tl.program_id(1)
        batch_head = tl.program_id(2)
        head = batch_head % H
        row_offsets = block * BLOCK + tl.arange(0, BLOCK)
        valid_rows = row_offsets < rows
        lane_offsets = tl.arange(0, 32)
        lane_base = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + lane_offsets[None, :] * stride_vd
        )
        lane0 = tl.load(
            v_ptr + lane_base,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane1 = tl.load(
            v_ptr + lane_base + 32 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane2 = tl.load(
            v_ptr + lane_base + 64 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane3 = tl.load(
            v_ptr + lane_base + 96 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane_sum = tl.zeros((BLOCK, 32), dtype=tl.float32)
        lane_sum = tl.fma(lane0, lane0, lane_sum)
        lane_sum = tl.fma(lane1, lane1, lane_sum)
        lane_sum = tl.fma(lane2, lane2, lane_sum)
        lane_sum = tl.fma(lane3, lane3, lane_sum)
        mean_square = tl.sum(lane_sum, axis=1) / D
        inv_rms = tl.rsqrt(mean_square + eps)
        tl.store(
            inv_rms_ptr
            + row_offsets * stride_it
            + head * stride_ih,
            inv_rms,
            mask=valid_rows,
        )


    @triton.jit
    def _reduce_route_rc_from_inv_rms_kernel(
        v_desc,
        v_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        inv_rms_ptr,
        rc_ptr,
        rows,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ct: tl.constexpr,
        stride_cd: tl.constexpr,
        stride_st: tl.constexpr,
        stride_sd: tl.constexpr,
        stride_it: tl.constexpr,
        stride_ih: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
        TILE_D: tl.constexpr,
        rope_half: tl.constexpr,
        rope_rot: tl.constexpr,
    ):
        """Diagnostic RC with the RMS reduction removed from the route graph."""
        d_tile, block, batch_head = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
        batch, head = batch_head // H, batch_head % H
        row_offsets = block * BLOCK + tl.arange(0, BLOCK)
        d_offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
        valid_rows = row_offsets < rows
        valid_d = d_offsets < D
        block_len = tl.minimum(BLOCK, rows - block * BLOCK)

        raw = v_desc.load(
            [batch, block * BLOCK, head, d_tile * TILE_D]
        ).reshape([BLOCK, TILE_D]).to(tl.float32)
        raw = tl.where(valid_rows[:, None] & valid_d[None, :], raw, 0.0)
        inv_rms = tl.load(
            inv_rms_ptr
            + row_offsets * stride_it
            + head * stride_ih,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + d_offsets,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        norm = (raw * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotating = d_offsets < rope_rot
        first_half = d_offsets < rope_half
        partner_d = tl.where(
            first_half,
            d_offsets + rope_half,
            d_offsets - rope_half,
        )
        partner_offsets = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + partner_d[None, :] * stride_vd
        )
        partner_raw = tl.load(
            v_ptr + partner_offsets,
            mask=valid_rows[:, None] & rotating[None, :],
            other=0.0,
        ).to(tl.float32)
        partner_weight = tl.load(
            weight_ptr + partner_d,
            mask=rotating & (partner_d < D),
            other=0.0,
        ).to(tl.float32)
        partner_norm = (
            partner_raw * inv_rms[:, None] * partner_weight[None, :]
        ).to(tl.bfloat16)

        pair_d = tl.where(
            first_half,
            d_offsets,
            d_offsets - rope_half,
        )
        cos_values = tl.load(
            cos_ptr
            + row_offsets[:, None] * stride_ct
            + pair_d[None, :] * stride_cd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=1.0,
        ).to(tl.float32)
        sin_values = tl.load(
            sin_ptr
            + row_offsets[:, None] * stride_st
            + pair_d[None, :] * stride_sd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=0.0,
        ).to(tl.float32)

        norm_math = norm.to(tl.float32)
        partner_math = partner_norm.to(tl.float32)
        first = norm_math * cos_values - partner_math * sin_values
        second = partner_math * sin_values + norm_math * cos_values
        route = tl.where(
            first_half[None, :],
            first,
            tl.where(rotating[None, :], second, norm_math),
        ).to(tl.bfloat16)
        route = tl.where(
            valid_rows[:, None] & valid_d[None, :],
            route,
            0.0,
        ).to(tl.bfloat16)

        summary = tl.sum(route, axis=0) / block_len
        output_offsets = ((batch * N + block) * H + head) * D + d_offsets
        tl.store(
            rc_ptr + output_offsets,
            summary,
            mask=valid_d,
        )


    @triton.jit
    def _materialize_sol_reduction_v4_route_kernel(
        v_desc,
        v_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows,
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
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
        TILE_D: tl.constexpr,
        rope_half: tl.constexpr,
        rope_rot: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Diagnostic-only materialization of the exact fused-v4 row arithmetic."""
        d_tile, block, batch_head = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
        batch, head = batch_head // H, batch_head % H
        row_offsets = block * BLOCK + tl.arange(0, BLOCK)
        d_offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
        valid_rows = row_offsets < rows
        valid_d = d_offsets < D

        raw = v_desc.load(
            [batch, block * BLOCK, head, d_tile * TILE_D]
        ).reshape([BLOCK, TILE_D]).to(tl.float32)
        raw = tl.where(valid_rows[:, None] & valid_d[None, :], raw, 0.0)

        lane_offsets = tl.arange(0, 32)
        lane_base = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + lane_offsets[None, :] * stride_vd
        )
        lane0 = tl.load(
            v_ptr + lane_base,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane1 = tl.load(
            v_ptr + lane_base + 32 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane2 = tl.load(
            v_ptr + lane_base + 64 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane3 = tl.load(
            v_ptr + lane_base + 96 * stride_vd,
            mask=valid_rows[:, None],
            other=0.0,
        ).to(tl.float32)
        lane_sum = tl.zeros((BLOCK, 32), dtype=tl.float32)
        lane_sum = tl.fma(lane0, lane0, lane_sum)
        lane_sum = tl.fma(lane1, lane1, lane_sum)
        lane_sum = tl.fma(lane2, lane2, lane_sum)
        lane_sum = tl.fma(lane3, lane3, lane_sum)
        mean_square = tl.sum(lane_sum, axis=1) / D
        inv_rms = tl.rsqrt(mean_square + eps)

        weight = tl.load(
            weight_ptr + d_offsets,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        norm = (raw * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotating = d_offsets < rope_rot
        first_half = d_offsets < rope_half
        partner_d = tl.where(
            first_half,
            d_offsets + rope_half,
            d_offsets - rope_half,
        )
        partner_offsets = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + partner_d[None, :] * stride_vd
        )
        partner_raw = tl.load(
            v_ptr + partner_offsets,
            mask=valid_rows[:, None] & rotating[None, :],
            other=0.0,
        ).to(tl.float32)
        partner_weight = tl.load(
            weight_ptr + partner_d,
            mask=rotating & (partner_d < D),
            other=0.0,
        ).to(tl.float32)
        partner_norm = (
            partner_raw * inv_rms[:, None] * partner_weight[None, :]
        ).to(tl.bfloat16)

        pair_d = tl.where(
            first_half,
            d_offsets,
            d_offsets - rope_half,
        )
        cos_values = tl.load(
            cos_ptr
            + row_offsets[:, None] * stride_ct
            + pair_d[None, :] * stride_cd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=1.0,
        ).to(tl.float32)
        sin_values = tl.load(
            sin_ptr
            + row_offsets[:, None] * stride_st
            + pair_d[None, :] * stride_sd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=0.0,
        ).to(tl.float32)

        norm_math = norm.to(tl.float32)
        partner_math = partner_norm.to(tl.float32)
        first = norm_math * cos_values - partner_math * sin_values
        second = partner_math * sin_values + norm_math * cos_values
        route = tl.where(
            first_half[None, :],
            first,
            tl.where(rotating[None, :], second, norm_math),
        ).to(tl.bfloat16)
        route = tl.where(valid_rows[:, None] & valid_d[None, :], route, 0.0)

        out_offsets = (
            row_offsets[:, None] * stride_ot
            + head * stride_oh
            + d_offsets[None, :] * stride_od
        )
        tl.store(
            out_ptr + out_offsets,
            route,
            mask=valid_rows[:, None] & valid_d[None, :],
        )

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
        # Match comfy-kitchen CUDA fused RMS+RoPE semantics. Normalized values
        # are materialized to BF16, then the 2x2 rotation executes in FP32.
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


    @triton.jit
    def _materialize_native_route_kernel(
        v_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows,
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
        eps: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        rope_half: tl.constexpr,
        rope_rot: tl.constexpr,
        round_rope_products: tl.constexpr,
        rope_compute_fp32: tl.constexpr,
    ):
        """Diagnostic-only materialization of selectable RoPE rounding semantics."""
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

        mean_square = tl.sum(raw * raw, axis=1) / head_dim
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(weight_ptr + d_offsets).to(tl.float32)
        norm = (raw * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotating = d_offsets < rope_rot
        first_half = d_offsets < rope_half
        partner_d = tl.where(
            first_half,
            d_offsets + rope_half,
            d_offsets - rope_half,
        )
        partner_offsets = (
            row_offsets[:, None] * stride_vt
            + head * stride_vh
            + partner_d[None, :] * stride_vd
        )
        partner_raw = tl.load(
            v_ptr + partner_offsets,
            mask=valid_rows[:, None] & rotating[None, :],
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
            d_offsets,
            d_offsets - rope_half,
        )
        cos_loaded = tl.load(
            cos_ptr
            + row_offsets[:, None] * stride_ct
            + pair_d[None, :] * stride_cd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=1.0,
        )
        sin_loaded = tl.load(
            sin_ptr
            + row_offsets[:, None] * stride_st
            + pair_d[None, :] * stride_sd,
            mask=valid_rows[:, None] & rotating[None, :],
            other=0.0,
        )
        if rope_compute_fp32:
            # comfy-kitchen CUDA fused RMS+RoPE uses ComputeType=float when
            # HasRms=true: normalized BF16 values and frequency values are
            # converted to FP32 before the 2x2 rotation, then output is cast
            # back to BF16.
            norm_math = norm.to(tl.float32)
            partner_math = partner_norm.to(tl.float32)
            cos_values = cos_loaded.to(tl.float32)
            sin_values = sin_loaded.to(tl.float32)
        else:
            norm_math = norm
            partner_math = partner_norm
            cos_values = cos_loaded.to(tl.bfloat16)
            sin_values = sin_loaded.to(tl.bfloat16)

        norm_cos = norm_math * cos_values
        partner_sin = partner_math * sin_values
        norm_sin = norm_math * sin_values
        partner_cos = partner_math * cos_values
        if round_rope_products:
            # PyTorch's public Keyless expression materializes each BF16 multiply
            # before the following add/subtract because they are distinct tensor ops.
            norm_cos = norm_cos.to(tl.bfloat16)
            partner_sin = partner_sin.to(tl.bfloat16)
            norm_sin = norm_sin.to(tl.bfloat16)
            partner_cos = partner_cos.to(tl.bfloat16)
        first = norm_cos - partner_sin
        second = partner_sin + norm_cos
        route = tl.where(
            first_half[None, :],
            first,
            tl.where(rotating[None, :], second, norm),
        ).to(tl.bfloat16)

        out_offsets = (
            row_offsets[:, None] * stride_ot
            + head * stride_oh
            + d_offsets[None, :] * stride_od
        )
        tl.store(
            out_ptr + out_offsets,
            route,
            mask=valid_rows[:, None],
        )

else:
    _reduce_route_rc_sol_kernel = None
    _compute_route_inv_rms_kernel = None
    _reduce_route_rc_from_inv_rms_kernel = None
    _materialize_sol_reduction_v4_route_kernel = None
    _reduce_route_vc_kernel = None
    _materialize_native_route_kernel = None


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


def materialized_native_route_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Materialize route(V) with the exact experimental K1/K2 Triton arithmetic.

    This exists only to decompose real-H3 replay failures. Production native Keyless
    execution must never call it because it intentionally allocates [T,H,128].
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _materialize_native_route_kernel is None:
        raise RuntimeError("native-route diagnostic requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("native-route diagnostic requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("native-route diagnostic requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("native-route diagnostic currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("native-route diagnostic tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(f"native-route diagnostic currently requires eps={NORM_EPS}")

    rows, heads, _ = v.shape
    out = torch.empty_like(v)
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    _materialize_native_route_kernel[(blocks, heads)](
        v,
        norm_weight,
        cos,
        sin,
        out,
        rows,
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
        eps=float(eps),
        block_size=BLOCK_SIZE,
        head_dim=HEAD_DIM,
        rope_half=ROPE_HALF_DIM,
        rope_rot=ROPE_ROT_DIM,
        round_rope_products=False,
        rope_compute_fp32=True,
        num_warps=8,
        num_stages=2,
    )
    return out


def materialized_public_rounding_route_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Materialize the route with explicit BF16 RoPE-product rounding.

    This diagnostic tests the hypothesis that public Keyless differs from the current
    fused Triton route only because PyTorch materializes each BF16 multiply before
    the subsequent add/subtract. It is never a production path.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _materialize_native_route_kernel is None:
        raise RuntimeError("public-rounding route diagnostic requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("public-rounding route diagnostic requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("public-rounding route diagnostic requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("public-rounding route diagnostic currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("public-rounding route diagnostic tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(f"public-rounding route diagnostic currently requires eps={NORM_EPS}")

    rows, heads, _ = v.shape
    out = torch.empty_like(v)
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    _materialize_native_route_kernel[(blocks, heads)](
        v,
        norm_weight,
        cos,
        sin,
        out,
        rows,
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
        eps=float(eps),
        block_size=BLOCK_SIZE,
        head_dim=HEAD_DIM,
        rope_half=ROPE_HALF_DIM,
        rope_rot=ROPE_ROT_DIM,
        round_rope_products=True,
        rope_compute_fp32=False,
        num_warps=8,
        num_stages=2,
    )
    return out


def materialized_fp32_rope_route_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Materialize route(V) with native RMSNorm and comfy-kitchen-style FP32 RoPE math.

    Diagnostic only. The route normalization remains the experimental Triton
    reduction, but normalized BF16 values and frequency values are promoted to
    FP32 for the 2x2 rotation exactly as comfy-kitchen CUDA does for fused
    RMS+RoPE before casting the route back to BF16.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _materialize_native_route_kernel is None:
        raise RuntimeError("FP32-RoPE route diagnostic requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("FP32-RoPE route diagnostic requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("FP32-RoPE route diagnostic requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("FP32-RoPE route diagnostic currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("FP32-RoPE route diagnostic tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(f"FP32-RoPE route diagnostic currently requires eps={NORM_EPS}")

    rows, heads, _ = v.shape
    out = torch.empty_like(v)
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    _materialize_native_route_kernel[(blocks, heads)](
        v,
        norm_weight,
        cos,
        sin,
        out,
        rows,
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
        eps=float(eps),
        block_size=BLOCK_SIZE,
        head_dim=HEAD_DIM,
        rope_half=ROPE_HALF_DIM,
        rope_rot=ROPE_ROT_DIM,
        round_rope_products=False,
        rope_compute_fp32=True,
        num_warps=8,
        num_stages=2,
    )
    return out



def materialized_sol_reduction_v4_route_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Materialize the exact current K1-v4 row route for diagnostic isolation.

    This deliberately allocates a full route tensor and is therefore restricted
    to the oracle/test path. Production K1-v4 still emits only bounded RC/VC
    summaries. The purpose is to run the exact existing Sol reduction over the
    same row arithmetic and distinguish row-route error from fused-reduction
    arithmetic/layout error.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _materialize_sol_reduction_v4_route_kernel is None:
        raise RuntimeError("K1-v4 route isolation requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("K1-v4 route isolation requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("K1-v4 route isolation requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("K1-v4 route isolation currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("K1-v4 route isolation tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(f"K1-v4 route isolation currently requires eps={NORM_EPS}")

    from triton.tools.tensor_descriptor import TensorDescriptor

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_d = min(128, triton.next_power_of_2(HEAD_DIM))
    vb = v.unsqueeze(0)
    v_desc = TensorDescriptor.from_tensor(
        vb,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    out = torch.empty_like(v)
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]

    grid = (triton.cdiv(HEAD_DIM, tile_d), blocks, heads)
    _materialize_sol_reduction_v4_route_kernel[grid](
        v_desc,
        v,
        norm_weight,
        cos,
        sin,
        out,
        rows,
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
        heads,
        HEAD_DIM,
        BLOCK_SIZE,
        tile_d,
        ROPE_HALF_DIM,
        ROPE_ROT_DIM,
        float(eps),
        num_warps=8,
        num_stages=2,
    )
    return out




def route_inv_rms(
    v: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute the exact D=128 row/head inverse RMS scalar used by Keyless routing.

    This is the production K4 scalar staging primitive. It preserves the
    comfy-kitchen D=128 lane/FMA reduction order already established by K1,
    but stores only one FP32 scalar per physical V row/head. The fused SM120
    executor uses those scalars to derive selected routed K tiles from raw V
    inside CTA shared memory; no [T,H,128] route tensor is materialized.
    """
    if not torch.is_tensor(v) or v.ndim != 3 or v.shape[-1] != HEAD_DIM:
        raise ValueError("Keyless route inverse RMS requires V [T,H,128]")
    if v.shape[0] <= 0 or v.shape[1] <= 0:
        raise ValueError("Keyless route inverse RMS requires nonempty rows and heads")
    if v.stride(-1) != 1:
        raise ValueError("Keyless route inverse RMS requires contiguous V head channels")
    if not math.isfinite(float(eps)) or float(eps) != NORM_EPS:
        raise ValueError(f"Keyless route inverse RMS requires eps={NORM_EPS}")
    if _compute_route_inv_rms_kernel is None:
        raise RuntimeError("Keyless route inverse RMS requires Triton")
    if v.dtype != torch.bfloat16:
        raise TypeError("Keyless route inverse RMS requires BF16 V")
    if v.device.type != "cuda":
        raise RuntimeError("Keyless route inverse RMS requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("Keyless route inverse RMS currently targets SM120 only")

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    inv_rms = torch.empty((rows, heads), device=v.device, dtype=torch.float32)
    _compute_route_inv_rms_kernel[(1, blocks, heads)](
        v,
        inv_rms,
        rows,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        inv_rms.stride(0),
        inv_rms.stride(1),
        heads,
        HEAD_DIM,
        BLOCK_SIZE,
        float(eps),
        num_warps=8,
        num_stages=2,
    )
    return inv_rms

def route_centroid_split_rms_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Break the fused RMS-to-route layout lineage with a small FP32 scratch.

    Diagnostic only. The first kernel preserves the fused candidate's
    three-dimensional program geometry and exact D=128 lane/FMA RMS loads while
    writing only the per-row/head inverse RMS into [T,H] FP32 storage. The
    second kernel reloads that scalar,
    derives the unchanged BF16/FP32 H3 route, and performs the same BF16 route
    reduction. No global [T,H,128] route tensor is materialized.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _compute_route_inv_rms_kernel is None or _reduce_route_rc_from_inv_rms_kernel is None:
        raise RuntimeError("split-RMS route-centroid diagnostic requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("split-RMS route-centroid diagnostic requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("split-RMS route-centroid diagnostic requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("split-RMS route-centroid diagnostic currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("split-RMS route-centroid diagnostic tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(
            f"split-RMS route-centroid diagnostic currently requires eps={NORM_EPS}"
        )

    from triton.tools.tensor_descriptor import TensorDescriptor

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_d = min(128, triton.next_power_of_2(HEAD_DIM))
    inv_rms = torch.empty((rows, heads), device=v.device, dtype=torch.float32)
    rc = torch.empty(
        (1, blocks, heads, HEAD_DIM),
        device=v.device,
        dtype=torch.bfloat16,
    )
    vb = v.unsqueeze(0)
    v_desc = TensorDescriptor.from_tensor(
        vb,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    grid = (triton.cdiv(HEAD_DIM, tile_d), blocks, heads)
    _compute_route_inv_rms_kernel[grid](
        v,
        inv_rms,
        rows,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        inv_rms.stride(0),
        inv_rms.stride(1),
        heads,
        HEAD_DIM,
        BLOCK_SIZE,
        float(eps),
        num_warps=8,
        num_stages=2,
    )

    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]
    _reduce_route_rc_from_inv_rms_kernel[grid](
        v_desc,
        v,
        norm_weight,
        cos,
        sin,
        inv_rms,
        rc,
        rows,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        inv_rms.stride(0),
        inv_rms.stride(1),
        heads,
        blocks,
        HEAD_DIM,
        BLOCK_SIZE,
        tile_d,
        ROPE_HALF_DIM,
        ROPE_ROT_DIM,
        num_warps=8,
        num_stages=2,
    )
    return rc[0], int(inv_rms.numel() * inv_rms.element_size())




def route_centroid_bounded_spill_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Spill one BF16 route block at a time, then run exact released Sol reduction.

    Diagnostic only. Route construction is unchanged, but each physical
    64-token route block is written to bounded BF16 storage and then reloaded by
    the exact pinned _reduce_kv_kernel. The scratch never scales with T and no
    global [T,H,128] route tensor is allocated by this helper.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _materialize_sol_reduction_v4_route_kernel is None:
        raise RuntimeError("bounded-spill route-centroid diagnostic requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("bounded-spill route-centroid diagnostic requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("bounded-spill route-centroid diagnostic requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("bounded-spill route-centroid diagnostic currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("bounded-spill route-centroid diagnostic tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(
            f"bounded-spill route-centroid diagnostic currently requires eps={NORM_EPS}"
        )

    from triton.tools.tensor_descriptor import TensorDescriptor

    from ._vendor.sol_attn.preprocess import _reduce_kv_kernel

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_d = min(128, triton.next_power_of_2(HEAD_DIM))
    rc = torch.empty(
        (blocks, heads, HEAD_DIM),
        device=v.device,
        dtype=torch.bfloat16,
    )
    tile_kc = torch.empty(
        (1, 1, heads, HEAD_DIM),
        device=v.device,
        dtype=torch.bfloat16,
    )
    tile_vc = torch.empty_like(tile_kc)
    max_route_tile_bytes = 0
    max_route_tile_rows = 0

    for block in range(blocks):
        start = block * BLOCK_SIZE
        end = min(rows, start + BLOCK_SIZE)
        tile_rows = end - start
        route_tile = materialized_sol_reduction_v4_route_diagnostic(
            v[start:end],
            norm_weight,
            eps,
            rope_freqs[:, start:end],
        )
        max_route_tile_rows = max(max_route_tile_rows, int(tile_rows))
        max_route_tile_bytes = max(
            max_route_tile_bytes,
            int(route_tile.numel() * route_tile.element_size()),
        )

        route_tile_b = route_tile.unsqueeze(0)
        route_desc = TensorDescriptor.from_tensor(
            route_tile_b,
            [1, BLOCK_SIZE, 1, tile_d],
        )
        grid = (triton.cdiv(HEAD_DIM, tile_d), 1, heads)
        _reduce_kv_kernel[grid](
            route_desc,
            route_desc,
            tile_kc,
            tile_vc,
            tile_rows,
            heads,
            1,
            HEAD_DIM,
            BLOCK_SIZE,
            tile_d,
        )
        rc[block].copy_(tile_kc[0, 0])
        del route_desc, route_tile_b, route_tile

    summary_scratch_bytes = int(
        tile_kc.numel() * tile_kc.element_size()
        + tile_vc.numel() * tile_vc.element_size()
    )
    return rc, {
        "max_route_tile_rows": max_route_tile_rows,
        "max_route_tile_bytes": max_route_tile_bytes,
        "summary_scratch_bytes": summary_scratch_bytes,
        "max_working_scratch_bytes": max_route_tile_bytes + summary_scratch_bytes,
    }




def route_summary_bounded_spill_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    *,
    chunk_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Derive exact Sol RC/VC through bounded BF16 route spill/reload chunks.

    Diagnostic only. Each chunk materializes at most chunk_blocks physical V64
    route blocks, then reloads that bounded BF16 route through the exact pinned
    released _reduce_kv_kernel while the same launch consumes raw V for VC.
    Returned RC/VC are final K1 outputs; the only route scratch is the current
    bounded chunk and therefore does not scale with total sequence length.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    chunk_blocks = int(chunk_blocks)
    if chunk_blocks < 1 or chunk_blocks > 16:
        raise ValueError("bounded-spill chunk_blocks must be in [1,16]")
    if _materialize_sol_reduction_v4_route_kernel is None:
        raise RuntimeError("bounded-spill route-summary diagnostic requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("bounded-spill route-summary diagnostic requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("bounded-spill route-summary diagnostic requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("bounded-spill route-summary diagnostic currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("bounded-spill route-summary diagnostic tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(
            f"bounded-spill route-summary diagnostic currently requires eps={NORM_EPS}"
        )

    from triton.tools.tensor_descriptor import TensorDescriptor

    from ._vendor.sol_attn.preprocess import _reduce_kv_kernel

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_d = min(128, triton.next_power_of_2(HEAD_DIM))
    rc = torch.empty(
        (blocks, heads, HEAD_DIM),
        device=v.device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_like(rc)
    max_route_chunk_rows = 0
    max_route_chunk_blocks = 0
    max_route_chunk_bytes = 0
    reducer_configs_by_n: dict[str, dict[str, object]] = {}

    for first_block in range(0, blocks, chunk_blocks):
        first_row = first_block * BLOCK_SIZE
        end_row = min(rows, (first_block + chunk_blocks) * BLOCK_SIZE)
        chunk_rows = end_row - first_row
        chunk_n = (chunk_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
        route_chunk = materialized_sol_reduction_v4_route_diagnostic(
            v[first_row:end_row],
            norm_weight,
            eps,
            rope_freqs[:, first_row:end_row],
        )
        route_chunk_b = route_chunk.unsqueeze(0)
        raw_chunk_b = v[first_row:end_row].unsqueeze(0)
        route_desc = TensorDescriptor.from_tensor(
            route_chunk_b,
            [1, BLOCK_SIZE, 1, tile_d],
        )
        raw_desc = TensorDescriptor.from_tensor(
            raw_chunk_b,
            [1, BLOCK_SIZE, 1, tile_d],
        )
        rc_chunk = rc[first_block:first_block + chunk_n]
        vc_chunk = vc[first_block:first_block + chunk_n]
        grid = (triton.cdiv(HEAD_DIM, tile_d), chunk_n, heads)
        _reduce_kv_kernel[grid](
            route_desc,
            raw_desc,
            rc_chunk,
            vc_chunk,
            chunk_rows,
            heads,
            chunk_n,
            HEAD_DIM,
            BLOCK_SIZE,
            tile_d,
        )

        config = getattr(_reduce_kv_kernel, "best_config", None)
        if config is None:
            raise RuntimeError("bounded-spill reducer did not expose an autotune best_config")
        reducer_configs_by_n[str(chunk_n)] = {
            "num_warps": int(config.num_warps),
            "num_stages": int(config.num_stages),
            "num_ctas": int(config.num_ctas),
            "kwargs": dict(config.kwargs),
        }
        max_route_chunk_rows = max(max_route_chunk_rows, int(chunk_rows))
        max_route_chunk_blocks = max(max_route_chunk_blocks, int(chunk_n))
        max_route_chunk_bytes = max(
            max_route_chunk_bytes,
            int(route_chunk.numel() * route_chunk.element_size()),
        )
        del route_desc, raw_desc, route_chunk_b, raw_chunk_b, route_chunk

    return rc, vc, {
        "contract": BOUNDED_SPILL_CONTRACT,
        "chunk_blocks": chunk_blocks,
        "max_route_chunk_rows": max_route_chunk_rows,
        "max_route_chunk_blocks": max_route_chunk_blocks,
        "max_route_chunk_bytes": max_route_chunk_bytes,
        "max_working_scratch_bytes": max_route_chunk_bytes,
        "reducer_configs_by_n": reducer_configs_by_n,
    }



def route_centroid_sol_reduction_config_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    *,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Run K1-v5 RC under one released-Sol launch configuration.

    Diagnostic only. Released Sol autotunes its BF16 reduction over 4/8 warps
    and 1..4 stages. Triton reduction lowering depends on the distributed
    layout, including the number of warps, so source/dtype parity alone does
    not prove arithmetic parity. This helper keeps the exact K1-v5 row
    arithmetic and varies only the launch configuration.

    It emits only bounded RC summaries and never materializes a global route.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _reduce_route_rc_sol_kernel is None:
        raise RuntimeError("Sol-reduction config diagnostic requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError("Sol-reduction config diagnostic requires BF16 V and norm weight")
    if v.device.type != "cuda":
        raise RuntimeError("Sol-reduction config diagnostic requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError("Sol-reduction config diagnostic currently targets SM120 only")
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError("Sol-reduction config diagnostic tensors must share one CUDA device")
    if float(eps) != NORM_EPS:
        raise ValueError(
            f"Sol-reduction config diagnostic currently requires eps={NORM_EPS}"
        )
    if int(num_warps) not in (4, 8):
        raise ValueError("Sol-reduction config diagnostic requires num_warps in {4,8}")
    if int(num_stages) not in (1, 2, 3, 4):
        raise ValueError("Sol-reduction config diagnostic requires num_stages in {1,2,3,4}")

    from triton.tools.tensor_descriptor import TensorDescriptor

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_d = min(128, triton.next_power_of_2(HEAD_DIM))
    vb = v.unsqueeze(0)
    rc = torch.empty(
        (1, blocks, heads, HEAD_DIM),
        device=v.device,
        dtype=torch.bfloat16,
    )
    v_desc = TensorDescriptor.from_tensor(
        vb,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]
    grid = (triton.cdiv(HEAD_DIM, tile_d), blocks, heads)
    _reduce_route_rc_sol_kernel[grid](
        v_desc,
        v,
        norm_weight,
        cos,
        sin,
        rc,
        rows,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        heads,
        blocks,
        HEAD_DIM,
        BLOCK_SIZE,
        tile_d,
        ROPE_HALF_DIM,
        ROPE_ROT_DIM,
        float(eps),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return rc[0]


def route_summary_sol_reduction_v5_diagnostic(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejected K1 v5 fused-reduction control retained for evidence comparison.

    RC keeps the v4 row-local route arithmetic, including the D=128
    comfy-kitchen CUDA RMS lane/FMA ordering, and reduces the BF16 route with the
    same tl.sum source expression as released Sana Sol. It still launches at a
    fixed 8-warps/2-stages configuration; real-SM120 evidence shows that this is
    not yet arithmetic-identical to the autotuned released reducer. VC is
    produced by the exact pinned Sana _reduce_kv_kernel used by the materialized
    QKV preprocessing path; raw V is supplied as its unused K-side diagnostic
    input and only the VC result is retained.

    This function is diagnostic only and is never eligible for provider dispatch.
    It never writes a full [T,H,128] route tensor.
    """
    _validate_common(v, norm_weight, eps, rope_freqs)
    if _reduce_route_rc_sol_kernel is None:
        raise RuntimeError("Sol-reduction Keyless route summary requires Triton")
    if v.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        raise TypeError(
            "Sol-reduction Keyless route summary requires BF16 V and norm weight"
        )
    if v.device.type != "cuda":
        raise RuntimeError("Sol-reduction Keyless route summary requires CUDA")
    if torch.cuda.get_device_capability(v.device) != (12, 0):
        raise RuntimeError(
            "Sol-reduction Keyless route summary currently targets SM120 only"
        )
    if norm_weight.device != v.device or rope_freqs.device != v.device:
        raise ValueError(
            "Sol-reduction Keyless route summary tensors must share one CUDA device"
        )
    if float(eps) != NORM_EPS:
        raise ValueError(
            f"Sol-reduction Keyless route summary currently requires eps={NORM_EPS}"
        )

    from triton.tools.tensor_descriptor import TensorDescriptor

    from ._vendor.sol_attn.preprocess import _reduce_kv_kernel

    rows, heads, _ = v.shape
    blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_d = min(128, triton.next_power_of_2(HEAD_DIM))
    vb = v.unsqueeze(0)
    rc5 = torch.empty(
        (1, blocks, heads, HEAD_DIM),
        device=v.device,
        dtype=torch.bfloat16,
    )
    vc5 = torch.empty_like(rc5)
    scratch_kc5 = torch.empty_like(rc5)
    v_desc = TensorDescriptor.from_tensor(
        vb,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    cos = rope_freqs[0, :, 0, :, 0, 0]
    sin = rope_freqs[0, :, 0, :, 1, 0]

    grid = (triton.cdiv(HEAD_DIM, tile_d), blocks, heads)
    _reduce_route_rc_sol_kernel[grid](
        v_desc,
        v,
        norm_weight,
        cos,
        sin,
        rc5,
        rows,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        heads,
        blocks,
        HEAD_DIM,
        BLOCK_SIZE,
        tile_d,
        ROPE_HALF_DIM,
        ROPE_ROT_DIM,
        float(eps),
        num_warps=8,
        num_stages=2,
    )
    _reduce_kv_kernel[grid](
        v_desc,
        v_desc,
        scratch_kc5,
        vc5,
        rows,
        heads,
        blocks,
        HEAD_DIM,
        BLOCK_SIZE,
        tile_d,
    )
    return rc5[0], vc5[0]


def route_summary_sol_reduction(
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K1 v6 fixed-b8 bounded BF16 spill/reload candidate.

    Real-SM120 v8 evidence established that chunk sizes 1/2/4/8/16 all produce
    RC bit-identical to materialized route + released Sol reduction, exact VC,
    and zero selector-bit divergence across the frozen 12-case calibration.
    Eight V64 blocks is the selected K1 candidate because it is the measured
    memory/performance knee: 7,340,032 bytes of maximum route scratch on H3
    geometry with stable tail timing across early/mid/late blocks.

    This candidate is still not wired into provider dispatch. It materializes
    only one bounded [<=512,H,128] BF16 route chunk at a time, never a global
    [T,H,128] route tensor.
    """
    rc, vc, _metadata = route_summary_bounded_spill_diagnostic(
        v,
        norm_weight,
        eps,
        rope_freqs,
        chunk_blocks=SOL_REDUCTION_CHUNK_BLOCKS,
    )
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
    "FUSED_V5_CONTROL_CONTRACT",
    "BOUNDED_SPILL_CONTRACT",
    "CONTRACT",
    "HEAD_DIM",
    "NORM_EPS",
    "ROPE_HALF_DIM",
    "ROPE_ROT_DIM",
    "SOL_REDUCTION_CONTRACT",
    "SOL_REDUCTION_CHUNK_BLOCKS",
    "materialized_route_reference",
    "materialized_native_route_diagnostic",
    "materialized_public_rounding_route_diagnostic",
    "materialized_fp32_rope_route_diagnostic",
    "materialized_sol_reduction_v4_route_diagnostic",
    "route_centroid_bounded_spill_diagnostic",
    "route_centroid_sol_reduction_config_diagnostic",
    "route_centroid_split_rms_diagnostic",
    "route_summary_bounded_spill_diagnostic",
    "route_summary",
    "route_summary_reference",
    "route_summary_sol_reduction",
    "route_inv_rms",
    "route_summary_sol_reduction_v5_diagnostic",
]
