"""Diagnostic K4 composition: exact K3 mask + bounded raw-V route tiles.

This is not provider promotion.  It composes the already-proven K1 summaries,
K3 exact-block route mask, and K1-identical row-route transform while keeping
route storage bounded to one selected V64 tile at a time.

The diagnostic deliberately receives a frozen route trace.  K3 already proved
that K1-v6 summaries reproduce the released Sol selector exactly; this layer
isolates the remaining execution boundary before any vendored CuTe source is
changed.
"""
from __future__ import annotations

import math

import torch

from .keyless_exact_attention import CANONICAL_SCALE
from .keyless_route_summary import (
    HEAD_DIM,
    NORM_EPS,
    materialized_sol_reduction_v4_route_diagnostic,
)


CONTRACT = "sol-h3-keyless-k4-selected-route-composition-v1"
BLOCK_SIZE = 64
ROUTE_GROUP_BLOCKS = 64
TRACE_WORDS = 2


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def exact_heads_from_route_trace(
    route_trace: torch.Tensor,
    *,
    q_tile: int,
    block_start: int,
    block_count: int,
) -> torch.Tensor:
    """Decode exact-block booleans as [H, block_count] without changing policy."""
    if (
        not torch.is_tensor(route_trace)
        or route_trace.ndim != 5
        or route_trace.shape[0] != 1
        or route_trace.shape[-1] != TRACE_WORDS
        or route_trace.dtype != torch.int32
    ):
        raise ValueError(
            "K4 route trace must be int32 [1,Qtiles,H,route_groups,2]"
        )
    if type(q_tile) is not int or not 0 <= q_tile < route_trace.shape[1]:
        raise ValueError("K4 q_tile is outside the route trace")
    if type(block_start) is not int or block_start < 0:
        raise ValueError("K4 block_start must be a nonnegative integer")
    if type(block_count) is not int or not 0 < block_count <= ROUTE_GROUP_BLOCKS:
        raise ValueError("K4 block_count must be in [1,64]")
    route_group = block_start // ROUTE_GROUP_BLOCKS
    if block_start % ROUTE_GROUP_BLOCKS:
        raise ValueError("K4 block_start must begin on a route-group boundary")
    if route_group >= route_trace.shape[3]:
        raise ValueError("K4 route group is outside the route trace")

    offsets = torch.arange(
        block_count,
        device=route_trace.device,
        dtype=torch.int64,
    )
    words = torch.div(offsets, 32, rounding_mode="floor")
    bits = offsets.remainder(32)
    packed = route_trace[0, q_tile, :, route_group, :]
    selected_words = packed[:, words]
    return ((selected_words >> bits.unsqueeze(0)) & 1).to(torch.bool)


def _validate(
    q: torch.Tensor,
    v: torch.Tensor,
    route_centroid: torch.Tensor,
    value_sum: torch.Tensor,
    route_trace: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    scale: float,
) -> None:
    if (
        not torch.is_tensor(q)
        or not torch.is_tensor(v)
        or q.ndim != 3
        or v.ndim != 3
        or q.shape[1:] != v.shape[1:]
        or q.shape[-1] != HEAD_DIM
    ):
        raise ValueError("K4 composition requires Q/V [T,H,128]")
    if q.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("K4 composition requires BF16 Q/V")
    if q.device != v.device or q.device.type != "cuda":
        raise ValueError("K4 composition requires Q/V on one CUDA device")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("K4 composition currently targets SM120 only")
    if float(eps) != NORM_EPS:
        raise ValueError(f"K4 composition requires eps={NORM_EPS}")
    if float(scale) != CANONICAL_SCALE:
        raise ValueError(f"K4 composition requires scale={CANONICAL_SCALE}")
    if (
        not torch.is_tensor(norm_weight)
        or norm_weight.shape != (HEAD_DIM,)
        or norm_weight.dtype != torch.bfloat16
        or norm_weight.device != q.device
    ):
        raise ValueError("K4 composition requires BF16 route_norm.weight [128]")
    expected_rope = (1, int(v.shape[0]), 1, 48, 2, 2)
    if (
        not torch.is_tensor(rope_freqs)
        or tuple(rope_freqs.shape) != expected_rope
        or rope_freqs.device != q.device
    ):
        raise ValueError(
            f"K4 composition requires rope_freqs {expected_rope} on the Q/V device"
        )

    blocks = _ceil_div(int(v.shape[0]), BLOCK_SIZE)
    expected_summary = (blocks, int(v.shape[1]), HEAD_DIM)
    if (
        tuple(route_centroid.shape) != expected_summary
        or tuple(value_sum.shape) != expected_summary
        or route_centroid.dtype != torch.bfloat16
        or value_sum.dtype != torch.bfloat16
        or route_centroid.device != q.device
        or value_sum.device != q.device
    ):
        raise ValueError(
            f"K4 summaries must be BF16 {expected_summary} on the Q/V device"
        )

    expected_trace = (
        1,
        _ceil_div(int(q.shape[0]), BLOCK_SIZE),
        int(q.shape[1]),
        _ceil_div(blocks, ROUTE_GROUP_BLOCKS),
        TRACE_WORDS,
    )
    if tuple(route_trace.shape) != expected_trace or route_trace.dtype != torch.int32:
        raise ValueError(
            f"K4 route trace must be int32 {expected_trace}, got "
            f"{tuple(route_trace.shape)} {route_trace.dtype}"
        )
    if route_trace.device != q.device:
        raise ValueError("K4 route trace must share the Q/V device")


def selected_route_composition(
    q: torch.Tensor,
    v: torch.Tensor,
    route_centroid: torch.Tensor,
    value_sum: torch.Tensor,
    route_trace: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    *,
    scale: float = CANONICAL_SCALE,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Execute the fixed K3 route mask without a global [Tv,H,128] route.

    Approximate blocks consume K1 RC/VC directly.  Exact blocks materialize only
    one raw-V block's K1 row-route view at a time, score against it, and retrieve
    the untouched raw V rows.
    """
    _validate(
        q,
        v,
        route_centroid,
        value_sum,
        route_trace,
        norm_weight,
        eps,
        rope_freqs,
        scale,
    )

    q_rows, heads, _ = q.shape
    v_rows = int(v.shape[0])
    blocks = _ceil_div(v_rows, BLOCK_SIZE)
    q_tiles = _ceil_div(int(q_rows), BLOCK_SIZE)
    output = torch.empty_like(q)
    max_route_tile_bytes = 0
    materialized_route_tile_calls = 0

    for q_tile in range(q_tiles):
        q_start = q_tile * BLOCK_SIZE
        q_stop = min(q_start + BLOCK_SIZE, int(q_rows))
        q_tile_bf16 = q[q_start:q_stop]
        q_float = q_tile_bf16.float()

        row_max = torch.full(
            (q_stop - q_start, heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        row_sum = torch.zeros_like(row_max)
        accumulator = torch.zeros(
            (q_stop - q_start, heads, HEAD_DIM),
            device=q.device,
            dtype=torch.float32,
        )

        for block_start in range(0, blocks, ROUTE_GROUP_BLOCKS):
            block_count = min(ROUTE_GROUP_BLOCKS, blocks - block_start)
            block_stop = block_start + block_count
            exact = exact_heads_from_route_trace(
                route_trace,
                q_tile=q_tile,
                block_start=block_start,
                block_count=block_count,
            )
            approximate = ~exact

            rc_group = route_centroid[block_start:block_stop].float()
            vc_group = value_sum[block_start:block_stop].float()
            approximate_logits = torch.einsum(
                "mhd,bhd->mhb",
                q_float,
                rc_group,
            ) * float(scale)
            masked_logits = approximate_logits.masked_fill(
                exact.unsqueeze(0),
                -float("inf"),
            )
            has_approximate = approximate.any(dim=1)
            approximate_max = masked_logits.amax(dim=-1)
            next_max = torch.where(
                has_approximate.unsqueeze(0),
                torch.maximum(row_max, approximate_max),
                row_max,
            )
            alpha = torch.where(
                has_approximate.unsqueeze(0),
                torch.exp(row_max - next_max),
                torch.ones_like(row_max),
            )
            probabilities = torch.where(
                approximate.unsqueeze(0),
                torch.exp(masked_logits - next_max.unsqueeze(-1)),
                torch.zeros_like(masked_logits),
            )
            lengths = torch.tensor(
                [
                    min(BLOCK_SIZE, v_rows - block * BLOCK_SIZE)
                    for block in range(block_start, block_stop)
                ],
                device=q.device,
                dtype=torch.float32,
            )
            row_sum = row_sum * alpha + torch.sum(
                probabilities * lengths.view(1, 1, -1),
                dim=-1,
            )
            accumulator = accumulator * alpha.unsqueeze(-1) + torch.einsum(
                "mhb,bhd->mhd",
                probabilities,
                vc_group,
            )
            row_max = next_max

            for local_block in range(block_count):
                exact_heads = exact[:, local_block]
                if not bool(exact_heads.any().item()):
                    continue
                block = block_start + local_block
                v_start = block * BLOCK_SIZE
                v_stop = min(v_start + BLOCK_SIZE, v_rows)
                raw_tile = v[v_start:v_stop]
                rope_tile = rope_freqs[:, v_start:v_stop]
                route_tile = materialized_sol_reduction_v4_route_diagnostic(
                    raw_tile,
                    norm_weight,
                    eps,
                    rope_tile,
                )
                materialized_route_tile_calls += 1
                max_route_tile_bytes = max(
                    max_route_tile_bytes,
                    int(route_tile.numel() * route_tile.element_size()),
                )

                exact_logits = torch.einsum(
                    "mhd,nhd->mhn",
                    q_float,
                    route_tile.float(),
                ) * float(scale)
                exact_max = exact_logits.amax(dim=-1)
                next_max = torch.where(
                    exact_heads.unsqueeze(0),
                    torch.maximum(row_max, exact_max),
                    row_max,
                )
                alpha = torch.where(
                    exact_heads.unsqueeze(0),
                    torch.exp(row_max - next_max),
                    torch.ones_like(row_max),
                )
                exact_probabilities = torch.where(
                    exact_heads.view(1, heads, 1),
                    torch.exp(exact_logits - next_max.unsqueeze(-1)),
                    torch.zeros_like(exact_logits),
                )
                row_sum = row_sum * alpha + exact_probabilities.sum(dim=-1)
                accumulator = (
                    accumulator * alpha.unsqueeze(-1)
                    + torch.einsum(
                        "mhn,nhd->mhd",
                        exact_probabilities,
                        raw_tile.float(),
                    )
                )
                row_max = next_max

        if not bool(torch.isfinite(row_sum).all().item()) or bool((row_sum <= 0).any().item()):
            raise RuntimeError("K4 composition produced an invalid softmax denominator")
        output[q_start:q_stop].copy_(
            (accumulator / row_sum.unsqueeze(-1)).to(torch.bfloat16)
        )

    full_route_bytes = int(v.numel() * v.element_size())
    return output, {
        "max_route_tile_bytes": max_route_tile_bytes,
        "full_materialized_route_bytes": full_route_bytes,
        "max_route_tile_fraction": (
            max_route_tile_bytes / full_route_bytes if full_route_bytes else 0.0
        ),
        "materialized_route_tile_calls": materialized_route_tile_calls,
    }


__all__ = [
    "BLOCK_SIZE",
    "CONTRACT",
    "ROUTE_GROUP_BLOCKS",
    "exact_heads_from_route_trace",
    "selected_route_composition",
]
