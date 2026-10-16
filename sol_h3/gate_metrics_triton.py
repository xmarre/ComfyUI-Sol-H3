"""Fused, bounded-memory Sol arithmetic-gate statistics on CUDA.

One Triton launch computes partial statistics over *every* output element.
Only the six partial vectors are reduced by PyTorch. This replaces thousands
of tiny PyTorch reduction launches without dropping Q rows, changing SOL's
attention output, skipping the full-domain gate, or retaining full FP32 QKV.
CPU and unsupported layouts continue to use the original reduction code.
"""
from __future__ import annotations

import logging

import torch

_log = logging.getLogger("comfy.sol_h3")
_COMPILE_FALLBACK_REPORTED = False

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _partial_gate_metrics(
        got, want, partial,
        N: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
        G0: tl.constexpr, G1: tl.constexpr, G2: tl.constexpr, G3: tl.constexpr,
        W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
        TILES: tl.constexpr, BLOCK: tl.constexpr,
    ):
        tile = tl.program_id(0)
        idx = tile * BLOCK + tl.arange(0, BLOCK)
        valid = idx < N
        a = idx // (S1 * S2 * S3)
        b = idx // (S2 * S3) % S1
        c = idx // S3 % S2
        d = idx % S3
        g = tl.load(got + a*G0 + b*G1 + c*G2 + d*G3, valid, 0).to(tl.float32)
        w = tl.load(want + a*W0 + b*W1 + c*W2 + d*W3, valid, 0).to(tl.float32)
        delta = g - w
        # BF16 has no representable value at the FP32 maximum; this detects
        # BF16 nonfinites and FP32 overflow in the difference without relying
        # on optional libdevice predicates.
        finite = (
            (tl.abs(g) < 3.4028234663852886e38)
            & (tl.abs(w) < 3.4028234663852886e38)
            & (tl.abs(delta) < 3.4028234663852886e38)
        )
        # Masked lanes represent zeros, so neither maxima nor norms change.
        all_finite = tl.min(tl.where(valid, finite.to(tl.int32), 1), 0)
        reference_max = tl.max(tl.abs(w), 0)
        error_max = tl.max(tl.abs(delta), 0)
        difference = delta.to(tl.float64)
        reference = w.to(tl.float64)
        abs_sum = tl.sum(tl.abs(difference), 0)
        diff_square = tl.sum(difference * difference, 0)
        ref_square = tl.sum(reference * reference, 0)
        tl.store(partial + 0*TILES + tile, all_finite.to(tl.float64))
        tl.store(partial + 1*TILES + tile, reference_max.to(tl.float64))
        tl.store(partial + 2*TILES + tile, error_max.to(tl.float64))
        tl.store(partial + 3*TILES + tile, abs_sum)
        tl.store(partial + 4*TILES + tile, diff_square)
        tl.store(partial + 5*TILES + tile, ref_square)


def gate_partials(got: torch.Tensor, want: torch.Tensor):
    """Return exhaustive CUDA gate partials [6, tiles], or None on fallback.

    This API permits large query sequences to accumulate several bounded
    all-selected gate chunks without any per-chunk host synchronization.
    """
    if (
        triton is None or got.device.type != "cuda"
        or got.shape != want.shape or got.dtype != want.dtype
        or got.dtype != torch.bfloat16
        or got.ndim not in (3, 4) or got.numel() <= 262144
        or got.stride(-1) != 1 or want.stride(-1) != 1
        or torch.is_grad_enabled()
    ):
        return None
    if got.ndim == 3:
        shape = (*got.shape, 1)
        gs = (*got.stride(), 0)
        ws = (*want.stride(), 0)
    else:
        shape = tuple(got.shape)
        gs = tuple(got.stride())
        ws = tuple(want.stride())
    n = got.numel()
    block = 2048
    tiles = triton.cdiv(n, block)
    global _COMPILE_FALLBACK_REPORTED
    try:
        partial = torch.empty((6, tiles), dtype=torch.float64, device=got.device)
        _partial_gate_metrics[(tiles,)](
            got, want, partial, n, *shape[1:], *gs, *ws, tiles, block,
            num_warps=4,
        )
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        # An unavailable Triton compiler/backend must never break a generation.
        # Fall back to the exhaustive PyTorch oracle, not an unchecked result.
        if not _COMPILE_FALLBACK_REPORTED:
            _log.warning(
                "Sol-H3 fused arithmetic-gate reducer unavailable; "
                "falling back to exhaustive PyTorch statistics: %s", exc,
            )
            _COMPILE_FALLBACK_REPORTED = True
        return None
    return partial


def gate_statistics(got: torch.Tensor, want: torch.Tensor):
    """Full-domain [finite, reference_peak, error_peak, mean_abs, rel_l2]."""
    partial = gate_partials(got, want)
    if partial is None:
        return None
    n = got.numel()
    # Preserve the original FP64 statistical aggregation and one host read.
    sums = torch.stack((
        partial[0].amin(),
        partial[1].amax(),
        partial[2].amax(),
        partial[3].sum(dtype=torch.float64) / n,
        partial[4].sum(dtype=torch.float64).sqrt()
        / partial[5].sum(dtype=torch.float64).sqrt().clamp_min(1e-12),
    ))
    return sums.tolist()
