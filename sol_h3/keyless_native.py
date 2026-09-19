"""Native SM120 Keyless Sol candidate orchestration.

This module owns the no-global-route K4 candidate boundary.  K1/K3 summaries
remain the already-proven raw-V RC/VC + threshold path.  The SM120 executor
receives raw V as both exact-K source and PV source, derives selected routed K
fragments inside the native kernel, and never allocates a complete [T,H,128]
route tensor.

The candidate is intentionally not production-promoted until its dedicated
real-SM120 calibration and subsequently frozen holdout are consumed.
"""
from __future__ import annotations

from functools import lru_cache
import math

import torch

from .keyless_route_summary import (
    NORM_EPS,
    ROPE_HALF_DIM,
    route_summary_sol_reduction,
)
from .keyless_selector import threshold_from_route_centroids


CONTRACT = "sol-h3-keyless-native-sm120-v1"
FUSED_RECEIPT_TAG = "sol_h3_keyless_fused_v1"
PROMOTION_READY = False
HEAD_DIM = 128


@lru_cache(maxsize=1)
def _verified_vendor_contract() -> str:
    from .provenance import verify_source

    manifest = verify_source()
    contract = str(manifest.get("contract"))
    from ._vendor.sol_attn import KEYLESS_FUSED_CONTRACT

    if contract != KEYLESS_FUSED_CONTRACT:
        raise RuntimeError(
            "packaged Sol source contract does not match the Keyless fused ABI: "
            f"{contract!r} != {KEYLESS_FUSED_CONTRACT!r}"
        )
    return contract


def _validate_inputs(
    q: torch.Tensor,
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    *,
    scale: float,
    tau: float,
) -> None:
    if (
        not torch.is_tensor(q)
        or not torch.is_tensor(v)
        or q.ndim != 3
        or v.ndim != 3
        or q.shape[1:] != v.shape[1:]
        or q.shape[-1] != HEAD_DIM
        or q.shape[0] <= 0
        or v.shape[0] <= 0
    ):
        raise ValueError(
            "native Keyless Sol requires Q [Tq,H,128] and raw V [Tv,H,128]"
        )
    if q.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("native Keyless Sol requires BF16 Q and V")
    if q.device != v.device or q.device.type != "cuda":
        raise ValueError("native Keyless Sol requires Q/V on one CUDA device")
    if tuple(torch.cuda.get_device_capability(q.device)) != (12, 0):
        raise RuntimeError("native Keyless Sol currently requires SM120")
    if q.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("native Keyless Sol requires contiguous head channels")
    if (
        not torch.is_tensor(norm_weight)
        or tuple(norm_weight.shape) != (HEAD_DIM,)
        or norm_weight.dtype != torch.bfloat16
        or norm_weight.device != q.device
        or norm_weight.stride(0) != 1
    ):
        raise ValueError("native Keyless Sol requires contiguous BF16 route_norm.weight [128]")
    expected_rope = (1, int(v.shape[0]), 1, ROPE_HALF_DIM, 2, 2)
    if (
        not torch.is_tensor(rope_freqs)
        or tuple(rope_freqs.shape) != expected_rope
        or rope_freqs.device != q.device
    ):
        raise ValueError(
            f"native Keyless Sol requires exact V-aligned rope_freqs {expected_rope}"
        )
    if float(eps) != NORM_EPS:
        raise ValueError(f"native Keyless Sol requires eps={NORM_EPS}")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("native Keyless Sol scale must be finite and positive")
    if not math.isclose(float(scale), HEAD_DIM ** -0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("native Keyless Sol requires canonical 128**-0.5 scale")
    if not math.isfinite(float(tau)) or not 0.0 <= float(tau) <= 3.0:
        raise ValueError("native Keyless Sol tau must be finite and in [0,3]")


def run_native_candidate(
    q: torch.Tensor,
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor,
    *,
    scale: float,
    tau: float,
    sink_start: int = 0,
    sink_tokens: int = 0,
    debug_route_trace: bool = False,
):
    """Execute the unfrozen K4 native candidate on raw V.

    This function is callable by the calibration probe while PROMOTION_READY is
    false.  It does not create or accept a materialized route tensor.
    """
    _validate_inputs(
        q,
        v,
        norm_weight,
        eps,
        rope_freqs,
        scale=scale,
        tau=tau,
    )
    if type(sink_start) is not int or type(sink_tokens) is not int:
        raise ValueError("native Keyless sink geometry must use integers")
    if sink_start < 0 or sink_tokens < 0 or sink_start + sink_tokens > int(v.shape[0]):
        raise ValueError("native Keyless sink range is outside V rows")
    if type(debug_route_trace) is not bool:
        raise TypeError("debug_route_trace must be bool")

    _verified_vendor_contract()

    rc, vc = route_summary_sol_reduction(v, norm_weight, eps, rope_freqs)
    qb = q.unsqueeze(0)
    vb = v.unsqueeze(0)
    rcb = rc.unsqueeze(0)
    vcb = vc.unsqueeze(0)
    threshold = threshold_from_route_centroids(
        qb,
        rcb,
        kv_rows=int(v.shape[0]),
        tau=float(tau),
        scale=float(scale),
    )
    route_cos = rope_freqs[:, :, 0, :, 0, 0]
    route_sin = rope_freqs[:, :, 0, :, 1, 0]

    from ._vendor.sol_attn import sol_attn_keyless

    result = sol_attn_keyless(
        qb,
        vb,
        rcb,
        vcb,
        threshold,
        norm_weight,
        route_cos,
        route_sin,
        scale=float(scale),
        sink_start=sink_start,
        sink_tokens=sink_tokens,
        debug_route_trace=debug_route_trace,
    )
    if debug_route_trace:
        output, trace = result
        return output[0], trace
    return result[0]


__all__ = [
    "CONTRACT",
    "FUSED_RECEIPT_TAG",
    "PROMOTION_READY",
    "run_native_candidate",
]
