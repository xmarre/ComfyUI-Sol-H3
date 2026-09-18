"""Real-H3 replay helpers for experimental Keyless SM120 K1/K2 calibration.

The replay CLI deliberately keeps this module free of ComfyUI and checkpoint I/O so
CPU tests can lock the predeclared arithmetic envelope independently of the runtime
fixture. The CLI uses exact Comfy H3 RMSNorm+RoPE output as the materialized oracle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ArithmeticLimit:
    rel_l2: float
    mean_abs: float
    max_abs: float


@dataclass(frozen=True)
class ScaleAwareArithmeticLimit:
    rel_l2: float
    max_abs_over_want_abs_max: float
    mean_abs_over_want_mean_abs: float
    worst_bf16_ulps: float


@dataclass(frozen=True)
class ReplayEnvelopeV2:
    """Frozen from the threshold-free v2 calibration before holdout replay.

    Aggregate limits are exactly 2x the observed 36-case calibration maxima.
    The local arithmetic invariant is intentionally stricter: every calibration
    case was at most one BF16 ULP at its worst element, so the holdout gate keeps
    that exact one-ULP bound rather than multiplying it.
    """

    contract: str = "sol-h3-keyless-real-h3-replay-envelope-v2"
    calibration_contract: str = "sol-h3-keyless-v2-calibration-v1"
    calibration_sha256: str = (
        "6cb5996e98613563947804d22c9a59cf11e8df58f1da4e566f3141709218531d"
    )
    calibration_case_count: int = 36
    aggregate_margin_multiplier: float = 2.0
    k1_route_centroid: ScaleAwareArithmeticLimit = ScaleAwareArithmeticLimit(
        rel_l2=0.0002579810388851911,
        max_abs_over_want_abs_max=0.007518796992481203,
        mean_abs_over_want_mean_abs=6.181015165732724e-06,
        worst_bf16_ulps=1.0,
    )
    k2_output: ScaleAwareArithmeticLimit = ScaleAwareArithmeticLimit(
        rel_l2=0.002869961317628622,
        max_abs_over_want_abs_max=0.012121212121212121,
        mean_abs_over_want_mean_abs=0.0009517048283547036,
        worst_bf16_ulps=1.0,
    )
    k1_value_max_abs: float = 0.0


V2_ENVELOPE = ReplayEnvelopeV2()


@dataclass(frozen=True)
class ReplayEnvelopeV1:
    """Frozen before any real-H3 replay result is observed."""

    contract: str = "sol-h3-keyless-real-h3-replay-envelope-v1"
    k1_route_centroid: ArithmeticLimit = ArithmeticLimit(
        rel_l2=0.004,
        mean_abs=0.0005,
        max_abs=0.04,
    )
    k2_output: ArithmeticLimit = ArithmeticLimit(
        rel_l2=0.0045,
        mean_abs=0.00065,
        max_abs=0.02,
    )
    k1_value_max_abs: float = 0.0


ENVELOPE = ReplayEnvelopeV1()


def tensor_metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float | bool]:
    if got.shape != want.shape:
        raise ValueError(f"metric tensors differ in shape: {tuple(got.shape)} vs {tuple(want.shape)}")
    got_f = got.float()
    want_f = want.float()
    delta = got_f - want_f
    finite = bool(
        torch.isfinite(got_f).all().item()
        and torch.isfinite(want_f).all().item()
        and torch.isfinite(delta).all().item()
    )
    if not finite:
        return {
            "finite": False,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "rel_l2": float("inf"),
        }
    return {
        "finite": True,
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float(
            (
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(want_f).clamp_min(1e-12)
            ).item()
        ),
    }


def tensor_scale_diagnostics(got: torch.Tensor, want: torch.Tensor) -> dict[str, object]:
    """Describe scale and the single worst element without changing any replay gate."""
    if got.shape != want.shape:
        raise ValueError(
            f"diagnostic tensors differ in shape: {tuple(got.shape)} vs {tuple(want.shape)}"
        )
    got_f = got.float()
    want_f = want.float()
    delta = got_f - want_f
    if not (
        torch.isfinite(got_f).all()
        and torch.isfinite(want_f).all()
        and torch.isfinite(delta).all()
    ):
        return {"finite": False}

    abs_delta = delta.abs()
    flat_index = int(abs_delta.reshape(-1).argmax().item())
    shape = tuple(int(x) for x in abs_delta.shape)
    remainder = flat_index
    coordinates = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        size = shape[axis]
        coordinates[axis] = remainder % size
        remainder //= size
    index = tuple(coordinates)
    got_worst = float(got_f[index].item())
    want_worst = float(want_f[index].item())
    delta_worst = float(delta[index].item())

    want_abs_max = float(want_f.abs().max().item())
    got_abs_max = float(got_f.abs().max().item())
    want_mean_abs = float(want_f.abs().mean().item())
    got_mean_abs = float(got_f.abs().mean().item())
    max_abs = float(abs(delta_worst))
    mean_abs = float(abs_delta.mean().item())

    ulp = None
    ulp_error = None
    magnitude = abs(want_worst)
    if math.isfinite(magnitude) and magnitude > 0.0:
        exponent = math.floor(math.log2(magnitude))
        ulp = math.ldexp(1.0, exponent - 7)
        if ulp > 0.0:
            ulp_error = max_abs / ulp

    return {
        "finite": True,
        "want_abs_max": want_abs_max,
        "got_abs_max": got_abs_max,
        "want_mean_abs": want_mean_abs,
        "got_mean_abs": got_mean_abs,
        "max_abs_over_want_abs_max": (
            max_abs / want_abs_max if want_abs_max > 0.0 else None
        ),
        "mean_abs_over_want_mean_abs": (
            mean_abs / want_mean_abs if want_mean_abs > 0.0 else None
        ),
        "worst": {
            "index": coordinates,
            "got": got_worst,
            "want": want_worst,
            "delta": delta_worst,
            "want_bf16_ulp_estimate": ulp,
            "abs_error_in_want_bf16_ulps": ulp_error,
        },
    }



def identity_split_half_rope_like(rope_freqs: torch.Tensor) -> torch.Tensor:
    """Return an identity rotation matrix with the same H3 RoPE geometry/dtype/device."""
    if rope_freqs.ndim != 6 or tuple(rope_freqs.shape[-2:]) != (2, 2):
        raise ValueError(
            "identity split-half RoPE requires [..., half, 2, 2] frequencies"
        )
    identity = torch.zeros_like(rope_freqs)
    identity[..., 0, 0] = 1
    identity[..., 1, 1] = 1
    return identity


def public_rms_norm(
    raw: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Public Keyless RMSNorm arithmetic: FP32 reduction/scale, materialize to input dtype."""
    if raw.ndim < 1 or raw.shape[-1] <= 0:
        raise ValueError("RMSNorm input must have a nonempty final dimension")
    if weight.shape != (raw.shape[-1],):
        raise ValueError(
            f"RMSNorm weight must have shape {(raw.shape[-1],)}, got {tuple(weight.shape)}"
        )
    work = raw.float()
    inv = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + float(eps))
    return (work * inv * weight.float()).to(dtype=raw.dtype)


def apply_split_half_rope_from_normalized(
    normalized: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    rot_dim: int,
) -> torch.Tensor:
    """Apply only split-half RoPE to an already-normalized [T,H,D] tensor."""
    if normalized.ndim != 3:
        raise ValueError("normalized route input must have shape [T,H,D]")
    if rot_dim <= 0 or rot_dim > normalized.shape[-1] or rot_dim % 2:
        raise ValueError("rot_dim must be a positive even prefix within head_dim")
    half = rot_dim // 2
    expected = (1, normalized.shape[0], 1, half, 2, 2)
    if tuple(rope_freqs.shape) != expected:
        raise ValueError(
            f"split-half RoPE requires rope_freqs {expected}, got {tuple(rope_freqs.shape)}"
        )
    freqs = rope_freqs.to(device=normalized.device, dtype=normalized.dtype)[0, :, 0]
    a = normalized[..., :half]
    b = normalized[..., half:rot_dim]
    f00 = freqs[..., 0, 0].unsqueeze(1)
    f01 = freqs[..., 0, 1].unsqueeze(1)
    f10 = freqs[..., 1, 0].unsqueeze(1)
    f11 = freqs[..., 1, 1].unsqueeze(1)
    first = f00 * a + f01 * b
    second = f10 * a + f11 * b
    return torch.cat((first, second, normalized[..., rot_dim:]), dim=-1).to(
        dtype=normalized.dtype
    )



def apply_split_half_rope_fp32_from_normalized(
    normalized: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    rot_dim: int,
) -> torch.Tensor:
    """Apply split-half RoPE with comfy-kitchen fused CUDA compute precision.

    The normalized tensor is already materialized in its storage dtype (BF16 for
    this replay). Rotated values and the 2x2 frequency matrix are promoted to
    FP32 for the multiply-adds, then the rotated prefix is cast back to the
    normalized dtype. The norm-only tail is preserved unchanged.
    """
    if normalized.ndim != 3:
        raise ValueError("normalized route input must have shape [T,H,D]")
    if rot_dim <= 0 or rot_dim > normalized.shape[-1] or rot_dim % 2:
        raise ValueError("rot_dim must be a positive even prefix within head_dim")
    half = rot_dim // 2
    expected = (1, normalized.shape[0], 1, half, 2, 2)
    if tuple(rope_freqs.shape) != expected:
        raise ValueError(
            f"split-half RoPE requires rope_freqs {expected}, got {tuple(rope_freqs.shape)}"
        )
    freqs = rope_freqs.to(device=normalized.device, dtype=torch.float32)[0, :, 0]
    a = normalized[..., :half].float()
    b = normalized[..., half:rot_dim].float()
    f00 = freqs[..., 0, 0].unsqueeze(1)
    f01 = freqs[..., 0, 1].unsqueeze(1)
    f10 = freqs[..., 1, 0].unsqueeze(1)
    f11 = freqs[..., 1, 1].unsqueeze(1)
    first = f00 * a + f01 * b
    second = f10 * a + f11 * b
    rotated = torch.cat((first, second), dim=-1).to(dtype=normalized.dtype)
    return torch.cat((rotated, normalized[..., rot_dim:]), dim=-1)

def metric_within_limit(metrics: dict[str, float | bool], limit: ArithmeticLimit) -> bool:
    if not bool(metrics.get("finite")):
        return False
    return (
        float(metrics["rel_l2"]) <= limit.rel_l2
        and float(metrics["mean_abs"]) <= limit.mean_abs
        and float(metrics["max_abs"]) <= limit.max_abs
    )


def scale_aware_metric_within_limit(
    metrics: dict[str, float | bool],
    scale: dict[str, object],
    limit: ScaleAwareArithmeticLimit,
) -> bool:
    if not bool(metrics.get("finite")) or not bool(scale.get("finite")):
        return False
    max_ratio = scale.get("max_abs_over_want_abs_max")
    mean_ratio = scale.get("mean_abs_over_want_mean_abs")
    worst = scale.get("worst")
    if (
        max_ratio is None
        or mean_ratio is None
        or not isinstance(worst, dict)
        or worst.get("abs_error_in_want_bf16_ulps") is None
    ):
        return False
    return (
        float(metrics["rel_l2"]) <= limit.rel_l2
        and float(max_ratio) <= limit.max_abs_over_want_abs_max
        and float(mean_ratio) <= limit.mean_abs_over_want_mean_abs
        and float(worst["abs_error_in_want_bf16_ulps"]) <= limit.worst_bf16_ulps
    )


def v2_envelope_dict() -> dict[str, object]:
    return asdict(V2_ENVELOPE)


def value_sum_within_limit(metrics: dict[str, float | bool], max_abs: float) -> bool:
    return bool(metrics.get("finite")) and float(metrics["max_abs"]) <= float(max_abs)


def replay_requires_process_failure(
    *,
    all_blocks_pass: bool,
    diagnostic_only: bool,
) -> bool:
    """Keep the frozen gate fail-closed unless the caller explicitly requests diagnostics."""
    return not bool(all_blocks_pass) and not bool(diagnostic_only)


def envelope_dict() -> dict[str, object]:
    return asdict(ENVELOPE)


def block_summary_oracle(
    route: torch.Tensor,
    raw_v: torch.Tensor,
    *,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce an already-materialized Comfy route and raw V over physical V blocks."""
    if route.shape != raw_v.shape or route.ndim != 3:
        raise ValueError("route/raw_v must share [T,H,D] shape")
    if route.shape[0] <= 0 or block_size <= 0:
        raise ValueError("route summary requires positive rows and block_size")

    rows, heads, dim = route.shape
    blocks = math.ceil(rows / block_size)
    padded_rows = blocks * block_size
    if padded_rows != rows:
        routed = torch.zeros(
            padded_rows, heads, dim, dtype=route.dtype, device=route.device
        )
        values = torch.zeros_like(routed)
        routed[:rows].copy_(route)
        values[:rows].copy_(raw_v)
    else:
        routed = route
        values = raw_v

    routed = routed.view(blocks, block_size, heads, dim)
    values = values.view(blocks, block_size, heads, dim)
    counts = torch.full(
        (blocks, 1, 1),
        block_size,
        device=route.device,
        dtype=torch.float32,
    )
    counts[-1] = rows - (blocks - 1) * block_size
    rc = (routed.float().sum(dim=1) / counts).to(route.dtype)
    vc = values.float().sum(dim=1).to(raw_v.dtype)
    return rc, vc


def checkpoint_tensor_names(kind: str, block_index: int) -> dict[str, str]:
    prefix = f"blocks.{int(block_index)}.attn."
    if kind == "teacher":
        return {
            "projection": prefix + "qkv_proj.weight",
            "q_norm": prefix + "q_norm.weight",
            "route_norm": prefix + "k_norm.weight",
        }
    if kind == "keyless":
        return {
            "projection": prefix + "qv_proj.weight",
            "q_norm": prefix + "q_norm.weight",
            "route_norm": prefix + "route_norm.weight",
        }
    raise ValueError(f"unsupported checkpoint kind: {kind!r}")


def split_projection(
    projection: torch.Tensor,
    *,
    kind: str,
    heads: int = 56,
    head_dim: int = 128,
    hidden_size: int = 5376,
) -> tuple[torch.Tensor, torch.Tensor]:
    inner = heads * head_dim
    if kind == "teacher":
        expected = (3 * inner, hidden_size)
        if tuple(projection.shape) != expected:
            raise ValueError(f"teacher qkv projection expected {expected}, got {tuple(projection.shape)}")
        q, _k, v = projection.split(inner, dim=0)
        return q, v
    if kind == "keyless":
        expected = (2 * inner, hidden_size)
        if tuple(projection.shape) != expected:
            raise ValueError(f"Keyless qv projection expected {expected}, got {tuple(projection.shape)}")
        return projection.split(inner, dim=0)
    raise ValueError(f"unsupported checkpoint kind: {kind!r}")


__all__ = [
    "ArithmeticLimit",
    "ScaleAwareArithmeticLimit",
    "ENVELOPE",
    "V2_ENVELOPE",
    "ReplayEnvelopeV1",
    "ReplayEnvelopeV2",
    "block_summary_oracle",
    "checkpoint_tensor_names",
    "envelope_dict",
    "identity_split_half_rope_like",
    "public_rms_norm",
    "apply_split_half_rope_from_normalized",
    "apply_split_half_rope_fp32_from_normalized",
    "replay_requires_process_failure",
    "metric_within_limit",
    "scale_aware_metric_within_limit",
    "v2_envelope_dict",
    "split_projection",
    "tensor_metrics",
    "tensor_scale_diagnostics",
    "value_sum_within_limit",
]
