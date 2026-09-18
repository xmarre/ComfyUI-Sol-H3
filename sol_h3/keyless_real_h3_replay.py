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


def metric_within_limit(metrics: dict[str, float | bool], limit: ArithmeticLimit) -> bool:
    if not bool(metrics.get("finite")):
        return False
    return (
        float(metrics["rel_l2"]) <= limit.rel_l2
        and float(metrics["mean_abs"]) <= limit.mean_abs
        and float(metrics["max_abs"]) <= limit.max_abs
    )


def value_sum_within_limit(metrics: dict[str, float | bool], max_abs: float) -> bool:
    return bool(metrics.get("finite")) and float(metrics["max_abs"]) <= float(max_abs)


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
    "ENVELOPE",
    "ReplayEnvelopeV1",
    "block_summary_oracle",
    "checkpoint_tensor_names",
    "envelope_dict",
    "metric_within_limit",
    "split_projection",
    "tensor_metrics",
    "tensor_scale_diagnostics",
    "value_sum_within_limit",
]
