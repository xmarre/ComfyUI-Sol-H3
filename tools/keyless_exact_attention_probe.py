#!/usr/bin/env python3
"""Bounded SM120 arithmetic/allocation probe for Keyless phase K2.

Synthetic evidence only. Production promotion additionally requires real H3 RoPE,
real early/mid/late route-norm weights, decoded media and the CuTe production path.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from sol_h3.keyless_exact_attention import (
    CANONICAL_SCALE,
    CONTRACT,
    exact_attention,
    materialized_exact_reference,
)
from sol_h3.keyless_route_summary import HEAD_DIM, NORM_EPS, ROPE_HALF_DIM


def _rope(rows: int, device: torch.device) -> torch.Tensor:
    angles = (
        torch.randn(rows, ROPE_HALF_DIM, device=device, dtype=torch.float32) * 0.2
    )
    rope = torch.zeros(
        1,
        rows,
        1,
        ROPE_HALF_DIM,
        2,
        2,
        device=device,
        dtype=torch.bfloat16,
    )
    rope[0, :, 0, :, 0, 0] = angles.cos().to(torch.bfloat16)
    rope[0, :, 0, :, 1, 0] = angles.sin().to(torch.bfloat16)
    return rope


def _metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float | bool]:
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
            "mean_abs": float("inf"),
            "max_abs": float("inf"),
            "rel_l2": float("inf"),
        }
    return {
        "finite": True,
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "rel_l2": float(
            (
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(want_f).clamp_min(1e-12)
            ).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--q-tokens", type=int, default=257)
    parser.add_argument("--v-tokens", type=int, default=257)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.q_tokens <= 0 or args.v_tokens <= 0:
        raise SystemExit("q/v token counts must be positive")
    if args.heads <= 0:
        raise SystemExit("heads must be positive")
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")
    if args.q_tokens > 2048 or args.v_tokens > 2048:
        raise SystemExit(
            "bounded K2 probe caps q/v tokens at 2048; use a dedicated production "
            "harness for larger arithmetic campaigns"
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("K2 probe requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise SystemExit(f"K2 probe requires SM120, got {capability}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    q = torch.randn(
        args.q_tokens,
        args.heads,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        args.v_tokens,
        args.heads,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    rope = _rope(args.v_tokens, device)
    out = torch.empty_like(q)

    exact_attention(
        q,
        v,
        weight,
        NORM_EPS,
        rope,
        scale=CANONICAL_SCALE,
        out=out,
    )
    torch.cuda.synchronize(device)

    want = materialized_exact_reference(
        q,
        v,
        weight,
        NORM_EPS,
        rope,
        scale=CANONICAL_SCALE,
    )
    torch.cuda.synchronize(device)
    arithmetic = _metrics(out, want)

    times_ms = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        exact_attention(
            q,
            v,
            weight,
            NORM_EPS,
            rope,
            scale=CANONICAL_SCALE,
            out=out,
        )
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - started) * 1000.0)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    exact_attention(
        q,
        v,
        weight,
        NORM_EPS,
        rope,
        scale=CANONICAL_SCALE,
        out=out,
    )
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    temporary_peak_delta = max(0, peak_allocated - baseline_allocated)
    route_tensor_bytes = v.numel() * v.element_size()

    record = {
        "contract": CONTRACT,
        "promotion_evidence": False,
        "device": str(device),
        "capability": list(capability),
        "torch_version": torch.__version__,
        "q_shape": list(q.shape),
        "v_shape": list(v.shape),
        "dtype": str(q.dtype),
        "scale": CANONICAL_SCALE,
        "epsilon": NORM_EPS,
        "arithmetic": arithmetic,
        "timing_ms": {
            "median": statistics.median(times_ms),
            "min": min(times_ms),
            "max": max(times_ms),
            "samples": times_ms,
        },
        "allocation": {
            "baseline_allocated": baseline_allocated,
            "peak_allocated": peak_allocated,
            "temporary_peak_delta": temporary_peak_delta,
            "full_materialized_route_bytes": route_tensor_bytes,
            "temporary_delta_below_route_bytes": (
                temporary_peak_delta < route_tensor_bytes
            ),
            "output_preallocated": True,
        },
        "limitations": [
            "synthetic Q/V, route norm and RoPE",
            "all blocks selected; no Sol sparse selector",
            "experimental Triton K2 primitive, not vendored CuTe production mainloop",
            "no decoded-media evidence",
            "no promotion threshold frozen from real H3 evidence",
        ],
    }
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
