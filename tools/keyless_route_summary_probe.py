"""Bounded CUDA probe for the Keyless raw-V route-summary primitive.

This answers one narrow question: does phase-K1 compute routed V64 centroids and
raw-V sums consistently with the materialized Keyless oracle without allocating a
full global route tensor?  It does not test sparse selection, the CuTe mainloop,
decoded media, or end-to-end performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from sol_h3.keyless_route_summary import (  # noqa: E402
    BLOCK_SIZE,
    CONTRACT,
    HEAD_DIM,
    NORM_EPS,
    ROPE_HALF_DIM,
    route_summary,
    route_summary_reference,
)


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
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "rel_l2": float("inf"),
            "exact_fraction": 0.0,
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
        "exact_fraction": float((got == want).float().mean().item()),
    }


def _rope(rows: int, device: torch.device, seed: int) -> torch.Tensor:
    # Synthetic angles deliberately exercise every rotary pair. Production evidence
    # must additionally replay exact Comfy H3 rope rows captured from a real call.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    angles = torch.randn(
        rows,
        ROPE_HALF_DIM,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * 0.35
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


def _measure(function, device: torch.device, repeats: int) -> dict[str, float | int]:
    for _ in range(3):
        output = function()
        del output
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del output
    torch.cuda.synchronize(device)
    return {
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "baseline_allocated_bytes": int(baseline_allocated),
        "peak_allocated_delta_bytes": int(
            torch.cuda.max_memory_allocated(device) - baseline_allocated
        ),
        "baseline_reserved_bytes": int(baseline_reserved),
        "peak_reserved_delta_bytes": int(
            torch.cuda.max_memory_reserved(device) - baseline_reserved
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=4097)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU fallback")
    if torch.cuda.get_device_capability(device) != (12, 0):
        parser.error("This evidence probe is scoped to SM120")
    if args.tokens <= 0 or args.heads <= 0 or args.repeats <= 0:
        parser.error("tokens, heads and repeats must be positive")

    torch.manual_seed(args.seed)
    with torch.cuda.device(device), torch.inference_mode():
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        v = torch.randn(
            args.tokens,
            args.heads,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        norm_weight = torch.randn(
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        rope = _rope(args.tokens, device, args.seed + 1)

        # Compile/execute the candidate before allocating the materialized oracle.
        candidate_rc, candidate_vc = route_summary(v, norm_weight, NORM_EPS, rope)
        torch.cuda.synchronize(device)
        candidate_shape = tuple(candidate_rc.shape)
        if candidate_vc.shape != candidate_rc.shape:
            raise RuntimeError("candidate RC/VC shapes differ")

        oracle_rc, oracle_vc = route_summary_reference(
            v, norm_weight, NORM_EPS, rope
        )
        torch.cuda.synchronize(device)
        rc_metrics = _metrics(candidate_rc, oracle_rc)
        vc_metrics = _metrics(candidate_vc, oracle_vc)
        if not rc_metrics["finite"] or not vc_metrics["finite"]:
            raise RuntimeError("route-summary arithmetic produced non-finite values")

        del candidate_rc, candidate_vc, oracle_rc, oracle_vc
        torch.cuda.synchronize(device)
        timing = _measure(
            lambda: route_summary(v, norm_weight, NORM_EPS, rope),
            device,
            args.repeats,
        )

        blocks = (args.tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        summary_output_bytes = (
            2 * blocks * args.heads * HEAD_DIM * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        full_route_bytes = (
            args.tokens * args.heads * HEAD_DIM * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        allocation_bounded = timing["peak_allocated_delta_bytes"] < full_route_bytes

        result = {
            "probe": "keyless_route_summary_k1_v1",
            "contract": CONTRACT,
            "synthetic_operator_probe": True,
            "promotion_evidence": False,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "tokens": args.tokens,
            "heads": args.heads,
            "head_dim": HEAD_DIM,
            "blocks": blocks,
            "candidate_shape": list(candidate_shape),
            "partial_last_block_rows": args.tokens - (blocks - 1) * BLOCK_SIZE,
            "route_centroid_error": rc_metrics,
            "raw_value_sum_error": vc_metrics,
            "candidate": timing,
            "summary_output_bytes": summary_output_bytes,
            "full_materialized_route_bytes": full_route_bytes,
            "peak_allocation_below_full_route": allocation_bounded,
            "scope": (
                "synthetic raw-V/weight/RoPE K1 arithmetic only; excludes CuTe mainloop, "
                "sparse selector, real checkpoint route_norm/RoPE, decoded media and end-to-end timing"
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if not allocation_bounded:
            raise RuntimeError(
                "candidate peak allocation is not bounded below one full materialized route tensor"
            )


if __name__ == "__main__":
    main()
