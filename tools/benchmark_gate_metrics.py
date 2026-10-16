"""GPU-only microbenchmark for exhaustive Sol arithmetic-gate metric reducers.

Runs without model weights, ComfyUI startup or video sampling:
python tools/benchmark_gate_metrics.py

Reports full-domain numerical agreement and warm-run wall time for both the
reference PyTorch reducer and the candidate SM120 Triton reducer. No result is
classified as a speedup until measured on the target GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sol_h3.gate_metrics_triton import gate_partials  # noqa: E402
from sol_h3.sparse import ErrorAccumulator  # noqa: E402


def _accumulate(got, want, *, fused):
    acc = ErrorAccumulator(got.device, enable_fused=fused)
    for start in range(0, got.shape[1], 4096):
        stop = min(got.shape[1], start + 4096)
        acc.add(got[:, start:stop], want[:, start:stop])
    return acc.statistics()


def _reference(got, want):
    return _accumulate(got, want, fused=False)


def _candidate(got, want):
    return _accumulate(got, want, fused=True)


def _time_cuda(func, rounds):
    measurements = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        measurements.append((time.perf_counter() - start) * 1000.0)
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required")
    report = []
    with torch.no_grad(), torch.cuda.device(device):
        for rows in (3168, 3960, 6580):
            # Produces the usual contiguous CuTe BTHD output and noncontiguous
            # SDPA-transposed BTHD reference (last dimension still contiguous).
            torch.manual_seed(619)
            want = torch.randn((1, 56, rows, 128), device=device, dtype=torch.bfloat16).transpose(1, 2)
            got = (want.contiguous().float() * 0.999).bfloat16()
            reference = _reference(got, want)
            # Enforce actual Triton activation, not a silent PyTorch fallback.
            if gate_partials(got[:, :min(rows, 4096)], want[:, :min(rows, 4096)]) is None:
                raise RuntimeError("candidate reducer did not activate on the target layout")
            candidate = _candidate(got, want)
            finite_match = bool(candidate[0]) == bool(reference[0])
            errors = [abs(float(a) - float(b)) for a, b in zip(candidate[1:], reference[1:], strict=True)]
            tol = [max(2e-6 * abs(float(v)), 2e-8) for v in reference[1:]]
            agree = finite_match and all(delta <= epsilon for delta, epsilon in zip(errors, tol, strict=True))
            if not agree:
                raise RuntimeError(f"arithmetic metric mismatch for {rows} rows: errors={errors} limits={tol}")
            # Warm both routes independently; compilation time is reported by
            # the first activation above and is not hidden in the gate mean.
            _reference(got, want)
            _candidate(got, want)
            old = _time_cuda(lambda: _reference(got, want), args.rounds)
            new = _time_cuda(lambda: _candidate(got, want), args.rounds)
            report.append({
                "q_rows": rows,
                "heads": 56,
                "head_dim": 128,
                "legacy_median_ms": statistics.median(old),
                "fused_median_ms": statistics.median(new),
                "speedup_ratio": statistics.median(old) / statistics.median(new),
                "maximum_statistic_difference": max(errors),
                "full_domain_compared": True,
            })
    print(json.dumps({"device": str(device), "results": report}, indent=2))


if __name__ == "__main__":
    main()
