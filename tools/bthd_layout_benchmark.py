#!/usr/bin/env python3
"""A/B the real SM120 Sol-Attn kernel on production BTHD layouts.

This probe isolates the exact question that full ComfyUI workflow timings cannot:
for identical Q/K/V values and identical sparse-routing semantics, is it faster to
pass the production strided BTHD views directly, or to materialize the old
``transpose(...).contiguous()`` layout before each kernel call?

It reports three paths per case:

* ``strided``: current Sol-H3 behavior;
* ``contiguous_kernel``: kernel-only timing on pre-materialized contiguous Q/K/V;
* ``copy_plus_kernel``: old bridge behavior, including per-call contiguous copies.

The mixed-grid case reproduces the production stride tuple observed from Flow's
external sequence: Q/V ``[7168, 21504, 128, 1]`` and K
``[7168, 7168, 128, 1]`` at 56 heads. The native rectangular control uses
already-contiguous BTHD tensors, for which the old ``.contiguous()`` bridge is a
no-op; it therefore exposes the benchmark's normal timing noise floor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# Direct execution (``python tools/bthd_layout_benchmark.py``) puts ``tools/``
# at sys.path[0], not the repository root. Bootstrap the checkout explicitly so
# the documented command works without requiring an editable package install.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sol_h3 import sparse  # noqa: E402


HEAD_DIM = 128
HEADS = 56


def _comfy_split_bthd(rows: int, *, device: torch.device):
    """Return exact BTHD views produced by Comfy's split->view->transpose path."""
    inner = HEADS * HEAD_DIM
    packed = torch.randn(rows, 3 * inner, dtype=torch.bfloat16, device=device)
    out = []
    for part in packed.split(inner, dim=-1):
        thd = part.view(rows, HEADS, HEAD_DIM)
        bhtd = thd.transpose(0, 1).unsqueeze(0)
        out.append(bhtd.transpose(1, 2))
    return packed, tuple(out)


def _contiguous_bthd(rows: int, *, device: torch.device):
    """Construct BTHD with the size-one batch stride seen in production."""
    thd = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    return thd.transpose(0, 1).unsqueeze(0).transpose(1, 2)


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _time_once(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = float(start.elapsed_time(end))
    del out
    return cuda_ms, wall_ms


def _call(kernel, q, k, v):
    return kernel(
        q,
        k,
        v,
        tau=1.0,
        thresh_type="diag",
        kv_splits=1,
        sink_start=0,
        sink_tokens=0,
    )


def _benchmark_case(name, kernel, q, k, v, *, warmup: int, repeats: int):
    original = (q, k, v)
    contiguous = tuple(x.contiguous() for x in original)
    materialized_bytes = sum(
        x.numel() * x.element_size() for x in original if not x.is_contiguous()
    )

    def strided():
        return _call(kernel, *original)

    def contiguous_kernel():
        return _call(kernel, *contiguous)

    def copy_plus_kernel():
        copied = tuple(x.contiguous() for x in original)
        return _call(kernel, *copied)

    # Compile/warm both CuTe stride specializations before collecting samples.
    with torch.inference_mode():
        for _ in range(warmup):
            out = strided()
            del out
            out = contiguous_kernel()
            del out
            out = copy_plus_kernel()
            del out
        torch.cuda.synchronize()

        cuda_samples = defaultdict(list)
        wall_samples = defaultdict(list)
        names = ("strided", "contiguous_kernel", "copy_plus_kernel")
        funcs = {
            "strided": strided,
            "contiguous_kernel": contiguous_kernel,
            "copy_plus_kernel": copy_plus_kernel,
        }
        # Reverse order every repetition so thermal/clock drift cannot always
        # favor the same path.
        for rep in range(repeats):
            order = names if rep % 2 == 0 else tuple(reversed(names))
            for path in order:
                cuda_ms, wall_ms = _time_once(funcs[path])
                cuda_samples[path].append(cuda_ms)
                wall_samples[path].append(wall_ms)

    cuda = {path: _summary(cuda_samples[path]) for path in names}
    wall = {path: _summary(wall_samples[path]) for path in names}
    strided_med = cuda["strided"]["median_ms"]
    contiguous_med = cuda["contiguous_kernel"]["median_ms"]
    old_med = cuda["copy_plus_kernel"]["median_ms"]

    return {
        "name": name,
        "shape_q": list(q.shape),
        "shape_kv": list(k.shape),
        "strides": [list(x.stride()) for x in original],
        "contiguous_strides": [list(x.stride()) for x in contiguous],
        "input_materialized_bytes_per_old_call": materialized_bytes,
        "cuda": cuda,
        "wall": wall,
        "derived": {
            "strided_vs_old_pct": 100.0 * (strided_med / old_med - 1.0),
            "strided_kernel_vs_contiguous_kernel_pct": 100.0 * (
                strided_med / contiguous_med - 1.0
            ),
            "old_copy_plus_kernel_minus_contiguous_kernel_ms": old_med - contiguous_med,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--mixed-rows", type=int, default=43545)
    parser.add_argument("--native-q-rows", type=int, default=5180)
    parser.add_argument("--native-kv-rows", type=int, default=21486)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3:
        parser.error("use --warmup >= 1 and --repeats >= 3")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise SystemExit("real SM120 is required")

    torch.manual_seed(args.seed)
    kernel = sparse.load_kernel(device)

    # Mixed Flow production layout: Q/V still reference whole-QKV slabs while
    # K has already been materialized into the exact restricted sequence.
    mixed_packed, mixed_parts = _comfy_split_bthd(args.mixed_rows, device=device)
    mixed_q, _, mixed_v = mixed_parts
    mixed_k = _contiguous_bthd(args.mixed_rows, device=device)
    expected_mixed = [
        [HEADS * HEAD_DIM, 3 * HEADS * HEAD_DIM, HEAD_DIM, 1],
        [HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEAD_DIM, 1],
        [HEADS * HEAD_DIM, 3 * HEADS * HEAD_DIM, HEAD_DIM, 1],
    ]
    actual_mixed = [list(x.stride()) for x in (mixed_q, mixed_k, mixed_v)]
    if actual_mixed != expected_mixed:
        raise RuntimeError(f"mixed production stride reconstruction failed: {actual_mixed}")

    mixed = _benchmark_case(
        "mixed_square_production_stride",
        kernel,
        mixed_q,
        mixed_k,
        mixed_v,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    del mixed_packed, mixed_parts, mixed_q, mixed_k, mixed_v
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Native VDN control: all BTHD inputs are already contiguous. The old
    # bridge's .contiguous() calls therefore return the same storages.
    native_q = _contiguous_bthd(args.native_q_rows, device=device)
    native_k = _contiguous_bthd(args.native_kv_rows, device=device)
    native_v = _contiguous_bthd(args.native_kv_rows, device=device)
    native = _benchmark_case(
        "native_rectangular_contiguous_control",
        kernel,
        native_q,
        native_k,
        native_v,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    result = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "backend": getattr(kernel, "backend_name", None),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [mixed, native],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
