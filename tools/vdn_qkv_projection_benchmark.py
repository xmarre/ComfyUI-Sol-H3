"""Isolate VDN's *real* quantized QKV projection and bypass-adapter cost.

This measures the full packed+V-clone path against bounded independent carriers,
not a VDN sampler, Sol sparse attention, or rendered output. Run with the
installed ComfyUI and VDN repos on the target CUDA GPU. No model weights or
application code are modified.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(operation, rows):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    sample_indices = torch.tensor([0, rows // 2, rows - 1], device=result[0].device)
    samples = tuple(t.index_select(0, sample_indices).cpu() for t in result)
    measurement = {
        "cuda_ms": start.elapsed_time(end),
        "incremental_peak_allocated_mib": (torch.cuda.max_memory_allocated() - allocated_before) / 1048576,
        "incremental_peak_reserved_mib": (torch.cuda.max_memory_reserved() - reserved_before) / 1048576,
        "live_output_mib": (torch.cuda.memory_allocated() - allocated_before) / 1048576,
    }
    del result, sample_indices
    torch.cuda.synchronize()
    measurement["retained_after_release_mib"] = (torch.cuda.memory_allocated() - allocated_before) / 1048576
    return measurement, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-path", type=Path, required=True)
    parser.add_argument("--vdn-path", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[54744, 107630])
    parser.add_argument("--chunks", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.repeats < 1 or args.rank < 1 or not args.rows or min(args.rows) < 1
            or not args.chunks or any(chunk <= 0 or chunk % 64 for chunk in args.chunks)):
        parser.error("rows/repeats/rank must be positive; chunks must be positive multiples of 64")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        parser.error("SM120/SM121 CUDA GPU required")
    sys.path.insert(0, str(args.comfyui_path.resolve()))
    sys.path.insert(0, str(args.vdn_path.resolve()))
    # Avoid handing standalone benchmark flags to ComfyUI's global CLI parser.
    sys.argv = [sys.argv[0]]
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    from vdn_h3.apply import _PostForwardLoRA
    from sol_h3.streamed import vdn_project_qkv
    from tools.streamed_attention_benchmark import make_attention

    torch.manual_seed(16102026)
    attn = make_attention(5376, 56, "int8-convrot", device="cuda")
    attn.eval().requires_grad_(False)
    down = torch.randn(args.rank, 5376, device="cuda", dtype=torch.bfloat16) * 0.01
    up = torch.randn(56 * 128 * 3, args.rank, device="cuda", dtype=torch.bfloat16) * 0.01
    adapter = _PostForwardLoRA([(down, up, 0.5)])
    adapter.prepare(attn.qkv_proj)
    handle = attn.qkv_proj.register_forward_hook(adapter)

    report = {
        "contract": "vdn-bounded-quantized-qkv-benchmark-v1",
        "synthetic_inputs": True, "end_to_end": False,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "rank": args.rank,
        "projection": "int8-convrot", "cases": [],
    }

    try:
        with torch.inference_mode():
            for rows in args.rows:
                x = torch.randn(rows, 5376, dtype=torch.bfloat16, device="cuda")

                def native():
                    packed = attn.qkv_proj(x)
                    q, k, v = packed.split(56 * 128, dim=-1)
                    return q.view(rows, 56, 128), k.view(rows, 56, 128), v.view(rows, 56, 128).clone()

                def chunked(size):
                    return vdn_project_qkv(attn.qkv_proj, x, 56, 128, size)

                arms = (0, *args.chunks)
                recorded = {}
                samples = {}
                for arm in arms:
                    operation = native if arm == 0 else lambda size=arm: chunked(size)
                    # One first-use invocation must not be treated as hot timing.
                    measure(operation, rows)
                    warm = []
                    for _ in range(args.repeats):
                        data, sampled = measure(operation, rows)
                        warm.append(data)
                        if arm not in samples:
                            samples[arm] = sampled
                        else:
                            for got, expected in zip(sampled, samples[arm], strict=True):
                                torch.testing.assert_close(got, expected, atol=0, rtol=0)
                    recorded[str(arm)] = {
                        "samples": warm,
                        "hot_cuda_median_ms": statistics.median(t["cuda_ms"] for t in warm),
                        "hot_peak_allocated_mib": statistics.median(t["incremental_peak_allocated_mib"] for t in warm),
                    }
                # Projection and post-hook are row-separable. BF16 GEMM may change
                # reduction shapes across chunk boundaries; compare with tolerance.
                for arm in args.chunks:
                    for actual, reference in zip(samples[arm], samples[0], strict=True):
                        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
                report["cases"].append({"rows": rows, "arms": recorded})
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                del x, samples
    finally:
        handle.remove()
        adapter.clear()
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]), "end_to_end": False}))


if __name__ == "__main__":
    main()
