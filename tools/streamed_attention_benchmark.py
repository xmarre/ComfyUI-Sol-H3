"""Measure the real native H3 projection/Sol path with synthetic activations.

This isolates projection memory and latency. It is not an end-to-end sampler or
decoded-quality result. Run in the installed ComfyUI environment on SM120/SM121.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_attention(hidden, heads, projection):
    import comfy.ops
    from comfy.ldm.minimax.model import Attention
    from comfy.quant_ops import QuantizedTensor

    operations = (comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
                  if projection == "int8-convrot" else comfy.ops.manual_cast)
    attn = Attention(hidden, heads, 128, 1e-5, dtype=torch.bfloat16, device="cuda", operations=operations).eval()
    for linear, n, m in ((attn.qkv_proj, heads * 128 * 3, hidden), (attn.out_proj, hidden, heads * 128)):
        weight = torch.randn(n, m, device="cuda", dtype=torch.bfloat16) / m ** 0.5
        if projection == "int8-convrot":
            weight = QuantizedTensor.from_float(weight, "TensorWiseINT8Layout", convrot=True,
                                                convrot_groupsize=256, is_weight=True)
            linear.quant_format = "int8_tensorwise"
            linear.layout_type = "TensorWiseINT8Layout"
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    attn.q_norm.weight = torch.nn.Parameter(torch.ones(128, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    attn.k_norm.weight = torch.nn.Parameter(torch.ones(128, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    return attn


def measure(function):
    gc.collect()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    started = time.perf_counter()
    out = function()
    end.record()
    torch.cuda.synchronize()
    result = {
        "wall_ms": (time.perf_counter() - started) * 1000, "cuda_ms": start.elapsed_time(end),
        "baseline_allocated": allocated, "baseline_reserved": reserved,
        "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated": torch.cuda.max_memory_allocated() - allocated,
        "live_with_output": torch.cuda.memory_allocated(),
    }
    # Copies for numerical comparison happen after timing and peak accounting.
    reference = out.cpu()
    del out
    gc.collect()
    torch.cuda.synchronize()
    result["retained_after_output_release"] = torch.cuda.memory_allocated() - allocated
    return result, reference


def run_arm(attn, x, rope, prefix, chunk, repeats, dense):
    from sol_h3.contracts import Config
    from sol_h3.runtime import Request
    from sol_h3 import sparse, streamed
    state = Request(Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0, qkv_chunk_size=chunk))

    def override(original, q, k, v, heads, mask=None, **kwargs):
        def native(qd, kd, vd):
            return original(qd, kd, vd, heads, mask=mask, skip_reshape=True,
                            skip_output_reshape=True, _inside_attn_wrapper=True).transpose(1, 2)
        scope = streamed._STREAM.get()
        if scope is not None:
            scope.check(q, k, v)
        if dense:
            return native(q, k, v).reshape(1, q.shape[2], -1)
        return sparse.attention(q, k, v, prefix, state.config, state, dense_attention=native,
                                query_start=0 if scope is None else scope.query_start, kv_workspace=scope)

    options = {"optimized_attention_override": override}

    def execute():
        if chunk:
            return streamed.attention(attn, x, rope, options, chunk)
        return attn(x, rope_freqs=rope, transformer_options=options)

    first, reference = measure(execute)
    warm = []
    for _ in range(repeats):
        result, output = measure(execute)
        if not torch.equal(output, reference):
            raise RuntimeError("Identical-input repeated attention changed within one arm")
        warm.append(result)
        del output
    gates = state.gates
    state = options = None
    del execute, override
    gc.collect()
    torch.cuda.synchronize()
    return {
        "qkv_chunk_size": chunk, "first_use": first, "warm": warm,
        "warm_median_wall_ms": statistics.median(r["wall_ms"] for r in warm),
        "warm_median_cuda_ms": statistics.median(r["cuda_ms"] for r in warm),
        "arithmetic_gates": gates, "allocated_after_request_cleanup": torch.cuda.memory_allocated(),
    }, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-path", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[16384, 32768, 65536, 100000])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--prefix", type=int, default=193)
    parser.add_argument("--projection", choices=("bf16", "int8-convrot"), default="int8-convrot")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dense", action="store_true", help="Measure the inherited dense warmup architecture")
    parser.add_argument("--order", choices=("native-first", "streamed-first"), default="native-first")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.chunk_size <= 0 or args.chunk_size % 64 or min(args.rows) <= args.prefix:
        parser.error("Need positive repeats, a positive 64-aligned chunk and rows exceeding prefix")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        parser.error("Requires real SM120/SM121 CUDA execution")
    sys.path.insert(0, str(args.comfyui_path.resolve()))
    # Comfy owns CLI parsing in its server; this standalone runner owns these args.
    sys.argv = [sys.argv[0]]
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    from comfy.ldm.minimax.model import Attention, rope_rotation_table
    from sol_h3.native_contract import function_digest
    from sol_h3.provenance import verify_source
    from sol_h3.sparse import arithmetic_gate_passes, error_metrics
    from sol_h3.streamed import NATIVE_ATTENTION_DIGESTS

    digest = function_digest(Attention.forward)
    if digest not in NATIVE_ATTENTION_DIGESTS:
        raise RuntimeError("Benchmark native Attention.forward is outside the audited producer contract")
    report = {
        "contract": "sol-h3-streamed-projection-benchmark-v1", "end_to_end": False,
        "synthetic_activations": True, "numerical_acceptance": "existing Sol all-selected error limits",
        "baseline": "native full QKV projection with candidate bounded arithmetic qualification",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "native_attention_digest": digest, "kernel_contract": verify_source()["contract"],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "sol_h3/streamed.py", "sol_h3/sparse.py", "sol_h3/sol_manifest.json")},
        "projection": args.projection, "heads": args.heads, "hidden": args.hidden, "dense": args.dense,
        "arm_order": args.order,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.manual_seed(7483)
        attn = make_attention(args.hidden, args.heads, args.projection)
        for rows in args.rows:
            torch.manual_seed(7483 + rows)
            x = torch.randn(rows, args.hidden, device="cuda", dtype=torch.bfloat16)
            angles = torch.randn(rows, 96, device="cuda")
            rope = rope_rotation_table(angles, x.dtype)
            del angles
            order = (0, args.chunk_size) if args.order == "native-first" else (args.chunk_size, 0)
            arms = {chunk: run_arm(attn, x, rope, args.prefix, chunk, args.repeats, args.dense) for chunk in order}
            baseline, expected = arms[0]
            optimized, actual = arms[args.chunk_size]
            metrics = error_metrics(actual, expected)
            case = {"rows": rows, "production_stream_eligible_by_length": rows > 4 * args.chunk_size,
                    "native": baseline, "streamed": optimized,
                    "output_error": metrics, "arithmetic_pass": arithmetic_gate_passes(metrics)}
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del expected, actual, x, rope, arms
            if not case["arithmetic_pass"]:
                raise RuntimeError(f"Full native-vs-streamed output exceeded the frozen arithmetic limits: {metrics}")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]), "end_to_end": False}))


if __name__ == "__main__":
    main()
