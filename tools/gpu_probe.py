"""GPU-resident affine probe. No model weights or host activation copies."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from sol_h3.exact import affine, native_affine  # noqa: E402


def measure(function, device, repeats):
    for _ in range(3):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--prefix", type=int, default=951)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU fallback")
    if not 0 < args.prefix < args.tokens or min(args.hidden, args.repeats) <= 0:
        parser.error("Require 0 < prefix < tokens and positive hidden/repeats")
    torch.manual_seed(123)
    with torch.cuda.device(device), torch.inference_mode():
        h = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
        params = torch.randn(7, args.hidden * 6, device=device, dtype=torch.float32)
        shift, scale = params.chunk(6, dim=-1)[:2]
        segments = [(0, args.prefix, 1), (args.prefix, args.tokens, 3)]
        verified = set()
        def reference():
            out = h.clone()
            for a, b, row in segments:
                native_affine(out[a:b], shift, scale, row)
            return out
        def candidate():
            return affine(h.clone(), shift, scale, segments, verified)
        if not torch.equal(reference(), candidate()):
            raise RuntimeError("Full affine operator parity failed")
        print(json.dumps({"synthetic_operator_probe": True, "bitwise_equal": True,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "capability": torch.cuda.get_device_capability(device),
            "tokens": args.tokens, "hidden": args.hidden,
            "scope": "warmed operators including identical input clone; excludes setup and initial parity gate",
            "native": measure(reference, device, args.repeats),
            "fused": measure(candidate, device, args.repeats)}, indent=2))


if __name__ == "__main__":
    main()
