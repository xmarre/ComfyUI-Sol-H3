"""SM120 synthetic SOL + inherited Sage probe; does not evaluate media quality."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sol_h3.contracts import Config
from sol_h3.runtime import BlockPatch, Request, _FORWARD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("pytorch", "sage", "sage3"), default="sage")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--prefix", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.prefix < args.tokens:
        parser.error("prefix must lie strictly inside the sequence")
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "SM120 required"
    def dense(original, q, k, v, heads, **kw):
        if args.backend == "sage":
            from sageattention import sageattn
            out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        elif args.backend == "sage3":
            from sageattn3 import sageattn3_blackwell
            out = sageattn3_blackwell(q, k, v, is_causal=False)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out if kw.get("skip_output_reshape") else out.transpose(1, 2).reshape(1, q.shape[2], -1)
    config = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(config)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=None))])
    layout = SimpleNamespace(seq_len=args.tokens, segments=[(0, args.prefix, "text"), (args.prefix, args.tokens, "video")])
    torch.manual_seed(17)
    q, k, v = (torch.randn(1, args.heads, args.tokens, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    def block(call_args):
        options = call_args["transformer_options"]
        result = options["optimized_attention_override"](None, q, k, v, args.heads,
                    skip_reshape=True, transformer_options=options, _inside_attn_wrapper=True)
        assert result.shape == (1, args.tokens, args.heads * 128)
        reference = dense(None, q[:, :, :args.prefix], k, v, args.heads)
        torch.testing.assert_close(result[:, :args.prefix], reference, rtol=0, atol=0)
        return {"img": result}
    with torch.no_grad():
        for _ in range(2):
            token = _FORWARD.set((model, state, state.evaluations, set(), []))
            try:
                BlockPatch(0, config)({"img": q.new_empty(args.tokens, args.heads * 128),
                    "layout": layout, "transformer_options": {"optimized_attention_override": dense}},
                    {"original_block": block})
            finally:
                _FORWARD.reset(token)
    torch.cuda.synchronize()
    assert state.sparse_calls == 2, dict(state.fallbacks)
    print(json.dumps({"backend": args.backend, "gpu": torch.cuda.get_device_name(),
                      **config.metadata(), "sol_backend": state.kernel.backend_name,
                      "sol_source_tree_verified": state.kernel.source_tree_verified,
                      "sparse_calls": state.sparse_calls, "arithmetic_gates": state.gates,
                      "prefix_parity": True, "fallbacks": dict(state.fallbacks)}, indent=2))


if __name__ == "__main__":
    main()
