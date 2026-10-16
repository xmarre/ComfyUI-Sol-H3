"""Execute gather/tile geometry against the upstream reducer on CPU, in FP32.

Triton 3.6's interpreter does not model BF16 descriptor I/O correctly. This test
therefore checks executable address/reduction semantics in FP32; actual BF16
parity and sparse route preservation require the separate SM120 acceptance test.
Run with TRITON_INTERPRET=1.
"""

from pathlib import Path
import os
import sys

import torch
from triton.tools.tensor_descriptor import TensorDescriptor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sol_h3.grouped_kv_triton import _gather_summaries  # noqa: E402
from sol_h3._vendor.sol_attn.preprocess import _reduce_kv_kernel  # noqa: E402


def main():
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("Run with TRITON_INTERPRET=1; this is a CPU geometry oracle")
    torch.manual_seed(812)
    cases = 0
    for sinks, windows, heads, interleaved in (
        (0, 1, 1, False),
        (1, 63, 2, False),
        (63, 65, 2, True),
        (64, 64, 3, False),
        (65, 129, 2, True),
        (129, 0, 1, False),
    ):
        source = torch.randn(319, heads, 3, 128)
        k, v = source[:, :, 0], source[:, :, 2]
        if not interleaved:
            k, v = k.contiguous(), v.contiguous()
        index = torch.randperm(len(k))[:windows].contiguous()
        rows, blocks = sinks + windows, (sinks + windows + 63) // 64
        ko = torch.full((rows, heads, 128), float("nan"))
        vo = torch.full_like(ko, float("nan"))
        ko[:sinks].copy_(k[:sinks])
        vo[:sinks].copy_(v[:sinks])
        kc = torch.empty(1, blocks, heads, 128)
        vc = torch.empty_like(kc)
        _gather_summaries[(blocks, heads)](
            k,
            v,
            ko,
            vo,
            index,
            kc,
            vc,
            rows,
            sinks,
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            heads,
            blocks,
        )
        wantk = torch.cat((k[:sinks], k[index]))
        wantv = torch.cat((v[:sinks], v[index]))
        torch.testing.assert_close(ko, wantk, atol=0, rtol=0)
        torch.testing.assert_close(vo, wantv, atol=0, rtol=0)
        rk, rv = torch.empty_like(kc), torch.empty_like(vc)
        _reduce_kv_kernel.fn[(1, blocks, heads)](
            TensorDescriptor.from_tensor(wantk.unsqueeze(0), [1, 64, 1, 128]),
            TensorDescriptor.from_tensor(wantv.unsqueeze(0), [1, 64, 1, 128]),
            rk,
            rv,
            rows,
            heads,
            blocks,
            128,
            64,
            128,
        )
        torch.testing.assert_close(kc, rk, atol=0, rtol=0)
        torch.testing.assert_close(vc, rv, atol=0, rtol=0)
        cases += 1
    print(f"PASS: {cases} exhaustive FP32 interpreter cases; GPU BF16 unqualified")


if __name__ == "__main__":
    main()
