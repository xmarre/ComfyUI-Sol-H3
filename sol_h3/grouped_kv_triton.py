"""One-pass restricted-domain K/V gather plus unchanged M64 Sol summaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_summaries(
    K,
    V,
    KO,
    VO,
    Index,
    KC,
    VC,
    Rows,
    Sinks,
    KRow: tl.constexpr,
    KHead: tl.constexpr,
    VRow: tl.constexpr,
    VHead: tl.constexpr,
    Heads: tl.constexpr,
    Blocks: tl.constexpr,
):
    block, head = tl.program_id(0), tl.program_id(1)
    rows = block * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 128)
    window = (rows >= Sinks) & (rows < Rows)
    source = tl.load(Index + rows - Sinks, mask=window, other=0)
    k = tl.load(K + source[:, None] * KRow + head * KHead + dims[None, :], mask=window[:, None], other=0)
    v = tl.load(V + source[:, None] * VRow + head * VHead + dims[None, :], mask=window[:, None], other=0)
    dst = (rows[:, None] * Heads + head) * 128 + dims[None, :]
    # Sink rows are already resident and never rewritten. The tile straddling
    # sink/window consumes both parts in their original restricted-domain order.
    kp = tl.load(KO + dst, mask=((rows < Sinks) & (rows < Rows))[:, None], other=0)
    vp = tl.load(VO + dst, mask=((rows < Sinks) & (rows < Rows))[:, None], other=0)
    tl.store(KO + dst, k, mask=window[:, None])
    tl.store(VO + dst, v, mask=window[:, None])
    k = tl.where((rows < Sinks)[:, None], kp, k)
    v = tl.where((rows < Sinks)[:, None], vp, v)
    count = tl.minimum(64, Rows - block * 64)
    summary = (block * Heads + head) * 128 + dims
    tl.store(KC + summary, tl.sum(k.to(tl.float32), axis=0) / count)
    tl.store(VC + summary, tl.sum(v.to(tl.float32), axis=0))


def gather_summaries(key, value, k_out, v_out, window_index, sink_rows):
    rows, heads, dim = k_out.shape
    blocks = triton.cdiv(rows, 64)
    kc = torch.empty((1, blocks, heads, dim), dtype=key.dtype, device=key.device)
    vc = torch.empty_like(kc)
    with torch.cuda.device(key.device):
        _gather_summaries[(blocks, heads)](
            key,
            value,
            k_out,
            v_out,
            window_index,
            kc,
            vc,
            rows,
            sink_rows,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            heads,
            blocks,
            num_warps=4,
            num_stages=1,
        )
    return kc, vc
