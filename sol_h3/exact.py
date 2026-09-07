import torch


def native_affine(h, shift, scale, row):
    return h.mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))


def affine(h, shift, scale, segments, verified):
    from .kernels import affine_kernel
    if h.device.type != "cuda" or h.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError("Exact Sol-H3 fusion requires CUDA BF16/FP16/FP32 activations")
    if h.ndim != 2 or h.stride(1) != 1 or shift.ndim != 2 or scale.shape != shift.shape:
        raise RuntimeError("Unsupported affine tensor layout")
    if shift.shape[1] != h.shape[1] or shift.device != h.device or scale.device != h.device:
        raise RuntimeError("Affine modulation shape/device mismatch")
    cursor = 0
    for a, b, row in segments:
        if a != cursor or not a <= b <= len(h):
            raise RuntimeError("Modulation segments must cover hidden rows contiguously")
        cursor = b
        indexed = torch.is_tensor(row)
        if indexed:
            if row.device != h.device or row.dtype != torch.int64 or row.shape != (b - a,):
                raise RuntimeError("Per-token modulation indices must be CUDA int64 [segment rows]")
            row = row.contiguous()
            torch._assert_async(((row >= 0) & (row < len(shift))).all(), "Modulation index out of range")
        elif type(row) is not int or not 0 <= row < len(shift):
            raise RuntimeError("Modulation row out of range")
        if b == a:
            continue
        key = (h.device, h.dtype, shift.dtype, scale.dtype, h.shape[1], indexed)
        want = None
        if key not in verified:
            probe_row = row[:32] if indexed else row
            want = native_affine(h[a:min(b, a + 32)].clone(), shift, scale, probe_row)
        affine_kernel[((b - a) * h.shape[1] + 255) // 256,](
            h[a:b], shift, scale, row if indexed else h,
            b - a, h.shape[1], h.stride(0), *shift.stride(), *scale.stride(),
            0 if indexed else row, indexed, 256, enable_fp_fusion=False)
        if want is not None:
            if not torch.equal(h[a:min(b, a + 32)], want):
                raise RuntimeError("Exact affine real-activation parity gate failed; disable exact fusion")
            verified.add(key)
    if cursor != len(h):
        raise RuntimeError("Modulation segments do not cover all hidden rows")
    return h


def execute_block(block, args, verified):
    # Keep native RMSNorm, AdaLN, attention, MLP and addcmul_ gate operations.
    from comfy.ldm.minimax.model import DiTBlock, _mod_gate
    if type(block) is not DiTBlock or getattr(block.forward, "__func__", None) is not DiTBlock.forward:
        raise RuntimeError("Exact fusion requires the native DiTBlock forward; another provider owns this block")
    if block._forward_hooks or block._forward_pre_hooks:
        raise RuntimeError("Exact fusion cannot bypass block-level hooks; disable exact fusion")
    x, segments = args["img"], args["mod_segments"]
    sm, cm, gm, sf, cf, gf = block.adaln_proj(args["t_emb"])
    h = affine(block.norm1(x), sm, cm, segments, verified)
    attention = args.get("attention") or block.attn
    x = _mod_gate(x, gm, attention(h, rope_freqs=args["rope_freqs"],
                                  transformer_options=args["transformer_options"]), segments)
    h = affine(block.norm2(x), sf, cf, segments, verified)
    return {"img": _mod_gate(x, gf, block.mlp(h), segments)}
