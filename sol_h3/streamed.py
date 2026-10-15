"""Bounded native H3 projection with full-precision BF16 K/V retrieval.

One invocation owns its K/V and preprocessing. No activation is cached on a
model or Request. Q is recomputed with the original projection module after all
K/V rows are available; there is never a full-sequence Q allocation.
"""
from contextvars import ContextVar
from dataclasses import dataclass
import sys

import torch

from .contracts import prefix_length
from .native_contract import function_digest

CONTRACT = "bf16-kv-two-pass-q-v1"
_STREAM = ContextVar("sol_h3_qkv_stream", default=None)
# Audited native Attention.forward: ComfyUI 8406da9 and ba99487 respectively.
NATIVE_ATTENTION_DIGESTS = frozenset({
    "6391487f468657c0c4f6ac908957b27774d71caa4c440ae884b3d196ab3a4f1e",
    "6abffeca6fc6bf4941b59c7523916d1c8f9323dd1df6c5f5e3b6a83964e8033b",
})


@dataclass
class QueryStream:
    rows: int
    k: torch.Tensor
    v: torch.Tensor
    query_start: int = 0
    prepared_kv: object = None

    def check(self, q, k, v):
        if (k.data_ptr() != self.k.data_ptr() or v.data_ptr() != self.v.data_ptr()
                or k.shape != self.k.shape or v.shape != self.v.shape
                or k.stride() != self.k.stride() or v.stride() != self.v.stride()
                or self.query_start + q.shape[2] > self.rows):
            raise RuntimeError("Streamed H3 attention lost its invocation-owned K/V geometry")


def ineligible_reason(block, args, state, previous):
    if not state.config.qkv_chunk_size:
        return "disabled"
    x = args.get("img")
    # Leave room for the packed chunk, gate outputs and dense prefix output,
    # and amortize the extra projection pass on sufficiently long sequences.
    if not torch.is_tensor(x) or x.ndim != 2 or x.shape[0] <= 8 * state.config.qkv_chunk_size:
        return "short_sequence"
    if sys.platform == "win32":
        return "native_windows"
    if (x.device.type != "cuda" or x.dtype != torch.bfloat16
            or torch.cuda.get_device_capability(x.device) not in ((12, 0), (12, 1))):
        return "activation_dtype_device"
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return "autograd_or_graph_capture"
    if args.get("attention") is not None:
        return "inherited_attention"
    options = args["transformer_options"]
    # These integrations own raw Q/K/V, transforms, physical key measures, or
    # query domains. Keep their established producer and provider contracts.
    if any(options.get(key) is not None for key in (
        "vdn_h3_external_sequence_v1", "h3_flow_mixed_grid_attention_measure_v1", "attention_measure_v1",
    )) or options.get("minimax_h3_untwist_rope", {}).get("enabled"):
        return "external_sequence_or_transform"
    if hasattr(previous, "attention_preprocess_v1"):
        return "inherited_preprocess"
    try:
        prefix_length(options.get("minimax_h3_layout", args.get("layout")), x.shape[0])
    except RuntimeError:
        return "packed_layout"
    from comfy.ldm.minimax.model import Attention, DiTBlock
    from torch.nn.modules import module as hooks
    attn = getattr(block, "attn", None)
    if (type(block) is not DiTBlock or getattr(block.forward, "__func__", None) is not DiTBlock.forward
            or type(attn) is not Attention or getattr(attn.forward, "__func__", None) is not Attention.forward):
        return "block_or_attention_owner"
    if attn.head_dim != 128 or args.get("rope_freqs") is None:
        return "head_or_rope_geometry"
    for module in (block, attn, attn.qkv_proj):
        if (module._forward_hooks or module._forward_pre_hooks
                or getattr(module, "_compiled_call_impl", None) is not None):
            return "module_hooks_or_compile"
    if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
        return "global_hooks"
    if getattr(attn.qkv_proj, "quant_format", None) not in (None, "int8_tensorwise"):
        return "projection_quantization"
    if any(getattr(attn.qkv_proj, name, None) for name in (
        "weight_function", "bias_function", "weight_lowvram_function", "bias_lowvram_function",
    )):
        # A runtime adapter can recast/dequantize and rebuild the entire weight
        # on each module call. Repeating that per chunk is an unbounded latency
        # regression, and a stateful hook need not produce the same Q/K/V twice.
        return "runtime_projection_patches"
    # Reject monkey-patched/class-replaced producers before any projection.
    producer = getattr(attn.qkv_proj.forward, "__func__", None)
    if producer is None or producer.__module__ not in ("comfy.ops", "torch.nn.modules.linear"):
        return "projection_owner"
    checked = state.streamed_native_verified
    if Attention.forward not in checked:
        try:
            checked[Attention.forward] = function_digest(Attention.forward) in NATIVE_ATTENTION_DIGESTS
        except (OSError, TypeError):
            checked[Attention.forward] = False
    if not checked[Attention.forward]:
        return "native_attention_source"
    return None


def project_chunk(attn, x, rope_freqs, start, stop):
    """Use the native projection/cast/adapter owner and native in-place norm/RoPE."""
    import comfy.model_management
    import comfy.quant_ops

    rows, heads, dim = stop - start, attn.heads, attn.head_dim
    q, k, v = attn.qkv_proj(x[start:stop]).split(heads * dim, dim=-1)
    q, k, v = (part.view(1, rows, heads, dim) for part in (q, k, v))
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    comfy.quant_ops.ck.rms_rope_split_half_(
        q, k, rope_freqs[:, start:stop], qw, kw,
        epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2,
    )
    return q, k, v


def attention(attn, x, rope_freqs, transformer_options, chunk_size, *, project=project_chunk, attend=None):
    """Two projection passes; only the current packed projection chunk survives.

    ``project``/``attend`` are injectable for deterministic CPU lifetime tests.
    Production always uses the native module, norm/RoPE and attention dispatch.
    """
    if type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 64:
        raise ValueError("Streamed H3 chunk size must be a positive multiple of 64")
    rows, heads, dim = x.shape[0], attn.heads, attn.head_dim
    from .memory_profile import mark
    k = x.new_empty((1, rows, heads, dim))
    v = torch.empty_like(k)
    mark("streamed_kv_allocated", rows=rows, chunk_size=chunk_size)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        qc, kc, vc = project(attn, x, rope_freqs, start, stop)
        mark("streamed_kv_projection", start=start, stop=stop)
        k[:, start:stop].copy_(kc)
        v[:, start:stop].copy_(vc)
        # Drop every view before the next projection, including unused Q.
        del qc, kc, vc
    k, v = k.transpose(1, 2), v.transpose(1, 2)
    output = x.new_empty((rows, heads * dim))
    mark("streamed_output_allocated", rows=rows)
    stream = QueryStream(rows, k, v)
    token = _STREAM.set(stream)
    complete = False
    q = result = None
    try:
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            q, kc, vc = project(attn, x, rope_freqs, start, stop)
            mark("streamed_q_projection", start=start, stop=stop)
            del kc, vc
            stream.query_start = start
            if attend is None:
                from comfy.ldm.minimax.model import optimized_attention
                result = optimized_attention(
                    q.transpose(1, 2), k, v, heads, mask=None, skip_reshape=True,
                    transformer_options=transformer_options,
                )
            else:
                result = attend(q.transpose(1, 2), k, v, heads, transformer_options)
            output[start:stop].copy_(result.reshape(stop - start, heads * dim))
            mark("streamed_attention_done", start=start, stop=stop)
            q = result = None
        complete = True
    finally:
        _STREAM.reset(token)
        # Explicitly clear references even when an exception traceback retains
        # the stream object. CUDA consumers execute on the allocating stream.
        stream.prepared_kv = None
        stream.k = stream.v = None
        del q, result, k, v, stream
        if not complete:
            output = None
    mark("streamed_kv_released", rows=rows)
    result = attn.out_proj(output)
    mark("streamed_out_projection", rows=rows)
    return result
