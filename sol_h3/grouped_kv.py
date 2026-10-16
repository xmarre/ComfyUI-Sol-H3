"""Synchronous, invocation-owned grouped K/V production for cooperating VDN.

The provider executes the consumer before returning. Scratch and its summaries
must never escape that call: VDN overwrites the same scratch for the next group.
"""
from contextvars import ContextVar

import torch

_ACTIVE = ContextVar("sol_h3_grouped_kv", default=None)


def current(k, v):
    from .runtime import _REQUEST, _FORWARD
    scoped = _ACTIVE.get()
    if scoped is None:
        return None
    prepared, request, forward = scoped
    if request is not _REQUEST.get() or forward is not _FORWARD.get():
        return None
    if prepared is None or not k.squeeze(0).is_set_to(prepared.k.squeeze(0)) or not v.squeeze(0).is_set_to(prepared.v.squeeze(0)):
        return None
    prepared.check(k, v)
    return prepared



def _can_fuse(state, key, value, k_out, v_out, window_index, transformer_options, force_dense):
    eligible = (
        not force_dense and state is not None and key.device.type == "cuda"
        and key.dtype == torch.bfloat16 and key.shape[-1] == 128
        and all(t.stride(-1) == 1 for t in (key, value))
        and k_out.is_contiguous() and v_out.is_contiguous()
        and window_index.is_contiguous() and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )
    if eligible:
        from .interop import dense_evaluation_warmup
        from .runtime import _FORWARD
        active = _FORWARD.get()
        eligible = (
            active is not None and state.config.backend == "sol"
            and not dense_evaluation_warmup(state.config, active[2], transformer_options or {})
            and torch.cuda.get_device_capability(key.device) in ((12, 0), (12, 1))
            and not torch.cuda.is_current_stream_capturing()
        )
    return eligible

def execute(key, value, k_out, v_out, window_index, sink_rows, consumer, *, transformer_options=None, force_dense=False):
    """Gather one restricted domain and synchronously consume its summaries.

    Prefix scratch has already been filled by VDN. Only window rows are written.
    Non-CUDA/unsupported layouts retain exactly VDN's original indexed gather.
    Execution errors, including compilation and OOM, propagate without retry.
    """
    if (key.ndim != 3 or key.shape != value.shape or k_out.shape != v_out.shape
            or k_out.ndim != 3 or key.shape[1:] != k_out.shape[1:]
            or type(sink_rows) is not int or not 0 <= sink_rows <= k_out.shape[0]
            or window_index.ndim != 1 or window_index.dtype != torch.int64
            or window_index.numel() + sink_rows != k_out.shape[0]
            or any(t.device != key.device for t in (value, k_out, v_out, window_index))
            or any(t.dtype != key.dtype for t in (value, k_out, v_out))):
        raise RuntimeError("Sol grouped K/V geometry/dtype/device mismatch")
    from .runtime import _REQUEST
    state = _REQUEST.get()
    eligible = _can_fuse(state, key, value, k_out, v_out, window_index,
                         transformer_options, force_dense)
    if not eligible:
        torch.index_select(key, 0, window_index, out=k_out[sink_rows:])
        torch.index_select(value, 0, window_index, out=v_out[sink_rows:])
        return consumer(k_out, v_out)
    from .runtime import _FORWARD
    prepared = _produce(key, value, k_out, v_out, window_index, sink_rows)
    token = _ACTIVE.set((prepared, state, _FORWARD.get()))
    state.grouped_kv_fused_calls += 1
    state.grouped_kv_fused_rows += k_out.shape[0]
    try:
        return consumer(k_out, v_out)
    finally:
        _ACTIVE.reset(token)


def _produce(key, value, k_out, v_out, window_index, sink_rows):
    from .grouped_kv_triton import gather_summaries
    from ._vendor.sol_attn.interface import PreparedKV
    from .partitioned_request import _verify_partitioned_source
    _verify_partitioned_source()
    kc, vc = gather_summaries(key, value, k_out, v_out, window_index, sink_rows)
    return PreparedKV(k_out.unsqueeze(0), v_out.unsqueeze(0), summaries=(kc, vc))
