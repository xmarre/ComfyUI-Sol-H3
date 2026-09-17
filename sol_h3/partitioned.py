"""Partitioned exact-prefix primitives for the SM120 Sol backend.

The packaged SM120 kernel already computes a natural-log softmax normalizer for
every query row.  The ordinary public Sol-Attn interface intentionally discards
that tensor.  Exact-prefix progressive attention needs the normalizer so
independently evaluated physical key domains can be recombined without changing
softmax semantics.

This module does not change the ordinary Sol-Attn API or routing.  It exposes a
narrow SM120-only primitive and a numerically stable LSE merge for the new Flow
partition contract.  Unsupported devices or execution modes fail closed.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

PARTITIONED_SM120_ABI = "sol-h3-partitioned-output-lse-v1"


def merge_lse_partitions(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
    log_measures: Sequence[float | torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized attention outputs using natural-log normalizers."""
    if not outputs or len(outputs) != len(lses) or len(outputs) != len(log_measures):
        raise ValueError("partition outputs/LSEs/measures must be non-empty and have equal length")
    reference_output = outputs[0]
    reference_lse = lses[0]
    if reference_lse.shape != reference_output.shape[:-1]:
        raise ValueError("partition LSE must match output without the value dimension")
    acc_dtype = torch.float64 if reference_output.dtype == torch.float64 or reference_lse.dtype == torch.float64 else torch.float32
    adjusted = []
    for index, (output, lse, measure) in enumerate(zip(outputs, lses, log_measures)):
        if output.shape != reference_output.shape or output.dtype != reference_output.dtype or output.device != reference_output.device:
            raise ValueError(f"partition output {index} does not match the first output")
        if lse.shape != reference_lse.shape or lse.device != reference_lse.device:
            raise ValueError(f"partition LSE {index} does not match the first LSE")
        adjusted.append(lse.to(acc_dtype) + torch.as_tensor(measure, dtype=acc_dtype, device=lse.device))
    merged_lse = torch.logsumexp(torch.stack(adjusted, dim=0), dim=0)
    merged = torch.zeros_like(reference_output, dtype=acc_dtype)
    for output, partition_lse in zip(outputs, adjusted):
        merged.add_(
            output.to(acc_dtype)
            * torch.exp(partition_lse - merged_lse).unsqueeze(-1)
        )
    return merged.to(reference_output.dtype), merged_lse


def _validate_partition_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if (
        any(t.ndim != 4 for t in (q, k, v))
        or k.shape != v.shape
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
        or q.shape[0] != 1
        or q.shape[1] <= 0
        or k.shape[1] <= 0
        or q.shape[-1] != 128
    ):
        raise ValueError("partitioned Sol requires Q [1,Tq,H,128] and KV [1,Tkv,H,128]")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("partitioned Sol requires BF16 QKV")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("partitioned Sol requires QKV on the same CUDA device")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("partitioned Sol currently requires SM120")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("partitioned Sol requires a contiguous head dimension")
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("partitioned Sol is unavailable under autograd, torch.compile, or CUDA graph capture")


def sm120_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float = 1.0,
    sink_start: int = 0,
    sink_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one rectangular SM120 Sol partition and return ``(output, LSE)``.

    Inputs and output are BTHD.  LSE is float32 ``[B,Tq,H]`` in natural-log
    units, matching the packaged kernel's existing internal finalizer.
    """
    _validate_partition_inputs(q, k, v)
    tau = float(tau)
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("partitioned Sol tau must be finite and non-negative")
    if type(sink_start) is not int or type(sink_tokens) is not int:
        raise TypeError("partitioned Sol sink geometry must use integers")
    if sink_start < 0 or sink_tokens < 0 or sink_start + sink_tokens > k.shape[1]:
        raise ValueError("partitioned Sol sink range is outside the K/V domain")

    from ._vendor.sol_attn import interface as interface
    from ._vendor.sol_attn.preprocess import prepare

    arch = tuple(torch.cuda.get_device_capability(q.device))
    if interface._backend_for_arch(arch) != "cute_sm120":
        raise RuntimeError("partitioned Sol requires the packaged cute_sm120 backend")
    interface._validate_cute(arch, k.shape[1], 1)
    scale = q.shape[-1] ** -0.5
    batch, q_rows, heads, _ = q.shape
    kv_rows = k.shape[1]

    with torch.cuda.device(q.device):
        kc, vc, threshold = prepare(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            thresh_type="diag",
            valid_tokens=q_rows,
            valid_kv_tokens=kv_rows,
        )
        output = torch.empty_like(q)
        lse = torch.empty((batch, q_rows, heads), device=q.device, dtype=torch.float32)
        stream = interface._stream(q.device)
        sink_start_block, sink_end_block = interface._sink_block_range(
            kv_rows,
            sink_start,
            sink_tokens,
        )
        # Optional SM120 arguments are disabled by reusing the already-live
        # threshold tensor, exactly as the ordinary interface does.  The compile
        # key contains only structural shape/layout identity; runtime values such
        # as tau and LSE contents do not create specializations.
        tensors = [
            q,
            k,
            v,
            output,
            kc,
            vc,
            threshold,
            threshold,
            threshold,
            lse,
        ]
        layout_key = tuple(tuple(int(s) for s in tensor.stride()) for tensor in (q, k, v))
        key = (
            PARTITIONED_SM120_ABI,
            q.device.index,
            arch,
            batch,
            q_rows,
            kv_rows,
            heads,
            layout_key,
        )
        compiled = interface._compiled.get(key)
        if compiled is None:
            with interface._compile_lock:
                compiled = interface._compiled.get(key)
                if compiled is None:
                    compiled, args = interface._compile_sm120(
                        key,
                        tensors,
                        scale,
                        sink_start_block,
                        sink_end_block,
                        stream,
                        False,
                        False,
                    )
                else:
                    args = interface._to_cute_tensors(tensors)
        else:
            args = interface._to_cute_tensors(tensors)
        compiled(
            *args,
            scale,
            sink_start_block,
            sink_end_block,
            stream=stream,
        )
    return output, lse


def partitioned_sm120_attention(
    q: torch.Tensor,
    key_partitions: Sequence[torch.Tensor],
    value_partitions: Sequence[torch.Tensor],
    log_measures: Sequence[float],
    *,
    tau: float = 1.0,
    sink_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate and merge explicit physical K/V domains for one Q partition."""
    if not key_partitions or len(key_partitions) != len(value_partitions) or len(key_partitions) != len(log_measures):
        raise ValueError("partitioned Sol requires matching non-empty K/V/measure partitions")
    if sink_ranges is None:
        sink_ranges = tuple((0, 0) for _ in key_partitions)
    if len(sink_ranges) != len(key_partitions):
        raise ValueError("partitioned Sol requires one sink range per K/V partition")
    outputs = []
    lses = []
    for k, v, (sink_start, sink_tokens) in zip(key_partitions, value_partitions, sink_ranges):
        output, lse = sm120_attention_with_lse(
            q,
            k,
            v,
            tau=tau,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        outputs.append(output)
        lses.append(lse)
    return merge_lse_partitions(outputs, lses, log_measures)


__all__ = [
    "PARTITIONED_SM120_ABI",
    "merge_lse_partitions",
    "partitioned_sm120_attention",
    "sm120_attention_with_lse",
]
