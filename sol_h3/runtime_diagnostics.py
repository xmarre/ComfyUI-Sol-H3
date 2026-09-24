"""Bounded one-shot receipts for the 00625 same-input runtime divergence.

This module is diagnostic-only. Production results are never replaced by a
shadow result and no RNG state is consumed.

CUDA capture deliberately does *not* copy samples to the host at the numerical
boundary being observed. A tiny deterministic gather is queued on the current
stream and the sampled CUDA tensors are retained in a private pending structure.
Host transfer, hashing, reductions, and JSON materialization happen only when the
outer sampling request has completed. This avoids inserting a device
synchronization into the first-low model path.

One first-low grouped-SDPA shadow replay is still executed intentionally. It is
never substituted into production output and it is not an extra transformer NFE,
but it can affect timing/cache state after the production attention result has
already been computed. The receipt reports that fact explicitly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import torch


SCHEMA = "sol_h3_00625_runtime_state_v3"
_SAMPLE_COUNT = 256
_DEFERRED = "_sol_h3_deferred_tensor_receipt"
log = logging.getLogger("comfy.sol_h3")


def _sm120_tensor(tensor) -> bool:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        return False
    try:
        return tuple(torch.cuda.get_device_capability(tensor.device)) == (12, 0)
    except Exception:
        return False


def _metadata(tensor: torch.Tensor, sample_count: int) -> tuple[dict, int]:
    detached = tensor.detach()
    count = min(max(int(sample_count), 1), detached.numel()) if detached.numel() else 0
    return {
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "sample_count": int(count),
    }, count


def tensor_metadata_receipt(tensor):
    """Host-only tensor metadata. This launches no CUDA work and performs no sync."""
    if not isinstance(tensor, torch.Tensor):
        return None
    detached = tensor.detach()
    pointer = int(detached.data_ptr())
    return {
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "storage_offset": int(detached.storage_offset()),
        "data_ptr": pointer,
        "data_ptr_mod_256": pointer % 256,
        "data_ptr_mod_4096": pointer % 4096,
    }


def _sample_indices(numel: int, count: int, device) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == 1 or numel <= 1:
        return torch.zeros(count, dtype=torch.long, device=device)
    # Integer arithmetic gives the same deterministic rounded linspace without a
    # CPU->GPU index transfer or a host synchronization.
    positions = torch.arange(count, dtype=torch.long, device=device)
    return (positions * (numel - 1) + (count - 1) // 2) // (count - 1)


def _digest_sample(metadata: dict, sample: torch.Tensor) -> dict:
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if sample.numel() == 0:
        return {
            **metadata,
            "sample_sha256": digest.hexdigest(),
            "sample_finite": True,
        }
    sample = sample.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest.update(bytes(sample.view(torch.uint8).tolist()))
    finite = bool(torch.isfinite(sample).all().item())
    return {
        **metadata,
        "sample_sha256": digest.hexdigest(),
        "sample_finite": finite,
        "sample_abs_max": float(sample.abs().max().item()),
        "sample_mean": float(sample.mean().item()),
    }


def tensor_receipt(tensor, *, sample_count=_SAMPLE_COUNT):
    """Return an immediate bounded deterministic receipt.

    This helper is primarily for unit tests and post-execution inspection. CUDA
    runtime instrumentation uses :func:`deferred_tensor_receipt` instead.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    metadata, count = _metadata(tensor, sample_count)
    if count == 0:
        sample = torch.empty(0, dtype=torch.float32)
    else:
        detached = tensor.detach()
        indices = _sample_indices(detached.numel(), count, detached.device)
        sample = torch.take(detached, indices)
    return _digest_sample(metadata, sample)


def deferred_tensor_receipt(tensor, *, sample_count=_SAMPLE_COUNT):
    """Queue only the bounded sample gather; defer every host-visible operation."""
    if not isinstance(tensor, torch.Tensor):
        return None
    detached = tensor.detach()
    metadata, count = _metadata(detached, sample_count)
    if count == 0:
        sample = torch.empty(0, dtype=torch.float32, device=detached.device)
    else:
        indices = _sample_indices(detached.numel(), count, detached.device)
        sample = torch.take(detached, indices).to(dtype=torch.float32)
    return {
        _DEFERRED: True,
        "metadata": metadata,
        "sample": sample,
    }


def _finalize_value(value):
    if isinstance(value, dict):
        if value.get(_DEFERRED) is True:
            return _digest_sample(value["metadata"], value["sample"])
        return {key: _finalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_finalize_value(item) for item in value]
    return value


def finalize(state):
    """Materialize pending receipts only after the outer sample has completed."""
    pending = getattr(state, "runtime_diagnostic_pending", None)
    completed = getattr(state, "runtime_diagnostic", None)
    if not isinstance(pending, dict) or not isinstance(completed, dict):
        return
    for key, payload in list(pending.items()):
        entry = _finalize_value(payload)
        if key == "eval0_block0_first_vdn_native_attention":
            first = entry.get("output") or {}
            replay = entry.get("shadow_replay_output") or {}
            if first.get("sample_sha256") and replay.get("sample_sha256"):
                entry["shadow_replay_sample_equal"] = (
                    first["sample_sha256"] == replay["sample_sha256"]
                )
        completed[key] = entry
        log.warning(
            "Sol-H3 runtime-state diagnostic %s",
            json.dumps({"key": key, **entry}, sort_keys=True),
        )
    pending.clear()


def _store_pending_once(state, key, payload) -> bool:
    pending = getattr(state, "runtime_diagnostic_pending", None)
    completed = getattr(state, "runtime_diagnostic", None)
    if (
        not isinstance(pending, dict)
        or not isinstance(completed, dict)
        or key in pending
        or key in completed
    ):
        return False
    pending[key] = {"schema": SCHEMA, **payload}
    return True


def capture_metadata(state, key, tensor, **metadata):
    receipt = tensor_metadata_receipt(tensor)
    if receipt is None:
        return False
    return _store_pending_once(
        state,
        key,
        {
            **metadata,
            "tensor_metadata": receipt,
            "cuda_work_before_observed_boundary": False,
        },
    )


def capture_stage(state, key, tensor, **metadata):
    if not _sm120_tensor(tensor):
        return False
    return _store_pending_once(
        state,
        key,
        {
            **metadata,
            "tensor": deferred_tensor_receipt(tensor),
        },
    )



def sdpa_capability_receipt(q, k, v):
    """Describe PyTorch SDPA eligibility for VDN's exact [1,H,Q,D] views.

    This only asks backend capability predicates; it does not execute an
    additional attention kernel.
    """
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return None
    qh = q.permute(1, 0, 2).unsqueeze(0)
    kh = k.permute(1, 0, 2).unsqueeze(0)
    vh = v.permute(1, 0, 2).unsqueeze(0)
    receipt = {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "pytorch_cuda_alloc_conf": (
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            or os.environ.get("PYTORCH_ALLOC_CONF")
        ),
    }
    for name in (
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "cudnn_sdp_enabled",
    ):
        getter = getattr(torch.backends.cuda, name, None)
        if callable(getter):
            try:
                receipt[name] = bool(getter())
            except Exception as exc:
                receipt[name] = f"{type(exc).__name__}: {exc}"
    try:
        receipt["current_stream"] = int(torch.cuda.current_stream(q.device).cuda_stream)
    except Exception as exc:
        receipt["current_stream"] = f"{type(exc).__name__}: {exc}"

    try:
        import comfy.ops as comfy_ops
        priority = getattr(comfy_ops, "SDPA_BACKEND_PRIORITY", None)
        if priority is None:
            receipt["comfy_ops_priority"] = None
        else:
            receipt["comfy_ops_priority"] = [
                getattr(item, "name", str(item)) for item in priority
            ]
    except Exception as exc:
        receipt["comfy_ops_priority"] = f"{type(exc).__name__}: {exc}"
    try:
        params = torch.backends.cuda.SDPAParams(qh, kh, vh, None, 0.0, False, False)
    except Exception as exc:
        receipt["params_error"] = f"{type(exc).__name__}: {exc}"
        return receipt
    eligibility = {}
    for name, attribute in (
        ("FLASH_ATTENTION", "can_use_flash_attention"),
        ("CUDNN_ATTENTION", "can_use_cudnn_attention"),
        ("EFFICIENT_ATTENTION", "can_use_efficient_attention"),
    ):
        predicate = getattr(torch.backends.cuda, attribute, None)
        if not callable(predicate):
            eligibility[name] = None
            continue
        try:
            eligibility[name] = bool(predicate(params))
        except Exception as exc:
            eligibility[name] = f"{type(exc).__name__}: {exc}"
    receipt["eligibility"] = eligibility

    priority = receipt.get("comfy_ops_priority")
    if isinstance(priority, list):
        first = None
        for backend in priority:
            if backend == "MATH":
                first = "MATH"
                break
            if eligibility.get(backend) is True:
                first = backend
                break
        receipt["priority_first_eligible"] = first
    return receipt

def replay_native_once(
    state,
    native,
    q,
    k,
    v,
    *,
    evaluation,
    block_index,
    kind,
    route,
    sink_rows,
    scale,
):
    """Run one bounded replay of the earliest eval-0/block-0 native VDN attention call.

    The first eligible VDN warmup call wins regardless of global/local/anchor kind,
    so this brackets the earliest attention operation that can change block-0
    numerics. The production native call executes first and its result is always returned.
    Q/K/V and production-output sampling is queued only after that result exists,
    so the production native call itself is not preceded by diagnostic CUDA work.
    The second native call is a deliberate shadow replay on the same Q/K/V; only
    its bounded receipt is retained.
    """
    key = "eval0_block0_first_vdn_native_attention"
    pending = getattr(state, "runtime_diagnostic_pending", None)
    completed = getattr(state, "runtime_diagnostic", None)
    eligible = (
        bool(getattr(state, "runtime_diagnostic_enabled", False))
        and isinstance(pending, dict)
        and isinstance(completed, dict)
        and key not in pending
        and key not in completed
        and evaluation == 0
        and block_index == 0
        and _sm120_tensor(q)
        and _sm120_tensor(k)
        and _sm120_tensor(v)
    )
    if not eligible:
        return native()

    # Compute the production result before any diagnostic gather/replay.
    result = native()
    payload = {
        "evaluation": int(evaluation),
        "block_index": int(block_index),
        "kind": kind,
        "route": route,
        "sink_rows": int(sink_rows),
        "scale": float(scale),
        "q_runtime": tensor_metadata_receipt(q),
        "k_runtime": tensor_metadata_receipt(k),
        "v_runtime": tensor_metadata_receipt(v),
        "output_runtime": tensor_metadata_receipt(result),
        "q": deferred_tensor_receipt(q),
        "k": deferred_tensor_receipt(k),
        "v": deferred_tensor_receipt(v),
        "output": deferred_tensor_receipt(result),
        "sdpa_capabilities": sdpa_capability_receipt(q, k, v),
        "production_call_completed_before_diagnostic_cuda_work": True,
        "shadow_replay_calls": 1,
        "shadow_replay_substituted": False,
        "extra_transformer_nfe": 0,
    }
    try:
        replay = native()
    except Exception as exc:
        payload["shadow_replay_error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload["shadow_replay_output"] = deferred_tensor_receipt(replay)
        del replay
    _store_pending_once(state, key, payload)
    return result



def capture_sparse_runtime(state, q, k, v, output, *, context, gate=None):
    """Capture the first mapped-Sol boundary after production attention returns.

    Keeping this observation in the runtime/provider layer leaves the reviewed
    sparse/kernel source byte-identical. The production arithmetic gate remains
    authoritative; its already-materialized metrics are copied into the receipt.
    """
    if not getattr(state, "runtime_diagnostic_enabled", False) or not _sm120_tensor(q):
        return False
    return _store_pending_once(
        state,
        "first_sparse_runtime_boundary",
        {
            "context": dict(context) if isinstance(context, dict) else None,
            "q": deferred_tensor_receipt(q),
            "k": deferred_tensor_receipt(k),
            "v": deferred_tensor_receipt(v),
            "output": deferred_tensor_receipt(output),
            "arithmetic_gate": dict(gate) if isinstance(gate, dict) else None,
        },
    )
