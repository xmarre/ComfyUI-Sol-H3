"""Bounded one-shot receipts for the 00625 same-input runtime divergence.

This module is diagnostic-only. Production results are never replaced and no
RNG state is consumed. It launches no diagnostic attention/model computation.

CUDA capture deliberately does *not* copy samples to the host at the numerical
boundary being observed. A tiny deterministic gather is queued on the current
stream and the sampled CUDA tensors are retained in a private pending structure.
Host transfer, hashing, reductions, and JSON materialization happen only when the
outer sampling request has completed. This avoids inserting a device
synchronization into the first-low model path.
"""
from __future__ import annotations

import hashlib
import json
import logging

import torch


SCHEMA = "sol_h3_00625_runtime_state_v2"
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



def _sdpa_backend_name(value):
    try:
        from torch.nn.attention import SDPBackend
    except (ImportError, AttributeError):
        return None
    for name in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        backend = getattr(SDPBackend, name, None)
        if backend is not None and getattr(backend, "value", None) == value:
            return name.lower()
    return None


def sdpa_capability_receipt(q, k, v, *, scale=None):
    """Describe and resolve the backend for VDN's exact [1,H,Q,D] views.

    The choice query is executed under the exact priority object exposed by the
    installed ComfyUI runtime when available. It does not execute an attention
    kernel. Global SDPA/matmul flags are recorded because they are process state,
    not workflow state, and can otherwise make a same-workflow replay select a
    different numerical path.
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
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "scale": None if scale is None else float(scale),
    }

    for key, attribute in (
        ("flash_sdp_enabled", "flash_sdp_enabled"),
        ("cudnn_sdp_enabled", "cudnn_sdp_enabled"),
        ("efficient_sdp_enabled", "mem_efficient_sdp_enabled"),
        ("math_sdp_enabled", "math_sdp_enabled"),
        ("math_sdp_low_precision_reduction", "fp16_bf16_reduction_math_sdp_allowed"),
    ):
        getter = getattr(torch.backends.cuda, attribute, None)
        if not callable(getter):
            receipt[key] = None
            continue
        try:
            receipt[key] = bool(getter())
        except Exception as exc:
            receipt[key] = f"{type(exc).__name__}: {exc}"

    matmul = getattr(torch.backends.cuda, "matmul", None)
    if matmul is not None:
        for key, attribute in (
            ("matmul_allow_fp16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction"),
            ("matmul_allow_bf16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"),
            ("matmul_allow_fp16_accumulation", "allow_fp16_accumulation"),
        ):
            try:
                receipt[key] = bool(getattr(matmul, attribute))
            except Exception as exc:
                receipt[key] = f"{type(exc).__name__}: {exc}"

    try:
        params = torch.backends.cuda.SDPAParams(qh, kh, vh, None, 0.0, False, False)
    except Exception as exc:
        receipt["params_error"] = f"{type(exc).__name__}: {exc}"
        return receipt
    for name, attribute in (
        ("flash", "can_use_flash_attention"),
        ("cudnn", "can_use_cudnn_attention"),
        ("efficient", "can_use_efficient_attention"),
    ):
        predicate = getattr(torch.backends.cuda, attribute, None)
        if not callable(predicate):
            receipt[name] = None
            continue
        try:
            receipt[name] = bool(predicate(params))
        except Exception as exc:
            receipt[name] = f"{type(exc).__name__}: {exc}"

    choice = getattr(torch, "_fused_sdp_choice", None)
    if callable(choice):
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            fallback_priority = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
            try:
                import comfy.ops as comfy_ops
                runtime_priority = getattr(comfy_ops, "SDPA_BACKEND_PRIORITY", None)
            except Exception as exc:
                runtime_priority = None
                receipt["comfy_priority_import_error"] = f"{type(exc).__name__}: {exc}"

            if isinstance(runtime_priority, (list, tuple)) and runtime_priority:
                priority = list(runtime_priority)
                receipt["priority_source"] = "comfy.ops.SDPA_BACKEND_PRIORITY"
            else:
                priority = fallback_priority
                receipt["priority_source"] = "diagnostic_fallback_matches_comfy_v0.37.0"

            receipt["priority"] = [
                _sdpa_backend_name(int(getattr(item, "value", -1))) or str(item)
                for item in priority
            ]
            with sdpa_kernel(priority, set_priority=True):
                value = int(choice(qh, kh, vh, scale=scale))
            receipt["selected_backend_value"] = value
            receipt["selected_backend"] = _sdpa_backend_name(value)
        except Exception as exc:
            receipt["selected_backend_error"] = f"{type(exc).__name__}: {exc}"
    return receipt

def observe_native_once(
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
    """Observe one eval0/block0 VDN native route after its production result exists.

    Global/local/anchor each get one production Q/K/V/output receipt. The
    diagnostic performs no replay and never substitutes production output.
    """
    key = f"eval0_block0_vdn_{kind}_native"
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
        and kind in {"global", "local", "anchor"}
        and _sm120_tensor(q)
        and _sm120_tensor(k)
        and _sm120_tensor(v)
    )
    if not eligible:
        return native()

    # Production executes before any diagnostic CUDA gather.
    result = native()
    payload = {
        "evaluation": int(evaluation),
        "block_index": int(block_index),
        "kind": kind,
        "route": route,
        "sink_rows": int(sink_rows),
        "q": deferred_tensor_receipt(q),
        "k": deferred_tensor_receipt(k),
        "v": deferred_tensor_receipt(v),
        "output": deferred_tensor_receipt(result),
        "sdpa_capabilities": sdpa_capability_receipt(q, k, v, scale=scale),
        "extra_attention_calls": 0,
        "extra_transformer_nfe": 0,
    }
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
