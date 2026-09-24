"""Bounded one-shot receipts for the 00625 same-input runtime divergence.

This module is diagnostic-only. Production results are never replaced by a
shadow result and no RNG state is consumed.

Each observed production attention call executes before any diagnostic CUDA
gather. Bounded samples are then gathered on-device and retained privately;
host transfer, hashing, reductions, and JSON serialization are deferred until the
outer sampling request has completed. No attention call is replayed and no
diagnostic result is substituted into production output.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import torch


SCHEMA = "sol_h3_00625_first_global_repeatability_v1"
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


def deferred_tensor_delta_receipt(lhs, rhs, *, sample_count=_SAMPLE_COUNT):
    """Queue a bounded sample of lhs-rhs without materializing a full delta tensor."""
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return None
    if lhs.shape != rhs.shape or lhs.device != rhs.device:
        raise ValueError("repeatability delta requires matching tensor shape/device")
    left = lhs.detach()
    right = rhs.detach()
    metadata, count = _metadata(left, sample_count)
    metadata = {
        **metadata,
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "receipt_kind": "bounded_pair_delta",
    }
    if count == 0:
        sample = torch.empty(0, dtype=torch.float32, device=left.device)
    else:
        indices = _sample_indices(left.numel(), count, left.device)
        sample = torch.take(left, indices).float() - torch.take(right, indices).float()
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



def _sdpa_backend_name(value, backend_type):
    for name in (
        "FLASH_ATTENTION",
        "CUDNN_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
        "OVERRIDEABLE",
        "ERROR",
    ):
        member = getattr(backend_type, name, None)
        if member is not None and int(member.value) == int(value):
            return name
    return f"UNKNOWN_{int(value)}"


def sdpa_capability_receipt(q, k, v, scale):
    """Describe the exact native SDPA dispatch after production attention returned.

    The fused-choice query runs the dispatcher selector only; it does not execute
    another attention kernel.
    """
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return None
    qh = q.permute(1, 0, 2).unsqueeze(0)
    kh = k.permute(1, 0, 2).unsqueeze(0)
    vh = v.permute(1, 0, 2).unsqueeze(0)
    receipt = {
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", None)),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(q.device),
        "device_capability": list(torch.cuda.get_device_capability(q.device)),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "pytorch_cuda_alloc_conf": (
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            or os.environ.get("PYTORCH_ALLOC_CONF")
        ),
        "torch_cudnn_sdpa_deprioritized": os.environ.get("TORCH_CUDNN_SDPA_DEPRIORITIZED"),
        "cudnn_rescale_threshold": os.environ.get("CUDNN_RESCALE_THRESHOLD"),
        "cudnn_use_ex2_emulation": os.environ.get("CUDNN_USE_EX2_EMULATION"),
    }
    matmul = torch.backends.cuda.matmul
    for name in (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    ):
        if hasattr(matmul, name):
            try:
                receipt["matmul_" + name] = bool(getattr(matmul, name))
            except Exception as exc:
                receipt["matmul_" + name] = f"{type(exc).__name__}: {exc}"
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
        from torch.nn.attention import SDPBackend, sdpa_kernel
        priority = getattr(comfy_ops, "SDPA_BACKEND_PRIORITY", None)
        if priority is None:
            receipt["comfy_ops_priority"] = None
        else:
            receipt["comfy_ops_priority"] = [
                getattr(item, "name", str(item)) for item in priority
            ]
    except Exception as exc:
        receipt["comfy_ops_priority"] = f"{type(exc).__name__}: {exc}"
        receipt["selector_error"] = f"{type(exc).__name__}: {exc}"
        return receipt

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

    if priority is not None:
        try:
            with sdpa_kernel(priority, set_priority=True):
                selected = int(torch._fused_sdp_choice(
                    qh,
                    kh,
                    vh,
                    None,
                    0.0,
                    False,
                    scale=float(scale),
                    enable_gqa=False,
                ))
            receipt["selected_backend_value"] = selected
            receipt["selected_backend"] = _sdpa_backend_name(selected, SDPBackend)
        except Exception as exc:
            receipt["selector_error"] = f"{type(exc).__name__}: {exc}"
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
    """Replay only the first global native SDPA after its production result exists.

    The first production global attention call remains observer-clean: no
    diagnostic CUDA work is launched before it, and the returned production
    tensor remains authoritative. One output-neutral shadow call then executes
    with the exact same Q/K/V and native closure. All downstream model execution
    is intentionally considered observer-perturbed in this diagnostic.
    """
    if kind != "global":
        return native()
    key = f"eval0_block0_vdn_native_{kind}"
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

    result = native()
    shadow_result = native()
    state.runtime_diagnostic_downstream_perturbed = True
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
        "shadow_output": deferred_tensor_receipt(shadow_result),
        "shadow_minus_production": deferred_tensor_delta_receipt(shadow_result, result),
        "sdpa_dispatch": sdpa_capability_receipt(q, k, v, scale),
        "production_call_completed_before_diagnostic_cuda_work": True,
        "shadow_call_completed_before_diagnostic_cuda_work": True,
        "production_result_substituted": False,
        "downstream_execution_perturbed_after_shadow": True,
        "repeatability_scope": "eval0_block0_first_global_native_sdpa",
        "extra_attention_shadow_calls": 1,
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
