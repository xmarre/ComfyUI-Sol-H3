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
        if key == "eval0_block0_first_vdn_dense_warmup":
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
):
    """Run one bounded first-low native grouped-SDPA shadow replay.

    The production native call executes first and its result is always returned.
    Q/K/V and production-output sampling is queued only after that result exists,
    so the production native call itself is not preceded by diagnostic CUDA work.
    The second native call is a deliberate shadow replay on the same Q/K/V; only
    its bounded receipt is retained.
    """
    key = "eval0_block0_first_vdn_dense_warmup"
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
        and kind == "local"
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
        "q": deferred_tensor_receipt(q),
        "k": deferred_tensor_receipt(k),
        "v": deferred_tensor_receipt(v),
        "output": deferred_tensor_receipt(result),
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


def capture_sparse_gate(state, q, k, v, got, want, *, metrics):
    if not getattr(state, "runtime_diagnostic_enabled", False) or not _sm120_tensor(q):
        return False
    context = getattr(state, "runtime_diagnostic_sparse_context", None)
    return _store_pending_once(
        state,
        "first_sparse_calibration_gate",
        {
            "context": dict(context) if isinstance(context, dict) else None,
            "q": deferred_tensor_receipt(q),
            "k": deferred_tensor_receipt(k),
            "v": deferred_tensor_receipt(v),
            "all_selected_kernel_output": deferred_tensor_receipt(got),
            "dense_reference_output": deferred_tensor_receipt(want),
            "reference_peak_abs": float(metrics.get("reference_peak_abs", float("nan"))),
            "max_abs": float(metrics.get("max_abs", float("nan"))),
            "mean_abs": float(metrics.get("mean_abs", float("nan"))),
            "rel_l2": float(metrics.get("rel_l2", float("nan"))),
        },
    )


def capture_sparse_output(state, output):
    if not getattr(state, "runtime_diagnostic_enabled", False) or not _sm120_tensor(output):
        return False
    return _store_pending_once(
        state,
        "first_sparse_runtime_output",
        {
            "context": (
                dict(state.runtime_diagnostic_sparse_context)
                if isinstance(getattr(state, "runtime_diagnostic_sparse_context", None), dict)
                else None
            ),
            "tensor": deferred_tensor_receipt(output),
        },
    )
