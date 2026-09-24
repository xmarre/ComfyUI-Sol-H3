"""Bounded one-shot receipts for the 00625 same-input runtime divergence.

This module is intentionally diagnostic-only. It never substitutes a replayed
result into model execution and it never consumes RNG state. On real SM120 CUDA
runs it samples a bounded set of tensor elements, hashes those samples plus tensor
layout metadata, and records exactly one evaluation-0 VDN dense replay plus the
first Sol calibration boundary.
"""
from __future__ import annotations

import hashlib
import json
import logging

import torch


SCHEMA = "sol_h3_00625_runtime_state_v1"
_SAMPLE_COUNT = 256
log = logging.getLogger("comfy.sol_h3")


def _sm120_tensor(tensor) -> bool:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        return False
    try:
        return tuple(torch.cuda.get_device_capability(tensor.device)) == (12, 0)
    except Exception:
        return False


def tensor_receipt(tensor, *, sample_count=_SAMPLE_COUNT):
    """Return a bounded deterministic receipt without materializing the full tensor."""
    if not isinstance(tensor, torch.Tensor):
        return None
    detached = tensor.detach()
    count = min(max(int(sample_count), 1), detached.numel()) if detached.numel() else 0
    metadata = {
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "sample_count": int(count),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if count == 0:
        return {**metadata, "sample_sha256": digest.hexdigest(), "sample_finite": True}

    # torch.take addresses the logical flattened tensor directly and avoids a
    # potentially enormous contiguous() materialization for BHQD views.
    indices = torch.linspace(
        0,
        detached.numel() - 1,
        steps=count,
        dtype=torch.float64,
        device="cpu",
    ).round().to(torch.long).to(detached.device)
    sample = torch.take(detached, indices).to(device="cpu", dtype=torch.float32).contiguous()
    digest.update(bytes(sample.view(torch.uint8).tolist()))
    finite = bool(torch.isfinite(sample).all().item())
    return {
        **metadata,
        "sample_sha256": digest.hexdigest(),
        "sample_finite": finite,
        "sample_abs_max": float(sample.abs().max().item()),
        "sample_mean": float(sample.mean().item()),
    }


def _store_once(state, key, payload) -> bool:
    receipts = getattr(state, "runtime_diagnostic", None)
    if not isinstance(receipts, dict) or key in receipts:
        return False
    entry = {"schema": SCHEMA, **payload}
    receipts[key] = entry
    log.warning(
        "Sol-H3 runtime-state diagnostic %s",
        json.dumps({"key": key, **entry}, sort_keys=True),
    )
    return True


def capture_stage(state, key, tensor, **metadata):
    if not _sm120_tensor(tensor):
        return False
    return _store_once(
        state,
        key,
        {
            **metadata,
            "tensor": tensor_receipt(tensor),
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
    """Replay one first-low VDN native attention call observationally.

    The first result is always returned. The replay is used only to test whether
    identical Q/K/V produce bit-identical native grouped-SDPA output inside the
    same runtime.
    """
    key = "eval0_block0_first_vdn_dense_warmup"
    receipts = getattr(state, "runtime_diagnostic", None)
    eligible = (
        isinstance(receipts, dict)
        and key not in receipts
        and evaluation == 0
        and block_index == 0
        and kind == "local"
        and _sm120_tensor(q)
        and _sm120_tensor(k)
        and _sm120_tensor(v)
    )
    if not eligible:
        return native()

    payload = {
        "evaluation": int(evaluation),
        "block_index": int(block_index),
        "kind": kind,
        "route": route,
        "sink_rows": int(sink_rows),
        "q": tensor_receipt(q),
        "k": tensor_receipt(k),
        "v": tensor_receipt(v),
    }
    result = native()
    payload["output"] = tensor_receipt(result)
    try:
        replay = native()
    except Exception as exc:
        payload["replay_error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload["replay_output"] = tensor_receipt(replay)
        payload["replay_exact_equal"] = bool(torch.equal(result, replay))
        del replay
    _store_once(state, key, payload)
    return result


def capture_sparse_gate(state, q, k, v, got, want, *, metrics):
    if not _sm120_tensor(q):
        return False
    return _store_once(
        state,
        "first_sparse_calibration_gate",
        {
            "q": tensor_receipt(q),
            "k": tensor_receipt(k),
            "v": tensor_receipt(v),
            "all_selected_kernel_output": tensor_receipt(got),
            "dense_reference_output": tensor_receipt(want),
            "reference_peak_abs": float(metrics.get("reference_peak_abs", float("nan"))),
            "max_abs": float(metrics.get("max_abs", float("nan"))),
            "mean_abs": float(metrics.get("mean_abs", float("nan"))),
            "rel_l2": float(metrics.get("rel_l2", float("nan"))),
        },
    )


def capture_sparse_output(state, output):
    if not _sm120_tensor(output):
        return False
    return _store_once(
        state,
        "first_sparse_runtime_output",
        {"tensor": tensor_receipt(output)},
    )
