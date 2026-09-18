#!/usr/bin/env python3
"""Replay K1/K2 on real MiniMax-H3 Stage-A activations and exact Comfy RoPE.

The capture bundle supplies the real post-AdaLN H3 attention input and exact
`rope_freqs` rows observed during a plain native BF16 H3 forward. Projection and
normalization weights come from either the pinned native teacher (`teacher`) or a
canonical trained Keyless deploy checkpoint (`keyless`).

The materialized arithmetic oracle is produced with Comfy's own
`comfy.quant_ops.ck.rms_rope_split_half`, not the Sol torch route helper. K1/K2
remain experimental Triton primitives and are never promoted by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

_TOOL_PATH = Path(__file__).resolve()
_REPO_ROOT = _TOOL_PATH.parents[1]
_DEFAULT_COMFY_ROOT = _TOOL_PATH.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
if _DEFAULT_COMFY_ROOT.joinpath("comfy").is_dir():
    sys.path.insert(0, str(_DEFAULT_COMFY_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

from sol_h3.keyless_exact_attention import (  # noqa: E402
    CANONICAL_SCALE,
    CONTRACT as K2_CONTRACT,
    exact_attention,
)
from sol_h3.keyless_real_h3_replay import (  # noqa: E402
    ENVELOPE,
    apply_split_half_rope_fp32_from_normalized,
    apply_split_half_rope_from_normalized,
    block_summary_oracle,
    checkpoint_tensor_names,
    envelope_dict,
    identity_split_half_rope_like,
    metric_within_limit,
    public_rms_norm,
    replay_requires_process_failure,
    split_projection,
    tensor_metrics,
    tensor_scale_diagnostics,
    value_sum_within_limit,
)
from sol_h3.keyless_route_summary import (  # noqa: E402
    CONTRACT as K1_CONTRACT,
    HEAD_DIM,
    NORM_EPS,
    ROPE_HALF_DIM,
    materialized_fp32_rope_route_diagnostic,
    materialized_native_route_diagnostic,
    materialized_public_rounding_route_diagnostic,
    materialized_route_reference,
    route_summary,
)


HEADS = 56
HIDDEN = 5376
REPLAY_CONTRACT = "sol-h3-keyless-real-h3-replay-v1"
V2_CALIBRATION_CONTRACT = "sol-h3-keyless-v2-calibration-v1"
PROBE_CONTRACT = "sol-h3-keyless-v2-calibration-campaign-v5"
V2_K1_CONTRACT = "sol-h3-keyless-route-summary-v2"
V2_K2_CONTRACT = "sol-h3-keyless-exact-allselected-v2"

V2_CALIBRATION_CASES = (
    ("head-63x63", "head", 63, 63),
    ("head-64x64", "head", 64, 64),
    ("head-65x65", "head", 65, 65),
    ("quarter-127x127", "quarter", 127, 127),
    ("quarter-128x128", "quarter", 128, 128),
    ("quarter-129x129", "quarter", 129, 129),
    ("middle-255x255", "middle", 255, 255),
    ("middle-256x256", "middle", 256, 256),
    ("middle-257x257", "middle", 257, 257),
    ("three-quarter-193x511", "three_quarter", 193, 511),
    ("tail-511x193", "tail", 511, 193),
    ("tail-1025x1537", "tail", 1025, 1537),
)


def _probe_source_identity(repo_root: Path) -> dict[str, object]:
    """Bind an expensive replay to the exact local Sol-H3 checkout before loading data."""
    head_run = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if head_run.returncode != 0:
        raise RuntimeError(
            "cannot determine Sol-H3 probe source commit: "
            + head_run.stderr.strip()
        )
    status_run = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if status_run.returncode != 0:
        raise RuntimeError(
            "cannot determine Sol-H3 probe worktree state: "
            + status_run.stderr.strip()
        )
    return {
        "repo_root": str(repo_root),
        "git_commit": head_run.stdout.strip().lower(),
        "tracked_worktree_dirty": bool(status_run.stdout.strip()),
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _drop_read_cache(fd: int, offset: int, length: int) -> None:
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if callable(advise) and dontneed is not None:
        try:
            advise(fd, offset, length, dontneed)
        except OSError:
            pass


def _sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    """Hash very large artifacts without retaining them in Linux/WSL page cache."""
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        advise = getattr(os, "posix_fadvise", None)
        sequential = getattr(os, "POSIX_FADV_SEQUENTIAL", None)
        if callable(advise) and sequential is not None:
            try:
                advise(handle.fileno(), 0, 0, sequential)
            except OSError:
                pass
        offset = 0
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            consumed = len(chunk)
            _drop_read_cache(handle.fileno(), offset, consumed)
            offset += consumed
        _drop_read_cache(handle.fileno(), 0, 0)
    return digest.hexdigest()


def _load_capture(
    path: Path,
    *,
    keyless_repo: Path | None,
    expected_receipt_sha256: str | None,
    device: torch.device,
):
    if keyless_repo is not None:
        sys.path.insert(0, str(keyless_repo.resolve()))
    try:
        from minimax_h3_keyless.capture_io import load_captured_pilot_bundle
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-H3-Keyless is required to validate Stage-A captures; "
            "pass --keyless-repo or put it on PYTHONPATH"
        ) from exc
    return load_captured_pilot_bundle(
        path,
        expected_receipt_sha256=expected_receipt_sha256,
        map_location=device,
    )


def _load_block_weights(
    checkpoint: Path,
    *,
    kind: str,
    block_index: int,
    device: torch.device,
):
    names = checkpoint_tensor_names(kind, block_index)
    with safe_open(str(checkpoint), framework="pt", device=str(device)) as handle:
        metadata = dict(handle.metadata() or {})
        available = set(handle.keys())
        missing = [name for name in names.values() if name not in available]
        if missing:
            raise RuntimeError(f"checkpoint is missing block {block_index} tensors: {missing}")
        projection = handle.get_tensor(names["projection"])
        q_norm = handle.get_tensor(names["q_norm"])
        route_norm = handle.get_tensor(names["route_norm"])

    if projection.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"replay checkpoint projection must be BF16, got {projection.dtype}"
        )
    if q_norm.dtype is not torch.bfloat16 or route_norm.dtype is not torch.bfloat16:
        raise RuntimeError("replay q_norm/route_norm weights must be BF16")
    if tuple(q_norm.shape) != (HEAD_DIM,) or tuple(route_norm.shape) != (HEAD_DIM,):
        raise RuntimeError("replay q_norm/route_norm weights must have shape [128]")
    q_weight, v_weight = split_projection(projection, kind=kind)
    return (
        q_weight.contiguous(),
        v_weight.contiguous(),
        q_norm.contiguous(),
        route_norm.contiguous(),
        metadata,
        names,
    )


def _project(
    hidden_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    *,
    start: int,
    rows: int,
    device: torch.device,
) -> torch.Tensor:
    stop = start + rows
    if start < 0 or rows <= 0 or stop > hidden_cpu.shape[0]:
        raise ValueError(
            f"requested rows [{start},{stop}) exceed captured length {hidden_cpu.shape[0]}"
        )
    hidden = hidden_cpu[start:stop].to(device=device, dtype=torch.bfloat16)
    weight = weight_cpu.to(device=device, dtype=torch.bfloat16)
    try:
        return F.linear(hidden, weight)
    finally:
        del weight


def _rope_slice(
    rope_cpu: torch.Tensor,
    *,
    start: int,
    rows: int,
    device: torch.device,
) -> torch.Tensor:
    expected_tail = (1, ROPE_HALF_DIM, 2, 2)
    if rope_cpu.ndim != 6 or tuple(rope_cpu.shape[2:]) != expected_tail:
        raise RuntimeError(
            "captured H3 rope_freqs must have shape [1,T,1,48,2,2], got "
            f"{tuple(rope_cpu.shape)}"
        )
    stop = start + rows
    if start < 0 or rows <= 0 or stop > rope_cpu.shape[1]:
        raise ValueError(
            f"requested RoPE rows [{start},{stop}) exceed captured length {rope_cpu.shape[1]}"
        )
    return rope_cpu[:, start:stop].to(device=device, dtype=torch.bfloat16).contiguous()


def _comfy_position(
    raw: torch.Tensor,
    norm_weight: torch.Tensor,
    rope: torch.Tensor,
) -> torch.Tensor:
    try:
        import comfy.quant_ops
    except ImportError as exc:
        raise RuntimeError("ComfyUI must be on PYTHONPATH for exact H3 replay") from exc

    ck = getattr(comfy.quant_ops, "ck", None)
    function = getattr(ck, "rms_rope_split_half", None)
    if not callable(function):
        raise RuntimeError("current Comfy quant backend lacks rms_rope_split_half")
    x = raw.unsqueeze(0).contiguous()
    shadow = raw.unsqueeze(0).contiguous()
    weight = norm_weight.to(device=raw.device, dtype=torch.bfloat16).contiguous()
    rot_dim = int(rope.shape[-3]) * 2
    q, _ = function(
        x,
        shadow,
        rope,
        weight,
        weight,
        epsilon=NORM_EPS,
        rot_dim=rot_dim,
    )
    if tuple(q.shape) != (1, raw.shape[0], raw.shape[1], raw.shape[2]):
        raise RuntimeError(
            f"Comfy rms_rope_split_half returned unexpected shape {tuple(q.shape)}"
        )
    return q[0].contiguous()


def _comfy_apply_split_half_rope(
    normalized: torch.Tensor,
    rope: torch.Tensor,
) -> torch.Tensor:
    """Apply Comfy's selected split-half RoPE backend to an already-normalized H3 route."""
    try:
        import comfy.quant_ops
    except ImportError as exc:
        raise RuntimeError("ComfyUI must be importable for exact H3 RoPE isolation") from exc

    ck = getattr(comfy.quant_ops, "ck", None)
    function = getattr(ck, "apply_rope_split_half", None)
    if not callable(function):
        raise RuntimeError("current Comfy quant backend lacks apply_rope_split_half")
    rot_dim = int(rope.shape[-3]) * 2
    prefix = normalized[..., :rot_dim].unsqueeze(0).contiguous()
    shadow = prefix.clone()
    q, _ = function(prefix, shadow, rope)
    if tuple(q.shape) != tuple(prefix.shape):
        raise RuntimeError(
            f"Comfy apply_rope_split_half returned unexpected shape {tuple(q.shape)}"
        )
    return torch.cat((q[0], normalized[..., rot_dim:]), dim=-1).contiguous()


def _comfy_apply_rope_backend_identity(
    normalized: torch.Tensor,
    rope: torch.Tensor,
) -> dict[str, object]:
    """Record the exact comfy-kitchen backend selected for standalone split-half RoPE."""
    try:
        import comfy.quant_ops
    except ImportError as exc:
        raise RuntimeError("ComfyUI must be importable for backend provenance") from exc

    ck = getattr(comfy.quant_ops, "ck", None)
    registry = getattr(ck, "registry", None)
    if registry is None:
        return {"selected_backend": None, "reason": "comfy-kitchen registry unavailable"}

    rot_dim = int(rope.shape[-3]) * 2
    prefix = normalized[..., :rot_dim].unsqueeze(0).contiguous()
    shadow = prefix.clone()
    kwargs = {
        "xq": prefix,
        "xk": shadow,
        "freqs_cis": rope,
    }
    backend = registry.get_capable_backend("apply_rope_split_half", kwargs=kwargs)
    implementation = registry.get_implementation(
        "apply_rope_split_half",
        backend=backend,
        kwargs=kwargs,
    )
    try:
        from importlib.metadata import version as distribution_version
        kitchen_version = distribution_version("comfy-kitchen")
    except Exception:
        kitchen_version = None
    return {
        "selected_backend": str(backend),
        "implementation_module": str(getattr(implementation, "__module__", "")),
        "implementation_name": str(getattr(implementation, "__name__", "")),
        "comfy_kitchen_version": kitchen_version,
    }


def _comfy_rms_rope_backend_identity(
    raw: torch.Tensor,
    norm_weight: torch.Tensor,
    rope: torch.Tensor,
) -> dict[str, object]:
    """Record the exact comfy-kitchen backend selected for this H3 fused call."""
    try:
        import comfy.quant_ops
    except ImportError as exc:
        raise RuntimeError("ComfyUI must be importable for backend provenance") from exc

    ck = getattr(comfy.quant_ops, "ck", None)
    registry = getattr(ck, "registry", None)
    if registry is None:
        return {"selected_backend": None, "reason": "comfy-kitchen registry unavailable"}

    x = raw.unsqueeze(0).contiguous()
    shadow = x.clone()
    weight = norm_weight.to(device=raw.device, dtype=torch.bfloat16).contiguous()
    kwargs = {
        "q": x,
        "k": shadow,
        "freqs_cis": rope,
        "q_scale": weight,
        "k_scale": weight,
        "epsilon": NORM_EPS,
        "rot_dim": int(rope.shape[-3]) * 2,
    }
    backend = registry.get_capable_backend("rms_rope_split_half", kwargs=kwargs)
    implementation = registry.get_implementation(
        "rms_rope_split_half",
        backend=backend,
        kwargs=kwargs,
    )
    try:
        from importlib.metadata import version as distribution_version
        kitchen_version = distribution_version("comfy-kitchen")
    except Exception:
        kitchen_version = None
    return {
        "selected_backend": str(backend),
        "implementation_module": str(getattr(implementation, "__module__", "")),
        "implementation_name": str(getattr(implementation, "__name__", "")),
        "comfy_kitchen_version": kitchen_version,
    }


def _dense_attention(
    q: torch.Tensor,
    route: torch.Tensor,
    raw_v: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        route.transpose(0, 1).unsqueeze(0),
        raw_v.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
        scale=CANONICAL_SCALE,
    ).squeeze(0).transpose(0, 1)


def _dense_attention_fp32(
    q: torch.Tensor,
    route: torch.Tensor,
    raw_v: torch.Tensor,
) -> torch.Tensor:
    """Explicit FP32-softmax diagnostic oracle, returned in BF16."""
    qh = q.float().transpose(0, 1)
    rh = route.float().permute(1, 2, 0)
    vh = raw_v.float().transpose(0, 1)
    logits = torch.matmul(qh, rh) * float(CANONICAL_SCALE)
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, vh).transpose(0, 1).to(dtype=torch.bfloat16)


def _timings(function, device: torch.device, repeats: int) -> dict[str, Any]:
    for _ in range(2):
        value = function()
        del value
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
        del value
    return {
        "median": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "samples": samples,
    }


def _candidate_allocation(function, device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    value = function()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del value
    return {
        "baseline_allocated": int(baseline),
        "peak_allocated": int(peak),
        "temporary_peak_delta": int(max(0, peak - baseline)),
    }


def _resolve_calibration_start(total_rows: int, span: int, anchor: str) -> int:
    if total_rows <= 0 or span <= 0 or span > total_rows:
        raise ValueError(
            f"calibration span {span} does not fit captured length {total_rows}"
        )
    available = total_rows - span
    if anchor == "head":
        return 0
    if anchor == "quarter":
        return available // 4
    if anchor == "middle":
        return available // 2
    if anchor == "three_quarter":
        return (available * 3) // 4
    if anchor == "tail":
        return available
    raise ValueError(f"unsupported v2 calibration anchor: {anchor!r}")


def _calibration_v2_for_block(
    record,
    *,
    checkpoint: Path,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Collect threshold-free K1/K2 v2 arithmetic evidence across fixed windows."""
    if record.case.rope_freqs is None or not torch.is_tensor(record.case.rope_freqs):
        raise RuntimeError(f"capture block {record.block_index} has no exact H3 RoPE tensor")
    if tuple(record.attention_input.shape) != tuple(record.case.x.shape):
        raise RuntimeError("capture attention_input no longer matches block-input shape")
    if record.attention_input.ndim != 2 or record.attention_input.shape[1] != HIDDEN:
        raise RuntimeError(
            f"capture attention_input must be [T,{HIDDEN}], got "
            f"{tuple(record.attention_input.shape)}"
        )

    q_weight, v_weight, q_norm, route_norm, metadata, names = _load_block_weights(
        checkpoint,
        kind=checkpoint_kind,
        block_index=record.block_index,
        device=device,
    )
    total_rows = int(record.attention_input.shape[0])
    q_norm = q_norm.to(device=device, dtype=torch.bfloat16)
    route_norm = route_norm.to(device=device, dtype=torch.bfloat16)
    cases: list[dict[str, Any]] = []

    try:
        for name, anchor, q_rows, v_rows in V2_CALIBRATION_CASES:
            span = max(q_rows, v_rows)
            start = _resolve_calibration_start(total_rows, span, anchor)
            q_start = start
            v_start = start

            q_raw = _project(
                record.attention_input,
                q_weight,
                start=q_start,
                rows=q_rows,
                device=device,
            ).view(q_rows, HEADS, HEAD_DIM)
            v_raw = _project(
                record.attention_input,
                v_weight,
                start=v_start,
                rows=v_rows,
                device=device,
            ).view(v_rows, HEADS, HEAD_DIM)
            q_rope = _rope_slice(
                record.case.rope_freqs,
                start=q_start,
                rows=q_rows,
                device=device,
            )
            v_rope = _rope_slice(
                record.case.rope_freqs,
                start=v_start,
                rows=v_rows,
                device=device,
            )
            q = _comfy_position(q_raw, q_norm, q_rope)
            route = _comfy_position(v_raw, route_norm, v_rope)

            oracle_rc, oracle_vc = block_summary_oracle(route, v_raw)
            candidate_rc, candidate_vc = route_summary(
                v_raw,
                route_norm,
                NORM_EPS,
                v_rope,
            )
            k1_rc = tensor_metrics(candidate_rc, oracle_rc)
            k1_rc_scale = tensor_scale_diagnostics(candidate_rc, oracle_rc)
            k1_vc = tensor_metrics(candidate_vc, oracle_vc)

            dense = _dense_attention(q, route, v_raw)
            out = torch.empty_like(q)
            exact_attention(
                q,
                v_raw,
                route_norm,
                NORM_EPS,
                v_rope,
                scale=CANONICAL_SCALE,
                out=out,
            )
            torch.cuda.synchronize(device)
            k2 = tensor_metrics(out, dense)
            k2_scale = tensor_scale_diagnostics(out, dense)

            k1_allocation = _candidate_allocation(
                lambda: route_summary(v_raw, route_norm, NORM_EPS, v_rope),
                device,
            )
            k2_allocation = _candidate_allocation(
                lambda: exact_attention(
                    q,
                    v_raw,
                    route_norm,
                    NORM_EPS,
                    v_rope,
                    scale=CANONICAL_SCALE,
                    out=out,
                ),
                device,
            )
            full_route_bytes = int(v_raw.numel() * v_raw.element_size())
            k1_allocation["full_materialized_route_bytes"] = full_route_bytes
            k1_allocation["below_full_route"] = (
                k1_allocation["temporary_peak_delta"] < full_route_bytes
            )
            k2_allocation["full_materialized_route_bytes"] = full_route_bytes
            k2_allocation["below_full_route"] = (
                k2_allocation["temporary_peak_delta"] < full_route_bytes
            )

            cases.append(
                {
                    "name": name,
                    "anchor": anchor,
                    "q_start": q_start,
                    "q_rows": q_rows,
                    "v_start": v_start,
                    "v_rows": v_rows,
                    "k1_route_centroid": k1_rc,
                    "k1_route_centroid_scale": k1_rc_scale,
                    "k1_raw_value_sum": k1_vc,
                    "k1_allocation": k1_allocation,
                    "k2_output": k2,
                    "k2_output_scale": k2_scale,
                    "k2_allocation": k2_allocation,
                }
            )
    finally:
        pass

    return {
        "block_index": int(record.block_index),
        "case_id": record.case.case_id,
        "sigma": record.case.sigma,
        "modality_label": record.case.modality_label,
        "capture_rows": total_rows,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_tensor_names": names,
        "checkpoint_metadata_identity": {
            key: metadata.get(key)
            for key in (
                "architecture",
                "checkpoint_format_version",
                "manifest_sha256",
                "parent_model_sha256",
                "teacher_model_sha256",
                "training_run",
                "export_commit",
            )
            if metadata.get(key) is not None
        },
        "cases": cases,
    }


def _calibration_v2_extrema(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    flat = [
        {
            "block_index": block["block_index"],
            **case,
        }
        for block in blocks
        for case in block["cases"]
    ]
    if not flat:
        raise RuntimeError("v2 calibration produced no cases")

    def maximum(metric_key: str, field: str) -> dict[str, object]:
        row = max(flat, key=lambda item: float(item[metric_key][field]))
        return {
            "value": float(row[metric_key][field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def maximum_ulp(scale_key: str) -> dict[str, object]:
        def value(item: dict[str, Any]) -> float:
            raw = item[scale_key]["worst"]["abs_error_in_want_bf16_ulps"]
            return float(raw) if raw is not None else float("inf")

        row = max(flat, key=value)
        return {
            "value": value(row),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def maximum_scale_ratio(scale_key: str, field: str) -> dict[str, object]:
        def value(item: dict[str, Any]) -> float:
            raw = item[scale_key].get(field)
            return float(raw) if raw is not None else float("-inf")

        row = max(flat, key=value)
        return {
            "value": value(row),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    return {
        "case_count": len(flat),
        "k1_route_centroid": {
            "rel_l2": maximum("k1_route_centroid", "rel_l2"),
            "mean_abs": maximum("k1_route_centroid", "mean_abs"),
            "max_abs": maximum("k1_route_centroid", "max_abs"),
            "max_abs_over_want_abs_max": maximum_scale_ratio(
                "k1_route_centroid_scale",
                "max_abs_over_want_abs_max",
            ),
            "mean_abs_over_want_mean_abs": maximum_scale_ratio(
                "k1_route_centroid_scale",
                "mean_abs_over_want_mean_abs",
            ),
            "worst_bf16_ulps": maximum_ulp("k1_route_centroid_scale"),
        },
        "k1_raw_value_sum_max_abs": maximum("k1_raw_value_sum", "max_abs"),
        "k2_output": {
            "rel_l2": maximum("k2_output", "rel_l2"),
            "mean_abs": maximum("k2_output", "mean_abs"),
            "max_abs": maximum("k2_output", "max_abs"),
            "max_abs_over_want_abs_max": maximum_scale_ratio(
                "k2_output_scale",
                "max_abs_over_want_abs_max",
            ),
            "mean_abs_over_want_mean_abs": maximum_scale_ratio(
                "k2_output_scale",
                "mean_abs_over_want_mean_abs",
            ),
            "worst_bf16_ulps": maximum_ulp("k2_output_scale"),
        },
        "all_k1_allocations_below_full_route": all(
            bool(row["k1_allocation"]["below_full_route"]) for row in flat
        ),
        "all_k2_allocations_below_full_route": all(
            bool(row["k2_allocation"]["below_full_route"]) for row in flat
        ),
    }


def _record_for_block(
    record,
    *,
    checkpoint: Path,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    device: torch.device,
    q_start: int,
    q_rows: int,
    v_start: int,
    v_rows: int,
    repeats: int,
) -> dict[str, Any]:
    if record.case.rope_freqs is None or not torch.is_tensor(record.case.rope_freqs):
        raise RuntimeError(f"capture block {record.block_index} has no exact H3 RoPE tensor")
    if tuple(record.attention_input.shape) != tuple(record.case.x.shape):
        raise RuntimeError("capture attention_input no longer matches block-input shape")
    if record.attention_input.ndim != 2 or record.attention_input.shape[1] != HIDDEN:
        raise RuntimeError(
            f"capture attention_input must be [T,{HIDDEN}], got "
            f"{tuple(record.attention_input.shape)}"
        )

    q_weight, v_weight, q_norm_cpu, route_norm_cpu, metadata, names = _load_block_weights(
        checkpoint,
        kind=checkpoint_kind,
        block_index=record.block_index,
        device=device,
    )
    q_raw = _project(
        record.attention_input,
        q_weight,
        start=q_start,
        rows=q_rows,
        device=device,
    ).view(q_rows, HEADS, HEAD_DIM)
    v_raw = _project(
        record.attention_input,
        v_weight,
        start=v_start,
        rows=v_rows,
        device=device,
    ).view(v_rows, HEADS, HEAD_DIM)
    del q_weight, v_weight

    q_rope = _rope_slice(
        record.case.rope_freqs,
        start=q_start,
        rows=q_rows,
        device=device,
    )
    v_rope = _rope_slice(
        record.case.rope_freqs,
        start=v_start,
        rows=v_rows,
        device=device,
    )
    q_norm = q_norm_cpu.to(device=device, dtype=torch.bfloat16)
    route_norm = route_norm_cpu.to(device=device, dtype=torch.bfloat16)
    q = _comfy_position(q_raw, q_norm, q_rope)
    route = _comfy_position(v_raw, route_norm, v_rope)
    del q_raw, q_norm, q_rope

    oracle_rc, oracle_vc = block_summary_oracle(route, v_raw)
    candidate_rc, candidate_vc = route_summary(v_raw, route_norm, NORM_EPS, v_rope)
    k1_rc = tensor_metrics(candidate_rc, oracle_rc)
    k1_vc = tensor_metrics(candidate_vc, oracle_vc)
    k1_rc_scale = tensor_scale_diagnostics(candidate_rc, oracle_rc)

    # Diagnostic-only decomposition. The frozen pass/fail gate remains against the
    # exact Comfy-positioned route above. This second materialization asks whether
    # a failure is already introduced by Comfy-vs-public-Keyless RMSNorm/RoPE
    # rounding, or later by the attention mainloop.
    torch_route = materialized_route_reference(
        v_raw,
        route_norm,
        NORM_EPS,
        v_rope,
    )
    torch_route_vs_comfy = tensor_metrics(torch_route, route)
    torch_route_vs_comfy_scale = tensor_scale_diagnostics(torch_route, route)
    torch_oracle_rc, _ = block_summary_oracle(torch_route, v_raw)
    k1_vs_torch_route = tensor_metrics(candidate_rc, torch_oracle_rc)
    k1_vs_torch_route_scale = tensor_scale_diagnostics(candidate_rc, torch_oracle_rc)

    # Materialize the exact route arithmetic duplicated from the experimental
    # K1/K2 Triton kernels. This intentionally allocates a full route only for
    # diagnostics, never for production dispatch.
    native_route = materialized_native_route_diagnostic(
        v_raw,
        route_norm,
        NORM_EPS,
        v_rope,
    )
    native_route_vs_comfy = tensor_metrics(native_route, route)
    native_route_vs_comfy_scale = tensor_scale_diagnostics(native_route, route)
    native_route_vs_public = tensor_metrics(native_route, torch_route)
    native_route_vs_public_scale = tensor_scale_diagnostics(native_route, torch_route)
    native_oracle_rc, _ = block_summary_oracle(native_route, v_raw)
    k1_vs_native_route = tensor_metrics(candidate_rc, native_oracle_rc)
    k1_vs_native_route_scale = tensor_scale_diagnostics(candidate_rc, native_oracle_rc)

    public_rounding_route = materialized_public_rounding_route_diagnostic(
        v_raw,
        route_norm,
        NORM_EPS,
        v_rope,
    )
    public_rounding_vs_public = tensor_metrics(public_rounding_route, torch_route)
    public_rounding_vs_public_scale = tensor_scale_diagnostics(
        public_rounding_route,
        torch_route,
    )
    public_rounding_vs_comfy = tensor_metrics(public_rounding_route, route)
    public_rounding_vs_comfy_scale = tensor_scale_diagnostics(
        public_rounding_route,
        route,
    )

    # Isolate RMSNorm from RoPE. H3 uses partial RoPE (96/128), but using an
    # identity 2x2 rotation lets the exact Comfy fused backend expose its
    # materialized BF16 RMSNorm result over all 128 channels without changing
    # the backend or reduction implementation.
    identity_rope = identity_split_half_rope_like(v_rope)
    comfy_norm = _comfy_position(v_raw, route_norm, identity_rope)
    public_norm = public_rms_norm(v_raw, route_norm, NORM_EPS)
    native_norm = materialized_native_route_diagnostic(
        v_raw,
        route_norm,
        NORM_EPS,
        identity_rope,
    )
    rms_public_vs_comfy = tensor_metrics(public_norm, comfy_norm)
    rms_public_vs_comfy_scale = tensor_scale_diagnostics(public_norm, comfy_norm)
    rms_native_vs_comfy = tensor_metrics(native_norm, comfy_norm)
    rms_native_vs_comfy_scale = tensor_scale_diagnostics(native_norm, comfy_norm)
    rms_native_vs_public = tensor_metrics(native_norm, public_norm)
    rms_native_vs_public_scale = tensor_scale_diagnostics(native_norm, public_norm)

    tail = slice(ROPE_HALF_DIM * 2, HEAD_DIM)
    rms_tail_public_vs_comfy = tensor_metrics(public_norm[..., tail], comfy_norm[..., tail])
    rms_tail_native_vs_comfy = tensor_metrics(native_norm[..., tail], comfy_norm[..., tail])
    rms_tail_native_vs_public = tensor_metrics(native_norm[..., tail], public_norm[..., tail])

    # Now hold normalization fixed and vary only RoPE arithmetic. This separates
    # reduction/rsqrt/scaling differences from the split-half rotation itself.
    torch_rope_from_comfy_norm = apply_split_half_rope_from_normalized(
        comfy_norm,
        v_rope,
        rot_dim=ROPE_HALF_DIM * 2,
    )
    torch_rope_from_public_norm = apply_split_half_rope_from_normalized(
        public_norm,
        v_rope,
        rot_dim=ROPE_HALF_DIM * 2,
    )
    torch_rope_from_native_norm = apply_split_half_rope_from_normalized(
        native_norm,
        v_rope,
        rot_dim=ROPE_HALF_DIM * 2,
    )
    rope_torch_from_comfy_vs_comfy = tensor_metrics(torch_rope_from_comfy_norm, route)
    rope_torch_from_comfy_vs_comfy_scale = tensor_scale_diagnostics(
        torch_rope_from_comfy_norm,
        route,
    )
    rope_torch_from_public_vs_public = tensor_metrics(
        torch_rope_from_public_norm,
        torch_route,
    )
    rope_torch_from_native_vs_native = tensor_metrics(
        torch_rope_from_native_norm,
        native_route,
    )
    rope_torch_from_native_vs_native_scale = tensor_scale_diagnostics(
        torch_rope_from_native_norm,
        native_route,
    )

    comfy_rope_from_comfy_norm = _comfy_apply_split_half_rope(comfy_norm, v_rope)
    comfy_rope_from_native_norm = _comfy_apply_split_half_rope(native_norm, v_rope)
    rope_comfy_from_comfy_vs_fused = tensor_metrics(comfy_rope_from_comfy_norm, route)
    rope_comfy_from_comfy_vs_fused_scale = tensor_scale_diagnostics(
        comfy_rope_from_comfy_norm,
        route,
    )
    rope_torch_vs_comfy_from_comfy_norm = tensor_metrics(
        torch_rope_from_comfy_norm,
        comfy_rope_from_comfy_norm,
    )
    rope_torch_vs_comfy_from_comfy_norm_scale = tensor_scale_diagnostics(
        torch_rope_from_comfy_norm,
        comfy_rope_from_comfy_norm,
    )
    rope_comfy_from_native_vs_native = tensor_metrics(
        comfy_rope_from_native_norm,
        native_route,
    )
    rope_comfy_from_native_vs_native_scale = tensor_scale_diagnostics(
        comfy_rope_from_native_norm,
        native_route,
    )

    # Test the exact compute-precision distinction in comfy-kitchen CUDA:
    # fused RMS+RoPE promotes the materialized BF16 normalized pair and frequency
    # values to FP32 for the 2x2 rotation before casting the result back to BF16.
    fp32_rope_from_comfy_norm = apply_split_half_rope_fp32_from_normalized(
        comfy_norm,
        v_rope,
        rot_dim=ROPE_HALF_DIM * 2,
    )
    fp32_rope_from_native_norm = apply_split_half_rope_fp32_from_normalized(
        native_norm,
        v_rope,
        rot_dim=ROPE_HALF_DIM * 2,
    )
    rope_fp32_from_comfy_vs_fused = tensor_metrics(fp32_rope_from_comfy_norm, route)
    rope_fp32_from_comfy_vs_fused_scale = tensor_scale_diagnostics(
        fp32_rope_from_comfy_norm,
        route,
    )

    native_fp32_rope_route = materialized_fp32_rope_route_diagnostic(
        v_raw,
        route_norm,
        NORM_EPS,
        v_rope,
    )
    native_fp32_rope_vs_comfy = tensor_metrics(native_fp32_rope_route, route)
    native_fp32_rope_vs_comfy_scale = tensor_scale_diagnostics(
        native_fp32_rope_route,
        route,
    )
    native_fp32_rope_vs_native = tensor_metrics(native_fp32_rope_route, native_route)
    native_fp32_rope_vs_native_scale = tensor_scale_diagnostics(
        native_fp32_rope_route,
        native_route,
    )
    rope_fp32_from_native_vs_native_fp32_route = tensor_metrics(
        fp32_rope_from_native_norm,
        native_fp32_rope_route,
    )
    rope_fp32_from_native_vs_native_fp32_route_scale = tensor_scale_diagnostics(
        fp32_rope_from_native_norm,
        native_fp32_rope_route,
    )

    comfy_backend = _comfy_rms_rope_backend_identity(v_raw, route_norm, v_rope)
    comfy_apply_backend = _comfy_apply_rope_backend_identity(comfy_norm, v_rope)

    dense = _dense_attention(q, route, v_raw)
    dense_torch_route = _dense_attention(q, torch_route, v_raw)
    dense_native_route = _dense_attention(q, native_route, v_raw)
    dense_public_rounding_route = _dense_attention(q, public_rounding_route, v_raw)
    dense_native_fp32_rope_route = _dense_attention(q, native_fp32_rope_route, v_raw)
    dense_comfy_fp32 = _dense_attention_fp32(q, route, v_raw)
    dense_native_fp32 = _dense_attention_fp32(q, native_route, v_raw)
    out = torch.empty_like(q)
    exact_attention(
        q,
        v_raw,
        route_norm,
        NORM_EPS,
        v_rope,
        scale=CANONICAL_SCALE,
        out=out,
    )
    torch.cuda.synchronize(device)
    k2 = tensor_metrics(out, dense)
    k2_scale = tensor_scale_diagnostics(out, dense)
    k2_vs_torch_route = tensor_metrics(out, dense_torch_route)
    k2_vs_torch_route_scale = tensor_scale_diagnostics(out, dense_torch_route)
    dense_route_effect = tensor_metrics(dense_torch_route, dense)
    dense_route_effect_scale = tensor_scale_diagnostics(dense_torch_route, dense)

    k2_vs_native_route = tensor_metrics(out, dense_native_route)
    k2_vs_native_route_scale = tensor_scale_diagnostics(out, dense_native_route)
    native_dense_vs_comfy_dense = tensor_metrics(dense_native_route, dense)
    native_dense_vs_comfy_dense_scale = tensor_scale_diagnostics(
        dense_native_route,
        dense,
    )
    native_sdpa_vs_fp32 = tensor_metrics(dense_native_route, dense_native_fp32)
    native_sdpa_vs_fp32_scale = tensor_scale_diagnostics(
        dense_native_route,
        dense_native_fp32,
    )
    comfy_sdpa_vs_fp32 = tensor_metrics(dense, dense_comfy_fp32)
    comfy_sdpa_vs_fp32_scale = tensor_scale_diagnostics(dense, dense_comfy_fp32)
    k2_vs_native_fp32 = tensor_metrics(out, dense_native_fp32)
    k2_vs_native_fp32_scale = tensor_scale_diagnostics(out, dense_native_fp32)
    dense_public_rounding_vs_public = tensor_metrics(
        dense_public_rounding_route,
        dense_torch_route,
    )
    dense_public_rounding_vs_public_scale = tensor_scale_diagnostics(
        dense_public_rounding_route,
        dense_torch_route,
    )
    native_fp32_rope_dense_vs_comfy = tensor_metrics(
        dense_native_fp32_rope_route,
        dense,
    )
    native_fp32_rope_dense_vs_comfy_scale = tensor_scale_diagnostics(
        dense_native_fp32_rope_route,
        dense,
    )

    route_bytes = int(v_raw.numel() * v_raw.element_size())
    del (
        route,
        torch_route,
        native_route,
        public_rounding_route,
        identity_rope,
        comfy_norm,
        public_norm,
        native_norm,
        torch_rope_from_comfy_norm,
        torch_rope_from_public_norm,
        torch_rope_from_native_norm,
        comfy_rope_from_comfy_norm,
        comfy_rope_from_native_norm,
        fp32_rope_from_comfy_norm,
        fp32_rope_from_native_norm,
        native_fp32_rope_route,
        oracle_rc,
        oracle_vc,
        torch_oracle_rc,
        native_oracle_rc,
        candidate_rc,
        candidate_vc,
        dense,
        dense_torch_route,
        dense_native_route,
        dense_public_rounding_route,
        dense_native_fp32_rope_route,
        dense_comfy_fp32,
        dense_native_fp32,
    )
    torch.cuda.synchronize(device)

    k1_allocation = _candidate_allocation(
        lambda: route_summary(v_raw, route_norm, NORM_EPS, v_rope),
        device,
    )
    k2_allocation = _candidate_allocation(
        lambda: exact_attention(
            q,
            v_raw,
            route_norm,
            NORM_EPS,
            v_rope,
            scale=CANONICAL_SCALE,
            out=out,
        ),
        device,
    )
    k1_allocation["full_materialized_route_bytes"] = route_bytes
    k2_allocation["full_materialized_route_bytes"] = route_bytes
    k1_allocation["below_full_route"] = (
        k1_allocation["temporary_peak_delta"] < route_bytes
    )
    k2_allocation["below_full_route"] = (
        k2_allocation["temporary_peak_delta"] < route_bytes
    )

    k1_timing = _timings(
        lambda: route_summary(v_raw, route_norm, NORM_EPS, v_rope),
        device,
        repeats,
    )
    k2_timing = _timings(
        lambda: exact_attention(
            q,
            v_raw,
            route_norm,
            NORM_EPS,
            v_rope,
            scale=CANONICAL_SCALE,
            out=out,
        ),
        device,
        repeats,
    )

    passes = {
        "k1_route_centroid": metric_within_limit(
            k1_rc,
            ENVELOPE.k1_route_centroid,
        ),
        "k1_value_sum": value_sum_within_limit(
            k1_vc,
            ENVELOPE.k1_value_max_abs,
        ),
        "k2_output": metric_within_limit(k2, ENVELOPE.k2_output),
        "k1_allocation": bool(k1_allocation["below_full_route"]),
        "k2_allocation": bool(k2_allocation["below_full_route"]),
    }
    passes["all"] = all(passes.values())

    return {
        "block_index": int(record.block_index),
        "case_id": record.case.case_id,
        "sigma": record.case.sigma,
        "modality_label": record.case.modality_label,
        "capture_rows": int(record.attention_input.shape[0]),
        "q_window": {"start": q_start, "rows": q_rows},
        "v_window": {"start": v_start, "rows": v_rows},
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_tensor_names": names,
        "rope_dtype": str(v_rope.dtype),
        "checkpoint_metadata_identity": {
            key: metadata.get(key)
            for key in (
                "architecture",
                "checkpoint_format_version",
                "manifest_sha256",
                "parent_model_sha256",
                "teacher_model_sha256",
                "training_run",
                "export_commit",
            )
            if metadata.get(key) is not None
        },
        "k1": {
            "contract": K1_CONTRACT,
            "route_centroid": k1_rc,
            "route_centroid_scale": k1_rc_scale,
            "raw_value_sum": k1_vc,
            "diagnostic_public_keyless_route": {
                "full_route_vs_comfy_route": torch_route_vs_comfy,
                "full_route_vs_comfy_route_scale": torch_route_vs_comfy_scale,
                "candidate_summary_vs_public_route_summary": k1_vs_torch_route,
                "candidate_summary_vs_public_route_summary_scale": k1_vs_torch_route_scale,
            },
            "diagnostic_public_rounding_route": {
                "full_route_vs_public_route": public_rounding_vs_public,
                "full_route_vs_public_route_scale": public_rounding_vs_public_scale,
                "full_route_vs_comfy_route": public_rounding_vs_comfy,
                "full_route_vs_comfy_route_scale": public_rounding_vs_comfy_scale,
            },
            "diagnostic_rmsnorm_isolation": {
                "comfy_backend": comfy_backend,
                "public_norm_vs_comfy_norm": rms_public_vs_comfy,
                "public_norm_vs_comfy_norm_scale": rms_public_vs_comfy_scale,
                "native_norm_vs_comfy_norm": rms_native_vs_comfy,
                "native_norm_vs_comfy_norm_scale": rms_native_vs_comfy_scale,
                "native_norm_vs_public_norm": rms_native_vs_public,
                "native_norm_vs_public_norm_scale": rms_native_vs_public_scale,
                "tail_96_128": {
                    "public_norm_vs_comfy_norm": rms_tail_public_vs_comfy,
                    "native_norm_vs_comfy_norm": rms_tail_native_vs_comfy,
                    "native_norm_vs_public_norm": rms_tail_native_vs_public,
                },
            },
            "diagnostic_rope_isolation": {
                "standalone_comfy_backend": comfy_apply_backend,
                "torch_rope_from_comfy_norm_vs_comfy_fused_route": rope_torch_from_comfy_vs_comfy,
                "torch_rope_from_comfy_norm_vs_comfy_fused_route_scale": rope_torch_from_comfy_vs_comfy_scale,
                "torch_rope_from_public_norm_vs_public_route": rope_torch_from_public_vs_public,
                "torch_rope_from_native_norm_vs_native_route": rope_torch_from_native_vs_native,
                "torch_rope_from_native_norm_vs_native_route_scale": rope_torch_from_native_vs_native_scale,
                "comfy_rope_from_comfy_norm_vs_comfy_fused_route": rope_comfy_from_comfy_vs_fused,
                "comfy_rope_from_comfy_norm_vs_comfy_fused_route_scale": rope_comfy_from_comfy_vs_fused_scale,
                "torch_rope_vs_comfy_rope_from_same_comfy_norm": rope_torch_vs_comfy_from_comfy_norm,
                "torch_rope_vs_comfy_rope_from_same_comfy_norm_scale": rope_torch_vs_comfy_from_comfy_norm_scale,
                "comfy_rope_from_native_norm_vs_native_route": rope_comfy_from_native_vs_native,
                "comfy_rope_from_native_norm_vs_native_route_scale": rope_comfy_from_native_vs_native_scale,
                "fp32_rope_from_comfy_norm_vs_comfy_fused_route": rope_fp32_from_comfy_vs_fused,
                "fp32_rope_from_comfy_norm_vs_comfy_fused_route_scale": rope_fp32_from_comfy_vs_fused_scale,
                "native_fp32_rope_route_vs_comfy_route": native_fp32_rope_vs_comfy,
                "native_fp32_rope_route_vs_comfy_route_scale": native_fp32_rope_vs_comfy_scale,
                "native_fp32_rope_route_vs_native_route": native_fp32_rope_vs_native,
                "native_fp32_rope_route_vs_native_route_scale": native_fp32_rope_vs_native_scale,
                "fp32_rope_from_native_norm_vs_native_fp32_rope_route": rope_fp32_from_native_vs_native_fp32_route,
                "fp32_rope_from_native_norm_vs_native_fp32_rope_route_scale": rope_fp32_from_native_vs_native_fp32_route_scale,
            },
            "diagnostic_native_route": {
                "full_route_vs_comfy_route": native_route_vs_comfy,
                "full_route_vs_comfy_route_scale": native_route_vs_comfy_scale,
                "full_route_vs_public_route": native_route_vs_public,
                "full_route_vs_public_route_scale": native_route_vs_public_scale,
                "candidate_summary_vs_native_route_summary": k1_vs_native_route,
                "candidate_summary_vs_native_route_summary_scale": k1_vs_native_route_scale,
            },
            "timing_ms": k1_timing,
            "allocation": k1_allocation,
        },
        "k2": {
            "contract": K2_CONTRACT,
            "output": k2,
            "output_scale": k2_scale,
            "diagnostic_public_keyless_route": {
                "output_vs_dense_public_route": k2_vs_torch_route,
                "output_vs_dense_public_route_scale": k2_vs_torch_route_scale,
                "dense_public_route_vs_dense_comfy_route": dense_route_effect,
                "dense_public_route_vs_dense_comfy_route_scale": dense_route_effect_scale,
            },
            "diagnostic_public_rounding_route": {
                "dense_output_vs_public_route": dense_public_rounding_vs_public,
                "dense_output_vs_public_route_scale": dense_public_rounding_vs_public_scale,
            },
            "diagnostic_fp32_rope_route": {
                "dense_fp32_rope_route_vs_dense_comfy_route": native_fp32_rope_dense_vs_comfy,
                "dense_fp32_rope_route_vs_dense_comfy_route_scale": native_fp32_rope_dense_vs_comfy_scale,
            },
            "diagnostic_native_route": {
                "output_vs_dense_native_route": k2_vs_native_route,
                "output_vs_dense_native_route_scale": k2_vs_native_route_scale,
                "dense_native_route_vs_dense_comfy_route": native_dense_vs_comfy_dense,
                "dense_native_route_vs_dense_comfy_route_scale": native_dense_vs_comfy_dense_scale,
                "dense_native_sdpa_vs_fp32": native_sdpa_vs_fp32,
                "dense_native_sdpa_vs_fp32_scale": native_sdpa_vs_fp32_scale,
                "dense_comfy_sdpa_vs_fp32": comfy_sdpa_vs_fp32,
                "dense_comfy_sdpa_vs_fp32_scale": comfy_sdpa_vs_fp32_scale,
                "output_vs_dense_native_fp32": k2_vs_native_fp32,
                "output_vs_dense_native_fp32_scale": k2_vs_native_fp32_scale,
            },
            "timing_ms": k2_timing,
            "allocation": k2_allocation,
        },
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("teacher", "keyless"),
        required=True,
    )
    parser.add_argument("--keyless-repo")
    parser.add_argument("--expected-receipt-sha256")
    parser.add_argument(
        "--expected-probe-contract",
        help=(
            "fail before loading the capture/checkpoint unless this replay probe "
            "implements the requested diagnostic contract; safe for Patcher PR overlays"
        ),
    )
    parser.add_argument(
        "--output-json",
        help="atomically write the complete replay JSON to this path",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--q-start", type=int, default=0)
    parser.add_argument("--q-rows", type=int, default=257)
    parser.add_argument("--v-start", type=int, default=0)
    parser.add_argument("--v-rows", type=int, default=511)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--blocks", type=int, nargs="*", default=None)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "emit the complete JSON diagnostic and return success even when the "
            "frozen arithmetic envelope remains failed; all_blocks_pass and "
            "promotion_evidence are unchanged"
        ),
    )
    parser.add_argument(
        "--calibration-v2",
        action="store_true",
        help=(
            "run the predeclared threshold-free K1/K2 v2 real-H3 calibration "
            "campaign across boundary lengths and sequence regions"
        ),
    )
    args = parser.parse_args()

    probe_source = _probe_source_identity(_REPO_ROOT)
    probe_source["probe_contract"] = PROBE_CONTRACT
    if args.expected_probe_contract:
        expected_probe_contract = args.expected_probe_contract.strip()
        if PROBE_CONTRACT != expected_probe_contract:
            parser.error(
                "stale Sol-H3 replay probe: expected diagnostic contract "
                f"{expected_probe_contract!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the Sol #21 Patcher PR overlay before running the expensive replay"
            )

    if args.q_rows <= 0 or args.v_rows <= 0 or args.repeats <= 0:
        parser.error("q-rows, v-rows and repeats must be positive")
    if args.calibration_v2 and (
        K1_CONTRACT != V2_K1_CONTRACT or K2_CONTRACT != V2_K2_CONTRACT
    ):
        parser.error(
            "v2 calibration requires the corrected owning K1/K2 contracts; "
            f"got K1={K1_CONTRACT!r}, K2={K2_CONTRACT!r}. Refresh the complete "
            "Sol #17/#20/#21 Patcher stack before running calibration."
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("real-H3 replay requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        parser.error(f"real-H3 replay requires SM120, got {capability}")

    capture_path = Path(args.capture_bundle).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    keyless_repo = Path(args.keyless_repo) if args.keyless_repo else None
    records, provenance = _load_capture(
        capture_path,
        keyless_repo=keyless_repo,
        expected_receipt_sha256=args.expected_receipt_sha256,
        device=device,
    )
    checkpoint_sha256 = _sha256_file(checkpoint)

    receipt_path = capture_path.with_suffix(capture_path.suffix + ".receipt.json")
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    capture_bundle_sha256 = str(receipt_payload["bundle_sha256"]).lower()
    capture_receipt_sha256 = _sha256_file(receipt_path)

    if args.checkpoint_kind == "teacher":
        if checkpoint_sha256.lower() != provenance.teacher_model_sha256.lower():
            raise RuntimeError(
                "teacher checkpoint SHA-256 does not match the Stage-A capture provenance"
            )
    else:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        parent = metadata.get("teacher_model_sha256") or metadata.get(
            "parent_model_sha256"
        )
        if (
            not isinstance(parent, str)
            or parent.lower() != provenance.teacher_model_sha256.lower()
        ):
            raise RuntimeError(
                "Keyless checkpoint does not bind the same pinned teacher as the capture"
            )

    selected = (
        set(args.blocks)
        if args.blocks
        else {int(record.block_index) for record in records}
    )
    by_block = {int(record.block_index): record for record in records}
    missing = sorted(selected.difference(by_block))
    if missing:
        raise RuntimeError(f"capture bundle does not contain requested blocks: {missing}")

    if args.calibration_v2:
        calibration_blocks = []
        with torch.cuda.device(device), torch.inference_mode():
            for block in sorted(selected):
                calibration_blocks.append(
                    _calibration_v2_for_block(
                        by_block[block],
                        checkpoint=checkpoint,
                        checkpoint_kind=args.checkpoint_kind,
                        checkpoint_sha256=checkpoint_sha256,
                        device=device,
                    )
                )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result = {
            "contract": V2_CALIBRATION_CONTRACT,
            "mode": "k1-k2-v2-real-h3-calibration",
            "promotion_evidence": False,
            "thresholds_frozen": False,
            "historical_v1_envelope": envelope_dict(),
            "historical_v1_envelope_evaluated": False,
            "probe_source": probe_source,
            "checkpoint_kind": args.checkpoint_kind,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "capture_bundle": str(capture_path),
            "capture_bundle_sha256": capture_bundle_sha256,
            "capture_receipt": str(receipt_path),
            "capture_receipt_sha256": capture_receipt_sha256,
            "capture_provenance": {
                "code_commit": provenance.code_commit,
                "comfy_commit": provenance.comfy_commit,
                "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
                "execution_descriptor": provenance.execution_descriptor,
                "teacher_model_revision": provenance.teacher_model_revision,
                "teacher_model_sha256": provenance.teacher_model_sha256,
            },
            "source_contracts": {
                "k1": K1_CONTRACT,
                "k2": K2_CONTRACT,
            },
            "calibration_case_definitions": [
                {
                    "name": name,
                    "anchor": anchor,
                    "q_rows": q_rows,
                    "v_rows": v_rows,
                }
                for name, anchor, q_rows, v_rows in V2_CALIBRATION_CASES
            ],
            "blocks": calibration_blocks,
            "observed_extrema": _calibration_v2_extrema(calibration_blocks),
            "memory_strategy": {
                "capture_deserialize_device": str(device),
                "checkpoint_tensor_device": str(device),
                "large_file_hash_page_cache_policy": "posix_fadvise_dontneed_when_available",
                "capture_bundle_rehash_after_validated_load": False,
                "cuda_free_bytes_after_calibration": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
            },
            "limitations": [
                "calibration evidence only; no v2 threshold is frozen by this run",
                "one captured sigma-1 native-H3 forward, sampled across multiple sequence regions",
                "blocks 0/25/49 cover early/mid/late route-norm weights but not all 50 blocks",
                "K2 is all-selected experimental Triton arithmetic; sparse Sol selector and vendored CuTe production mainloop remain unwired",
                "no decoded-media, sampler, full-sequence, or end-to-end performance evidence",
            ],
        }
        if args.output_json:
            output_path = Path(args.output_json)
            _write_json_atomic(output_path, result)
            print(
                json.dumps(
                    {
                        "calibration_complete": True,
                        "diagnostic_report": str(output_path.expanduser().resolve()),
                        "probe_contract": PROBE_CONTRACT,
                        "probe_git_commit": probe_source["git_commit"],
                        "promotion_evidence": False,
                        "thresholds_frozen": False,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return

    results = []
    with torch.cuda.device(device), torch.inference_mode():
        for block in sorted(selected):
            results.append(
                _record_for_block(
                    by_block[block],
                    checkpoint=checkpoint,
                    checkpoint_kind=args.checkpoint_kind,
                    checkpoint_sha256=checkpoint_sha256,
                    device=device,
                    q_start=args.q_start,
                    q_rows=args.q_rows,
                    v_start=args.v_start,
                    v_rows=args.v_rows,
                    repeats=args.repeats,
                )
            )

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "contract": REPLAY_CONTRACT,
        "promotion_evidence": False,
        "probe_source": probe_source,
        "process_exit_policy": {
            "diagnostic_only": bool(args.diagnostic_only),
            "envelope_failure_is_process_error": not bool(args.diagnostic_only),
        },
        "arithmetic_envelope": envelope_dict(),
        "arithmetic_envelope_scope": (
            "frozen v1 synthetic envelope retained unchanged as a historical "
            "comparison; K1/K2 v2 require fresh calibration before promotion"
        ),
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "capture_bundle": str(capture_path),
        "capture_bundle_sha256": capture_bundle_sha256,
        "capture_receipt": str(receipt_path),
        "capture_receipt_sha256": capture_receipt_sha256,
        "capture_provenance": {
            "code_commit": provenance.code_commit,
            "comfy_commit": provenance.comfy_commit,
            "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
            "execution_descriptor": provenance.execution_descriptor,
            "teacher_model_revision": provenance.teacher_model_revision,
            "teacher_model_sha256": provenance.teacher_model_sha256,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "memory_strategy": {
            "capture_deserialize_device": str(device),
            "checkpoint_tensor_device": str(device),
            "large_file_hash_page_cache_policy": "posix_fadvise_dontneed_when_available",
            "capture_bundle_rehash_after_validated_load": False,
            "cuda_free_bytes_after_replay": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "results": results,
        "all_blocks_pass": all(bool(row["passes"]["all"]) for row in results),
        "limitations": [
            "bounded row windows from real Stage-A activations, not full-sequence attention",
            "K1/K2 experimental Triton primitives; vendored CuTe production mainloop is unchanged",
            "teacher mode uses native teacher K-norm as the Keyless route-norm initialization prior",
            "no sparse selector, decoded-media, sampler, or end-to-end performance evidence",
            "diagnostic public-Keyless-route comparisons do not alter the frozen v1 pass/fail envelope",
            "diagnostic native-route materialization and FP32 dense comparisons intentionally allocate oracle tensors and are never production paths",
            "public-rounding route diagnostic tests a BF16 intermediate-rounding hypothesis and does not change K1/K2 arithmetic",
            "RMSNorm isolation uses identity RoPE through the exact selected Comfy fused backend plus independent public/native RMS paths",
            "RoPE isolation holds each materialized BF16 normalization fixed before applying split-half rotation",
            "FP32-RoPE isolation established comfy-kitchen CUDA fused ComputeType=float semantics; K1/K2 v2 now use that rotation arithmetic",
            "the frozen v1 envelope is historical comparison evidence only for v2; fresh v2 calibration is required before promotion",
        ],
    }
    if args.output_json:
        output_path = Path(args.output_json)
        _write_json_atomic(output_path, result)
        print(
            json.dumps(
                {
                    "all_blocks_pass": bool(result["all_blocks_pass"]),
                    "diagnostic_report": str(output_path.expanduser().resolve()),
                    "probe_contract": PROBE_CONTRACT,
                    "probe_git_commit": probe_source["git_commit"],
                    "promotion_evidence": False,
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    if replay_requires_process_failure(
        all_blocks_pass=bool(result["all_blocks_pass"]),
        diagnostic_only=bool(args.diagnostic_only),
    ):
        raise RuntimeError(
            "real-H3 replay exceeded the predeclared arithmetic/allocation envelope"
        )


if __name__ == "__main__":
    main()
