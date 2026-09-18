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
from pathlib import Path
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    block_summary_oracle,
    checkpoint_tensor_names,
    envelope_dict,
    metric_within_limit,
    split_projection,
    tensor_metrics,
    value_sum_within_limit,
)
from sol_h3.keyless_route_summary import (  # noqa: E402
    CONTRACT as K1_CONTRACT,
    HEAD_DIM,
    NORM_EPS,
    ROPE_HALF_DIM,
    route_summary,
)


HEADS = 56
HIDDEN = 5376
REPLAY_CONTRACT = "sol-h3-keyless-real-h3-replay-v1"


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_capture(
    path: Path,
    *,
    keyless_repo: Path | None,
    expected_receipt_sha256: str | None,
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
    )


def _load_block_weights(checkpoint: Path, *, kind: str, block_index: int):
    names = checkpoint_tensor_names(kind, block_index)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
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

    route_bytes = int(v_raw.numel() * v_raw.element_size())
    del route, oracle_rc, oracle_vc, candidate_rc, candidate_vc, dense
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
            "raw_value_sum": k1_vc,
            "timing_ms": k1_timing,
            "allocation": k1_allocation,
        },
        "k2": {
            "contract": K2_CONTRACT,
            "output": k2,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--q-start", type=int, default=0)
    parser.add_argument("--q-rows", type=int, default=257)
    parser.add_argument("--v-start", type=int, default=0)
    parser.add_argument("--v-rows", type=int, default=511)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--blocks", type=int, nargs="*", default=None)
    args = parser.parse_args()

    if args.q_rows <= 0 or args.v_rows <= 0 or args.repeats <= 0:
        parser.error("q-rows, v-rows and repeats must be positive")
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
    )
    checkpoint_sha256 = _sha256_file(checkpoint)

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

    receipt_path = capture_path.with_suffix(capture_path.suffix + ".receipt.json")
    result = {
        "contract": REPLAY_CONTRACT,
        "promotion_evidence": False,
        "arithmetic_envelope": envelope_dict(),
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "capture_bundle": str(capture_path),
        "capture_bundle_sha256": _sha256_file(capture_path),
        "capture_receipt": str(receipt_path),
        "capture_receipt_sha256": _sha256_file(receipt_path),
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
        "results": results,
        "all_blocks_pass": all(bool(row["passes"]["all"]) for row in results),
        "limitations": [
            "bounded row windows from real Stage-A activations, not full-sequence attention",
            "K1/K2 experimental Triton primitives; vendored CuTe production mainloop is unchanged",
            "teacher mode uses native teacher K-norm as the Keyless route-norm initialization prior",
            "no sparse selector, decoded-media, sampler, or end-to-end performance evidence",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_blocks_pass"]:
        raise RuntimeError(
            "real-H3 replay exceeded the predeclared arithmetic/allocation envelope"
        )


if __name__ == "__main__":
    main()
