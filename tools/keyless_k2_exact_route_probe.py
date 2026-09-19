#!/usr/bin/env python3
"""Calibrate K2 v3 all-selected attention against the proven K1 route identity.

This is a threshold-free real-SM120 campaign. The candidate consumes already
positioned Q plus raw V and derives route(V) inside bounded N64 tiles. The oracle
materializes the exact K1 row-route arithmetic only on the diagnostic side, then
runs dense attention with raw-V retrieval.

No sparse selector is enabled and no production/provider promotion is implied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_TOOL_PATH = Path(__file__).resolve()
_REPO_ROOT = _TOOL_PATH.parents[1]
_DEFAULT_COMFY_ROOT = _TOOL_PATH.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
if _DEFAULT_COMFY_ROOT.joinpath("comfy").is_dir():
    sys.path.insert(0, str(_DEFAULT_COMFY_ROOT))

import torch  # noqa: E402

import keyless_real_h3_replay_probe as replay_fixture  # noqa: E402
from sol_h3.keyless_exact_attention import (  # noqa: E402
    CANONICAL_SCALE,
    CONTRACT as K2_CONTRACT,
    exact_attention,
)
from sol_h3.keyless_real_h3_replay import (  # noqa: E402
    K2_V3_ENVELOPE,
    k2_v3_envelope_dict,
    scale_aware_metric_within_limit,
    tensor_metrics,
    tensor_scale_diagnostics,
)
from sol_h3.keyless_route_summary import (  # noqa: E402
    NORM_EPS,
    SOL_REDUCTION_CONTRACT as K1_CONTRACT,
    materialized_sol_reduction_v4_route_diagnostic,
)


PROBE_CONTRACT = "sol-h3-keyless-k2-exact-route-probe-v2"
CALIBRATION_EVIDENCE_CONTRACT = "sol-h3-keyless-k2-exact-route-calibration-v1"
HOLDOUT_EVIDENCE_CONTRACT = "sol-h3-keyless-k2-exact-route-holdout-v1"
REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
REQUIRED_K2_CONTRACT = "sol-h3-keyless-exact-allselected-v3-k1-route-identity"
HEADS = 56
HIDDEN = 5376


V3_HOLDOUT_CASES = (
    ("eighth-95x95", 1, 8, 95, 95),
    ("three-eighth-96x96", 3, 8, 96, 96),
    ("five-eighth-97x97", 5, 8, 97, 97),
    ("seven-eighth-191x191", 7, 8, 191, 191),
    ("eighth-192x192", 1, 8, 192, 192),
    ("three-eighth-193x193", 3, 8, 193, 193),
    ("five-eighth-383x385", 5, 8, 383, 385),
    ("seven-eighth-385x383", 7, 8, 385, 383),
    ("five-eighth-769x1025", 5, 8, 769, 1025),
    ("three-eighth-1025x769", 3, 8, 1025, 769),
)


def _run_block(
    record,
    *,
    checkpoint: Path,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, object]:
    if record.case.rope_freqs is None or not torch.is_tensor(record.case.rope_freqs):
        raise RuntimeError(f"capture block {record.block_index} has no exact H3 RoPE")
    if record.attention_input.ndim != 2 or record.attention_input.shape[1] != HIDDEN:
        raise RuntimeError("K2 capture attention input has invalid geometry")

    q_weight, v_weight, q_norm, route_norm, metadata, names = (
        replay_fixture._load_block_weights(
            checkpoint,
            kind=checkpoint_kind,
            block_index=record.block_index,
            device=device,
        )
    )
    total_rows = int(record.attention_input.shape[0])
    cases: list[dict[str, object]] = []

    for name, anchor, q_rows, v_rows in replay_fixture.V2_CALIBRATION_CASES:
        span = max(q_rows, v_rows)
        start = replay_fixture._resolve_calibration_start(total_rows, span, anchor)

        q_raw = replay_fixture._project(
            record.attention_input,
            q_weight,
            start=start,
            rows=q_rows,
            device=device,
        ).view(q_rows, HEADS, -1)
        v_raw = replay_fixture._project(
            record.attention_input,
            v_weight,
            start=start,
            rows=v_rows,
            device=device,
        ).view(v_rows, HEADS, -1)
        q_rope = replay_fixture._rope_slice(
            record.case.rope_freqs,
            start=start,
            rows=q_rows,
            device=device,
        )
        v_rope = replay_fixture._rope_slice(
            record.case.rope_freqs,
            start=start,
            rows=v_rows,
            device=device,
        )
        q = replay_fixture._comfy_position(q_raw, q_norm, q_rope)
        comfy_route = replay_fixture._comfy_position(v_raw, route_norm, v_rope)

        # Diagnostic/oracle only. K1 v6 intentionally preserves this exact row
        # arithmetic before its bounded BF16 spill/reload reduction boundary.
        k1_route = materialized_sol_reduction_v4_route_diagnostic(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        dense_k1 = replay_fixture._dense_attention(q, k1_route, v_raw)
        dense_comfy = replay_fixture._dense_attention(q, comfy_route, v_raw)

        out = torch.empty_like(q)
        exact_attention(
            q,
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
            scale=CANONICAL_SCALE,
            out=out,
        )
        torch.cuda.synchronize(device)

        allocation = replay_fixture._candidate_allocation(
            lambda: exact_attention(
                q,
                v_raw,
                route_norm.to(device=device, dtype=torch.bfloat16),
                NORM_EPS,
                v_rope,
                scale=CANONICAL_SCALE,
                out=out,
            ),
            device,
        )
        full_route_bytes = int(v_raw.numel() * v_raw.element_size())
        allocation["full_materialized_route_bytes"] = full_route_bytes
        allocation["below_full_route"] = (
            allocation["temporary_peak_delta"] < full_route_bytes
        )

        timing = None
        if name == "tail-1025x1537":
            timing = replay_fixture._timings(
                lambda: exact_attention(
                    q,
                    v_raw,
                    route_norm.to(device=device, dtype=torch.bfloat16),
                    NORM_EPS,
                    v_rope,
                    scale=CANONICAL_SCALE,
                    out=out,
                ),
                device,
                7,
            )

        cases.append(
            {
                "name": name,
                "anchor": anchor,
                "start": start,
                "q_rows": q_rows,
                "v_rows": v_rows,
                "candidate_vs_dense_k1_route": tensor_metrics(out, dense_k1),
                "candidate_vs_dense_k1_route_scale": tensor_scale_diagnostics(
                    out,
                    dense_k1,
                ),
                "candidate_vs_dense_comfy_route": tensor_metrics(out, dense_comfy),
                "candidate_vs_dense_comfy_route_scale": tensor_scale_diagnostics(
                    out,
                    dense_comfy,
                ),
                "k1_route_vs_comfy_route": tensor_metrics(k1_route, comfy_route),
                "k1_route_vs_comfy_route_scale": tensor_scale_diagnostics(
                    k1_route,
                    comfy_route,
                ),
                "dense_k1_route_vs_dense_comfy_route": tensor_metrics(
                    dense_k1,
                    dense_comfy,
                ),
                "dense_k1_route_vs_dense_comfy_route_scale": (
                    tensor_scale_diagnostics(dense_k1, dense_comfy)
                ),
                "allocation": allocation,
                "timing_ms": timing,
            }
        )

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



def _run_holdout_block(
    record,
    *,
    checkpoint: Path,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate the frozen K2-v3 envelope on disjoint deterministic windows."""
    if record.case.rope_freqs is None or not torch.is_tensor(record.case.rope_freqs):
        raise RuntimeError(f"capture block {record.block_index} has no exact H3 RoPE")
    if record.attention_input.ndim != 2 or record.attention_input.shape[1] != HIDDEN:
        raise RuntimeError("K2 holdout capture attention input has invalid geometry")

    q_weight, v_weight, q_norm, route_norm, metadata, names = (
        replay_fixture._load_block_weights(
            checkpoint,
            kind=checkpoint_kind,
            block_index=record.block_index,
            device=device,
        )
    )
    total_rows = int(record.attention_input.shape[0])
    cases: list[dict[str, object]] = []

    for name, numerator, denominator, q_rows, v_rows in V3_HOLDOUT_CASES:
        span = max(q_rows, v_rows)
        start = replay_fixture._resolve_fractional_start(
            total_rows,
            span,
            numerator,
            denominator,
        )
        q_raw = replay_fixture._project(
            record.attention_input,
            q_weight,
            start=start,
            rows=q_rows,
            device=device,
        ).view(q_rows, HEADS, -1)
        v_raw = replay_fixture._project(
            record.attention_input,
            v_weight,
            start=start,
            rows=v_rows,
            device=device,
        ).view(v_rows, HEADS, -1)
        q_rope = replay_fixture._rope_slice(
            record.case.rope_freqs,
            start=start,
            rows=q_rows,
            device=device,
        )
        v_rope = replay_fixture._rope_slice(
            record.case.rope_freqs,
            start=start,
            rows=v_rows,
            device=device,
        )
        q = replay_fixture._comfy_position(q_raw, q_norm, q_rope)
        comfy_route = replay_fixture._comfy_position(v_raw, route_norm, v_rope)
        k1_route = materialized_sol_reduction_v4_route_diagnostic(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        dense_k1 = replay_fixture._dense_attention(q, k1_route, v_raw)
        dense_comfy = replay_fixture._dense_attention(q, comfy_route, v_raw)

        out = torch.empty_like(q)
        exact_attention(
            q,
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
            scale=CANONICAL_SCALE,
            out=out,
        )
        torch.cuda.synchronize(device)

        primary = tensor_metrics(out, dense_k1)
        primary_scale = tensor_scale_diagnostics(out, dense_k1)
        allocation = replay_fixture._candidate_allocation(
            lambda: exact_attention(
                q,
                v_raw,
                route_norm.to(device=device, dtype=torch.bfloat16),
                NORM_EPS,
                v_rope,
                scale=CANONICAL_SCALE,
                out=out,
            ),
            device,
        )
        full_route_bytes = int(v_raw.numel() * v_raw.element_size())
        allocation["full_materialized_route_bytes"] = full_route_bytes
        allocation["below_full_route"] = (
            allocation["temporary_peak_delta"] < full_route_bytes
        )

        passes = {
            "k2_output": scale_aware_metric_within_limit(
                primary,
                primary_scale,
                K2_V3_ENVELOPE.k2_output,
            ),
            "allocation": bool(allocation["below_full_route"]),
        }
        passes["all"] = all(passes.values())

        cases.append(
            {
                "name": name,
                "fraction": {
                    "numerator": numerator,
                    "denominator": denominator,
                },
                "start": start,
                "q_rows": q_rows,
                "v_rows": v_rows,
                "candidate_vs_dense_k1_route": primary,
                "candidate_vs_dense_k1_route_scale": primary_scale,
                "candidate_vs_dense_comfy_route": tensor_metrics(out, dense_comfy),
                "candidate_vs_dense_comfy_route_scale": tensor_scale_diagnostics(
                    out,
                    dense_comfy,
                ),
                "k1_route_vs_comfy_route": tensor_metrics(k1_route, comfy_route),
                "k1_route_vs_comfy_route_scale": tensor_scale_diagnostics(
                    k1_route,
                    comfy_route,
                ),
                "dense_k1_route_vs_dense_comfy_route": tensor_metrics(
                    dense_k1,
                    dense_comfy,
                ),
                "dense_k1_route_vs_dense_comfy_route_scale": (
                    tensor_scale_diagnostics(dense_k1, dense_comfy)
                ),
                "allocation": allocation,
                "timing_ms": None,
                "passes": passes,
            }
        )

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
        "all_cases_pass": all(bool(case["passes"]["all"]) for case in cases),
    }

def _extrema(blocks: list[dict[str, object]]) -> dict[str, object]:
    flat = [
        {"block_index": int(block["block_index"]), **case}
        for block in blocks
        for case in block["cases"]
    ]

    def max_metric(path: tuple[str, ...], field: str) -> dict[str, object]:
        def obj(item):
            value = item
            for part in path:
                value = value[part]
            return value

        row = max(flat, key=lambda item: float(obj(item)[field]))
        return {
            "value": float(obj(row)[field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def max_scale_ratio(path: tuple[str, ...], field: str) -> dict[str, object]:
        def obj(item):
            value = item
            for part in path:
                value = value[part]
            return value

        eligible = [
            item
            for item in flat
            if obj(item).get(field) is not None
        ]
        row = max(eligible, key=lambda item: float(obj(item)[field]))
        return {
            "value": float(obj(row)[field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def max_worst_ulps(path: tuple[str, ...]) -> dict[str, object]:
        def obj(item):
            value = item
            for part in path:
                value = value[part]
            return value

        eligible = [
            item
            for item in flat
            if isinstance(obj(item).get("worst"), dict)
            and obj(item)["worst"].get("abs_error_in_want_bf16_ulps") is not None
        ]
        row = max(
            eligible,
            key=lambda item: float(
                obj(item)["worst"]["abs_error_in_want_bf16_ulps"]
            ),
        )
        return {
            "value": float(
                obj(row)["worst"]["abs_error_in_want_bf16_ulps"]
            ),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    return {
        "case_count": len(flat),
        "candidate_vs_dense_k1_route": {
            "rel_l2": max_metric(("candidate_vs_dense_k1_route",), "rel_l2"),
            "mean_abs": max_metric(("candidate_vs_dense_k1_route",), "mean_abs"),
            "max_abs": max_metric(("candidate_vs_dense_k1_route",), "max_abs"),
            "max_abs_over_want_abs_max": max_scale_ratio(
                ("candidate_vs_dense_k1_route_scale",),
                "max_abs_over_want_abs_max",
            ),
            "mean_abs_over_want_mean_abs": max_scale_ratio(
                ("candidate_vs_dense_k1_route_scale",),
                "mean_abs_over_want_mean_abs",
            ),
            "worst_bf16_ulps": max_worst_ulps(
                ("candidate_vs_dense_k1_route_scale",)
            ),
        },
        "candidate_vs_dense_comfy_route": {
            "rel_l2": max_metric(("candidate_vs_dense_comfy_route",), "rel_l2"),
            "max_abs": max_metric(("candidate_vs_dense_comfy_route",), "max_abs"),
        },
        "k1_route_vs_comfy_route": {
            "rel_l2": max_metric(("k1_route_vs_comfy_route",), "rel_l2"),
            "max_abs": max_metric(("k1_route_vs_comfy_route",), "max_abs"),
        },
        "dense_k1_route_vs_dense_comfy_route": {
            "rel_l2": max_metric(
                ("dense_k1_route_vs_dense_comfy_route",),
                "rel_l2",
            ),
            "max_abs": max_metric(
                ("dense_k1_route_vs_dense_comfy_route",),
                "max_abs",
            ),
        },
        "allocation": {
            "all_below_full_route": all(
                bool(item["allocation"]["below_full_route"])
                for item in flat
            ),
            "max_temporary_peak_delta": max(
                int(item["allocation"]["temporary_peak_delta"])
                for item in flat
            ),
        },
        "tail_timing_ms": [
            {
                "block_index": int(item["block_index"]),
                "case": str(item["name"]),
                **item["timing_ms"],
            }
            for item in flat
            if item["timing_ms"] is not None
        ],
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
    parser.add_argument("--expected-probe-contract")
    parser.add_argument("--output-json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, nargs="*", default=(0, 25, 49))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--calibration-v3",
        action="store_true",
        help="rerun the threshold-free K2-v3 calibration geometry",
    )
    mode.add_argument(
        "--holdout-v3",
        action="store_true",
        help="evaluate the frozen K2-v3 envelope on disjoint windows",
    )
    args = parser.parse_args()

    source = replay_fixture._probe_source_identity(_REPO_ROOT)
    source["probe_contract"] = PROBE_CONTRACT
    source["critical_source_sha256"] = {
        rel: replay_fixture._sha256_file(_REPO_ROOT / rel)
        for rel in (
            "tools/keyless_k2_exact_route_probe.py",
            "tools/keyless_real_h3_replay_probe.py",
            "sol_h3/keyless_exact_attention.py",
            "sol_h3/keyless_route_summary.py",
            "sol_h3/keyless_real_h3_replay.py",
        )
    }
    if args.expected_probe_contract:
        expected = args.expected_probe_contract.strip()
        if expected != PROBE_CONTRACT:
            parser.error(
                "stale Sol-H3 K2 exact-route probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the complete Patcher stack through the K2 PR"
            )
    if source["tracked_worktree_dirty"]:
        parser.error("K2 exact-route evidence requires a clean tracked worktree")
    if K1_CONTRACT != REQUIRED_K1_CONTRACT:
        parser.error(f"unexpected K1 contract {K1_CONTRACT!r}")
    if K2_CONTRACT != REQUIRED_K2_CONTRACT:
        parser.error(f"unexpected K2 contract {K2_CONTRACT!r}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("K2 exact-route evidence requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error(
            "K2 exact-route evidence requires SM120, got "
            f"{torch.cuda.get_device_capability(device)}"
        )

    capture_path = Path(args.capture_bundle).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    keyless_repo = Path(args.keyless_repo) if args.keyless_repo else None
    records, provenance = replay_fixture._load_capture(
        capture_path,
        keyless_repo=keyless_repo,
        expected_receipt_sha256=args.expected_receipt_sha256,
        device=device,
    )
    checkpoint_sha256 = replay_fixture._sha256_file(checkpoint)

    receipt_path = capture_path.with_suffix(capture_path.suffix + ".receipt.json")
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    capture_bundle_sha256 = str(receipt_payload["bundle_sha256"]).lower()
    capture_receipt_sha256 = replay_fixture._sha256_file(receipt_path)

    if args.checkpoint_kind == "teacher":
        if checkpoint_sha256.lower() != provenance.teacher_model_sha256.lower():
            raise RuntimeError(
                "teacher checkpoint SHA-256 does not match capture provenance"
            )
    else:
        from safetensors import safe_open

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
                "Keyless checkpoint does not bind the capture's pinned teacher"
            )

    selected = set(args.blocks)
    by_block = {int(record.block_index): record for record in records}
    missing = sorted(selected.difference(by_block))
    if missing:
        raise RuntimeError(f"capture bundle lacks requested K2 blocks: {missing}")

    blocks = []
    with torch.cuda.device(device), torch.inference_mode():
        for block in sorted(selected):
            if args.holdout_v3:
                blocks.append(
                    _run_holdout_block(
                        by_block[block],
                        checkpoint=checkpoint,
                        checkpoint_kind=args.checkpoint_kind,
                        checkpoint_sha256=checkpoint_sha256,
                        device=device,
                    )
                )
            else:
                blocks.append(
                    _run_block(
                        by_block[block],
                        checkpoint=checkpoint,
                        checkpoint_kind=args.checkpoint_kind,
                        checkpoint_sha256=checkpoint_sha256,
                        device=device,
                    )
                )

    holdout_pass = (
        all(bool(block["all_cases_pass"]) for block in blocks)
        if args.holdout_v3
        else None
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if args.holdout_v3:
        contract = HOLDOUT_EVIDENCE_CONTRACT
        mode_name = "keyless-k2-exact-route-holdout"
        thresholds_frozen = True
        case_definitions = [
            {
                "name": name,
                "fraction": {
                    "numerator": numerator,
                    "denominator": denominator,
                },
                "q_rows": q_rows,
                "v_rows": v_rows,
            }
            for name, numerator, denominator, q_rows, v_rows
            in V3_HOLDOUT_CASES
        ]
    else:
        contract = CALIBRATION_EVIDENCE_CONTRACT
        mode_name = "keyless-k2-exact-route-calibration"
        thresholds_frozen = False
        case_definitions = [
            {
                "name": name,
                "anchor": anchor,
                "q_rows": q_rows,
                "v_rows": v_rows,
            }
            for name, anchor, q_rows, v_rows in replay_fixture.V2_CALIBRATION_CASES
        ]

    result = {
        "contract": contract,
        "mode": mode_name,
        "promotion_evidence": False,
        "thresholds_frozen": thresholds_frozen,
        "probe_source": source,
        "source_contracts": {
            "k1": K1_CONTRACT,
            "k2": K2_CONTRACT,
        },
        "evidence_lineage": {
            "k1_v9_sha256": (
                "d824f51b86e3dea96296606cf27c8c46514f49c8b920cda640e9ed77f92732ba"
            ),
            "k2_v3_calibration_sha256": (
                "65ff8ccd679a7f5c2e1c82d59d98068167b94922ab0bc3f75e6f636363ebc306"
            ),
            "historical_k2_v2_calibration_sha256": (
                "6cb5996e98613563947804d22c9a59cf11e8df58f1da4e566f3141709218531d"
            ),
            "historical_k2_v2_holdout_sha256": (
                "dca535a508f8c7045a1e285b4afe0f3e3ec1c84c47d082f8f43cb473af874fb1"
            ),
        },
        "v3_envelope": k2_v3_envelope_dict() if args.holdout_v3 else None,
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
        "case_definitions": case_definitions,
        "blocks": blocks,
        "observed_extrema": _extrema(blocks),
        "all_holdout_cases_pass": holdout_pass,
        "memory_strategy": {
            "capture_deserialize_device": str(device),
            "checkpoint_tensor_device": str(device),
            "capture_bundle_rehash_after_validated_load": False,
            "large_file_hash_page_cache_policy": (
                "posix_fadvise_dontneed_when_available"
            ),
            "cuda_free_bytes_after_evidence": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "limitations": [
            (
                "frozen K2-v3 arithmetic holdout only; no provider promotion"
                if args.holdout_v3
                else "threshold-free K2-v3 calibration only"
            ),
            "the exact K1 route is materialized only on the diagnostic/oracle side",
            "all blocks are selected; no K3 sparse selector is active in this probe",
            "K2 v3 is experimental and not connected to production provider dispatch",
            "one sigma-1 real-H3 capture and blocks 0/25/49 only",
            "no vendored CuTe K2 mainloop, decoded-media, sampler, or end-to-end evidence",
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        replay_fixture._write_json_atomic(output_path, result)
        print(
            json.dumps(
                {
                    "all_holdout_cases_pass": holdout_pass,
                    "evidence_complete": True,
                    "diagnostic_report": str(output_path.expanduser().resolve()),
                    "probe_contract": PROBE_CONTRACT,
                    "probe_git_commit": source["git_commit"],
                    "promotion_evidence": False,
                    "thresholds_frozen": thresholds_frozen,
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

    if args.holdout_v3 and not bool(holdout_pass):
        raise RuntimeError("real-H3 K2-v3 holdout exceeded the frozen arithmetic envelope")


if __name__ == "__main__":
    main()
