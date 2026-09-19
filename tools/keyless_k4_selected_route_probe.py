#!/usr/bin/env python3
"""Calibrate K4 selected-route execution with the already-proven K3 route mask.

This probe is an integration boundary, not provider promotion.  The materialized
oracle obtains the released Sol route trace from exact K1 route rows.  The K4
candidate receives that exact trace plus K1-v6 RC/VC and derives only selected
V64 route tiles from raw V.  No global candidate route tensor is created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_TOOL_PATH = Path(__file__).resolve()
_REPO_ROOT = _TOOL_PATH.parents[1]
_DEFAULT_COMFY_ROOT = _TOOL_PATH.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
if _DEFAULT_COMFY_ROOT.joinpath("comfy").is_dir():
    sys.path.insert(0, str(_DEFAULT_COMFY_ROOT))

import torch  # noqa: E402

import keyless_k3_selector_holdout as k3_holdout  # noqa: E402
import keyless_k3_selector_probe as k3_probe  # noqa: E402
import keyless_real_h3_replay_probe as replay_fixture  # noqa: E402
from sol_h3.keyless_real_h3_replay import (  # noqa: E402
    tensor_metrics,
    tensor_scale_diagnostics,
)
from sol_h3.keyless_route_summary import (  # noqa: E402
    NORM_EPS,
    SOL_REDUCTION_CONTRACT as K1_CONTRACT,
    materialized_sol_reduction_v4_route_diagnostic,
    route_summary_sol_reduction,
)
from sol_h3.keyless_selector import (  # noqa: E402
    CONTRACT as K3_CONTRACT,
    run_materialized_exact_selector_isolation,
    threshold_from_route_centroids,
)
from sol_h3.keyless_sparse_composition import (  # noqa: E402
    CONTRACT as K4_CONTRACT,
    selected_route_composition,
)


PROBE_CONTRACT = "sol-h3-keyless-k4-selected-route-probe-v1"
EVIDENCE_CONTRACT = "sol-h3-keyless-k4-selected-route-calibration-v1"
REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
REQUIRED_K3_CONTRACT = "sol-h3-keyless-selector-k3-v2-k1-route-identity"
REQUIRED_K4_CONTRACT = "sol-h3-keyless-k4-selected-route-composition-v1"
K3_HOLDOUT_SHA256 = (
    "ac2137f430251e77847bdce74ee9c1e57cdb9765043c7a0692ececc1688e79ce"
)
K3_CALIBRATION_SHA256 = (
    "ea2daae99e6c3df515b9d8a03a9fb4b8b2920601d843b7363fb63309d7eff836"
)
K2_HOLDOUT_SHA256 = (
    "61267ae96c3a557f02fee2dd5e7747e7a2ba296506d7a17f11dd0300e59db18e"
)
K1_V9_SHA256 = (
    "d824f51b86e3dea96296606cf27c8c46514f49c8b920cda640e9ed77f92732ba"
)
SCALE = 128 ** -0.5
HEADS = 56
HIDDEN = 5376


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

    from sol_h3._vendor.sol_attn.preprocess import prepare

    for (
        name,
        numerator,
        denominator,
        q_rows,
        v_rows,
        sink_start,
        sink_tokens,
    ) in k3_holdout.K3_V2_HOLDOUT_CASES:
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
        k1_route = materialized_sol_reduction_v4_route_diagnostic(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        qb = q.unsqueeze(0).contiguous()
        kb = k1_route.unsqueeze(0).contiguous()
        vb = v_raw.unsqueeze(0).contiguous()

        candidate_rc, candidate_vc = route_summary_sol_reduction(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        candidate_rc_b = candidate_rc.unsqueeze(0).contiguous()
        candidate_vc_b = candidate_vc.unsqueeze(0).contiguous()
        candidate_threshold = threshold_from_route_centroids(
            qb,
            candidate_rc_b,
            kv_rows=v_rows,
            tau=1.0,
            scale=SCALE,
        )
        reference_rc, reference_vc, reference_threshold = prepare(
            qb,
            kb,
            vb,
            tau=1.0,
            scale=SCALE,
            thresh_type="diag",
            valid_tokens=q_rows,
            valid_kv_tokens=v_rows,
        )
        reference_output, reference_trace = run_materialized_exact_selector_isolation(
            qb,
            kb,
            vb,
            reference_rc,
            reference_vc,
            reference_threshold,
            scale=SCALE,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        candidate_output, bounded = selected_route_composition(
            q,
            v_raw,
            candidate_rc,
            candidate_vc,
            reference_trace,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
            scale=SCALE,
        )
        torch.cuda.synchronize(device)
        candidate_ms = (time.perf_counter() - started) * 1000.0

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
                "sink_start": sink_start,
                "sink_tokens": sink_tokens,
                "route_centroid": tensor_metrics(candidate_rc_b, reference_rc),
                "raw_value_sum": tensor_metrics(candidate_vc_b, reference_vc),
                "threshold": tensor_metrics(
                    candidate_threshold,
                    reference_threshold,
                ),
                "candidate_vs_materialized_sol": tensor_metrics(
                    candidate_output.unsqueeze(0),
                    reference_output,
                ),
                "candidate_vs_materialized_sol_scale": tensor_scale_diagnostics(
                    candidate_output.unsqueeze(0),
                    reference_output,
                ),
                "bounded_route_storage": bounded,
                "timing_ms": {
                    "candidate_diagnostic": candidate_ms,
                    "note": (
                        "Python-orchestrated K4 composition timing is diagnostic "
                        "only and is not production performance evidence"
                    ),
                },
            }
        )

        del (
            q_raw,
            v_raw,
            q_rope,
            v_rope,
            q,
            k1_route,
            qb,
            kb,
            vb,
            candidate_rc,
            candidate_vc,
            candidate_rc_b,
            candidate_vc_b,
            candidate_threshold,
            reference_rc,
            reference_vc,
            reference_threshold,
            reference_output,
            reference_trace,
            candidate_output,
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


def _extrema(blocks: list[dict[str, object]]) -> dict[str, object]:
    flat = [
        {"block_index": int(block["block_index"]), **case}
        for block in blocks
        for case in block["cases"]
    ]
    if not flat:
        raise RuntimeError("K4 calibration produced no cases")

    def maximum(metric: str, field: str) -> dict[str, object]:
        row = max(flat, key=lambda item: float(item[metric][field]))
        return {
            "value": float(row[metric][field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def max_scale(field: str) -> dict[str, object]:
        eligible = [
            item
            for item in flat
            if item["candidate_vs_materialized_sol_scale"].get(field) is not None
        ]
        row = max(
            eligible,
            key=lambda item: float(
                item["candidate_vs_materialized_sol_scale"][field]
            ),
        )
        return {
            "value": float(
                row["candidate_vs_materialized_sol_scale"][field]
            ),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    ulp_rows = [
        item
        for item in flat
        if item["candidate_vs_materialized_sol_scale"].get("worst", {}).get(
            "abs_error_in_want_bf16_ulps"
        ) is not None
    ]
    worst_ulp = max(
        ulp_rows,
        key=lambda item: float(
            item["candidate_vs_materialized_sol_scale"]["worst"][
                "abs_error_in_want_bf16_ulps"
            ]
        ),
    )
    return {
        "case_count": len(flat),
        "route_centroid_max_abs": maximum("route_centroid", "max_abs"),
        "raw_value_sum_max_abs": maximum("raw_value_sum", "max_abs"),
        "threshold_max_abs": maximum("threshold", "max_abs"),
        "candidate_vs_materialized_sol": {
            "rel_l2": maximum("candidate_vs_materialized_sol", "rel_l2"),
            "mean_abs": maximum("candidate_vs_materialized_sol", "mean_abs"),
            "max_abs": maximum("candidate_vs_materialized_sol", "max_abs"),
            "max_abs_over_want_abs_max": max_scale(
                "max_abs_over_want_abs_max"
            ),
            "mean_abs_over_want_mean_abs": max_scale(
                "mean_abs_over_want_mean_abs"
            ),
            "worst_bf16_ulps": {
                "value": float(
                    worst_ulp["candidate_vs_materialized_sol_scale"]["worst"][
                        "abs_error_in_want_bf16_ulps"
                    ]
                ),
                "block_index": int(worst_ulp["block_index"]),
                "case": str(worst_ulp["name"]),
            },
        },
        "bounded_route_storage": {
            "all_below_full_route": all(
                int(item["bounded_route_storage"]["max_route_tile_bytes"])
                < int(
                    item["bounded_route_storage"][
                        "full_materialized_route_bytes"
                    ]
                )
                for item in flat
            ),
            "maximum_route_tile_fraction": max(
                float(item["bounded_route_storage"]["max_route_tile_fraction"])
                for item in flat
            ),
        },
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
    args = parser.parse_args()

    source = replay_fixture._probe_source_identity(_REPO_ROOT)
    source["probe_contract"] = PROBE_CONTRACT
    source["critical_source_sha256"] = {
        rel: replay_fixture._sha256_file(_REPO_ROOT / rel)
        for rel in (
            "tools/keyless_k4_selected_route_probe.py",
            "sol_h3/keyless_sparse_composition.py",
            "sol_h3/keyless_route_summary.py",
            "sol_h3/keyless_selector.py",
        )
    }
    if args.expected_probe_contract:
        expected = args.expected_probe_contract.strip()
        if expected != PROBE_CONTRACT:
            parser.error(
                "stale K4 selected-route probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the complete Patcher stack through the K4 PR"
            )
    if source["tracked_worktree_dirty"]:
        parser.error("K4 selected-route evidence requires a clean tracked worktree")
    if K1_CONTRACT != REQUIRED_K1_CONTRACT:
        parser.error(f"unexpected K1 contract {K1_CONTRACT!r}")
    if K3_CONTRACT != REQUIRED_K3_CONTRACT:
        parser.error(f"unexpected K3 contract {K3_CONTRACT!r}")
    if K4_CONTRACT != REQUIRED_K4_CONTRACT:
        parser.error(f"unexpected K4 composition contract {K4_CONTRACT!r}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("K4 selected-route evidence requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error("K4 selected-route evidence requires SM120")

    capture_path = Path(args.capture_bundle).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    records, provenance = replay_fixture._load_capture(
        capture_path,
        keyless_repo=Path(args.keyless_repo) if args.keyless_repo else None,
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

    selected = set(args.blocks)
    by_block = {int(record.block_index): record for record in records}
    missing = sorted(selected.difference(by_block))
    if missing:
        raise RuntimeError(f"capture bundle lacks requested K4 blocks: {missing}")

    blocks = []
    with torch.cuda.device(device), torch.inference_mode():
        for block in sorted(selected):
            blocks.append(
                _run_block(
                    by_block[block],
                    checkpoint=checkpoint,
                    checkpoint_kind=args.checkpoint_kind,
                    checkpoint_sha256=checkpoint_sha256,
                    device=device,
                )
            )

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "contract": EVIDENCE_CONTRACT,
        "mode": "keyless-k4-selected-route-calibration",
        "promotion_evidence": False,
        "thresholds_frozen": False,
        "probe_source": source,
        "source_contracts": {
            "k1": K1_CONTRACT,
            "k3": K3_CONTRACT,
            "k4": K4_CONTRACT,
        },
        "evidence_lineage": {
            "k1_v9_sha256": K1_V9_SHA256,
            "k2_v3_holdout_sha256": K2_HOLDOUT_SHA256,
            "k3_v2_calibration_sha256": K3_CALIBRATION_SHA256,
            "k3_v2_holdout_sha256": K3_HOLDOUT_SHA256,
        },
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
        "case_definitions": [
            {
                "name": name,
                "fraction": {
                    "numerator": numerator,
                    "denominator": denominator,
                },
                "q_rows": q_rows,
                "v_rows": v_rows,
                "sink_start": sink_start,
                "sink_tokens": sink_tokens,
            }
            for (
                name,
                numerator,
                denominator,
                q_rows,
                v_rows,
                sink_start,
                sink_tokens,
            ) in k3_holdout.K3_V2_HOLDOUT_CASES
        ],
        "blocks": blocks,
        "observed_extrema": _extrema(blocks),
        "memory_strategy": {
            "capture_deserialize_device": str(device),
            "checkpoint_tensor_device": str(device),
            "candidate_route_storage": "one selected V64 route tile at a time",
            "candidate_global_route_tensor": False,
            "cuda_free_bytes_after_evidence": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "limitations": [
            "K4 selected-route composition calibration only; no acceptance envelope is frozen",
            "the route trace is supplied by the already-proven materialized K3 oracle",
            "the candidate does not yet own selector generation",
            "Python/Torch orchestration is diagnostic and not production-performance evidence",
            "the full exact K1 route exists only on the oracle side",
            "no vendored CuTe source is changed by this diagnostic layer",
            "no provider dispatch or fused receipt/history identity is enabled",
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        replay_fixture._write_json_atomic(output_path, result)
        print(
            json.dumps(
                {
                    "diagnostic_report": str(output_path.expanduser().resolve()),
                    "evidence_complete": True,
                    "probe_contract": PROBE_CONTRACT,
                    "probe_git_commit": source["git_commit"],
                    "promotion_evidence": False,
                    "thresholds_frozen": False,
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
