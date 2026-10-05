#!/usr/bin/env python3
"""Evaluate the frozen exact K3-v2 selector gate on unseen real-H3 windows.

The K3-v2 calibration established exact equality for routed centroids, raw value
sums, thresholds, route ballots, and sparse outputs when candidate K1-v6
summaries are compared with the exact K1-route materialized reference.

This holdout freezes that exact-equality gate before evaluating four deterministic
windows that are disjoint from the K3-v2 calibration windows. The full exact K1
route remains diagnostic/reference-only; this is not K4 or provider promotion.
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

import keyless_k3_selector_probe as calibration_probe  # noqa: E402
import keyless_real_h3_replay_probe as replay_fixture  # noqa: E402
from sol_h3.keyless_real_h3_replay import (  # noqa: E402
    tensor_metrics,
    tensor_scale_diagnostics,
)
from sol_h3.keyless_route_summary import (  # noqa: E402
    HEAD_DIM,
    NORM_EPS,
    SOL_REDUCTION_CONTRACT as K1_CONTRACT,
    materialized_sol_reduction_v4_route_diagnostic,
    route_summary_sol_reduction,
)
from sol_h3.keyless_selector import (  # noqa: E402
    CONTRACT as K3_CONTRACT,
    route_trace_metrics,
    run_materialized_exact_selector_isolation,
    sink_block_range,
    threshold_from_route_centroids,
)


PROBE_CONTRACT = "sol-h3-keyless-k3-selector-holdout-probe-v1"
EVIDENCE_CONTRACT = "sol-h3-keyless-k3-selector-holdout-v1"
EXACT_GATE_CONTRACT = "sol-h3-keyless-k3-v2-exact-gate-v1"
CALIBRATION_EVIDENCE_SHA256 = (
    "ea2daae99e6c3df515b9d8a03a9fb4b8b2920601d843b7363fb63309d7eff836"
)
REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
REQUIRED_K3_CONTRACT = "sol-h3-keyless-selector-k3-v2-k1-route-identity"
K1_V9_EVIDENCE_SHA256 = (
    "d824f51b86e3dea96296606cf27c8c46514f49c8b920cda640e9ed77f92732ba"
)
K2_V3_HOLDOUT_SHA256 = (
    "61267ae96c3a557f02fee2dd5e7747e7a2ba296506d7a17f11dd0300e59db18e"
)
HISTORICAL_K3_V1_SHA256 = (
    "8017876a937fe8a25287a231ccf8a331422514f36241a80675d52ef39a96d71b"
)
FROZEN_TAU = 1.0
HEADS = 56
HIDDEN = 5376

K3_V2_EXACT_GATE = {
    "contract": EXACT_GATE_CONTRACT,
    "calibration_contract": calibration_probe.EVIDENCE_CONTRACT,
    "calibration_sha256": CALIBRATION_EVIDENCE_SHA256,
    "calibration_case_count": 12,
    "tau": FROZEN_TAU,
    "route_centroid_max_abs": 0.0,
    "raw_value_sum_max_abs": 0.0,
    "threshold_max_abs": 0.0,
    "route_trace_differing_bits": 0,
    "output_max_abs": 0.0,
}

K3_V2_HOLDOUT_CASES = (
    ("two-twentyfourth-385x2049-nosink", 2, 24, 385, 2049, 0, 0),
    ("three-sixteenth-513x513-prefix64", 3, 16, 513, 513, 0, 64),
    ("six-sixteenth-769x1537-offset192", 6, 16, 769, 1537, 512, 192),
    ("ten-sixteenth-1025x1537-prefix192", 10, 16, 1025, 1537, 0, 192),
)


def _exact_tensor_pass(metrics: dict[str, object], field: str) -> bool:
    return bool(metrics.get("finite")) and float(metrics[field]) == 0.0


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
        raise RuntimeError("K3 holdout capture attention input has invalid geometry")

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
    ) in K3_V2_HOLDOUT_CASES:
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
        ).view(q_rows, HEADS, HEAD_DIM)
        v_raw = replay_fixture._project(
            record.attention_input,
            v_weight,
            start=start,
            rows=v_rows,
            device=device,
        ).view(v_rows, HEADS, HEAD_DIM)
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

        qb = q.unsqueeze(0).contiguous()
        kb = k1_route.unsqueeze(0).contiguous()
        vb = v_raw.unsqueeze(0).contiguous()

        candidate_rc, candidate_vc = route_summary_sol_reduction(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        candidate_rc = candidate_rc.unsqueeze(0).contiguous()
        candidate_vc = candidate_vc.unsqueeze(0).contiguous()
        candidate_threshold = threshold_from_route_centroids(
            qb,
            candidate_rc,
            kv_rows=v_rows,
            tau=FROZEN_TAU,
            scale=calibration_probe.SCALE,
        )

        reference_rc, reference_vc, reference_threshold = prepare(
            qb,
            kb,
            vb,
            tau=FROZEN_TAU,
            scale=calibration_probe.SCALE,
            thresh_type="diag",
            valid_tokens=q_rows,
            valid_kv_tokens=v_rows,
        )

        torch.cuda.synchronize(device)
        candidate_started = time.perf_counter()
        candidate_output, candidate_trace = run_materialized_exact_selector_isolation(
            qb,
            kb,
            vb,
            candidate_rc,
            candidate_vc,
            candidate_threshold,
            scale=calibration_probe.SCALE,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        torch.cuda.synchronize(device)
        candidate_ms = (time.perf_counter() - candidate_started) * 1000.0

        reference_started = time.perf_counter()
        reference_output, reference_trace = run_materialized_exact_selector_isolation(
            qb,
            kb,
            vb,
            reference_rc,
            reference_vc,
            reference_threshold,
            scale=calibration_probe.SCALE,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        torch.cuda.synchronize(device)
        reference_ms = (time.perf_counter() - reference_started) * 1000.0

        rc_metrics = tensor_metrics(candidate_rc, reference_rc)
        vc_metrics = tensor_metrics(candidate_vc, reference_vc)
        threshold_metrics = tensor_metrics(
            candidate_threshold,
            reference_threshold,
        )
        trace_metrics = route_trace_metrics(
            candidate_trace,
            reference_trace,
            kv_rows=v_rows,
        )
        output_metrics = tensor_metrics(candidate_output, reference_output)

        passes = {
            "route_centroid": _exact_tensor_pass(rc_metrics, "max_abs"),
            "raw_value_sum": _exact_tensor_pass(vc_metrics, "max_abs"),
            "threshold": _exact_tensor_pass(threshold_metrics, "max_abs"),
            "route_trace": (
                bool(trace_metrics["equal"])
                and int(trace_metrics["differing_bits"]) == 0
            ),
            "output": _exact_tensor_pass(output_metrics, "max_abs"),
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
                "sink_start": sink_start,
                "sink_tokens": sink_tokens,
                "sink_blocks": list(
                    sink_block_range(
                        kv_rows=v_rows,
                        sink_start=sink_start,
                        sink_tokens=sink_tokens,
                    )
                ),
                "route_centroid": rc_metrics,
                "route_centroid_scale": tensor_scale_diagnostics(
                    candidate_rc,
                    reference_rc,
                ),
                "raw_value_sum": vc_metrics,
                "threshold": threshold_metrics,
                "route_trace": trace_metrics,
                "output": output_metrics,
                "output_scale": tensor_scale_diagnostics(
                    candidate_output,
                    reference_output,
                ),
                "k1_route_vs_comfy_route": tensor_metrics(
                    k1_route,
                    comfy_route,
                ),
                "k1_route_vs_comfy_route_scale": tensor_scale_diagnostics(
                    k1_route,
                    comfy_route,
                ),
                "passes": passes,
                "timing_ms": {
                    "candidate_first_observed": candidate_ms,
                    "reference_after_candidate": reference_ms,
                    "note": (
                        "debug-route-trace timings include compile/cache effects "
                        "and are not production performance evidence"
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
            comfy_route,
            k1_route,
            qb,
            kb,
            vb,
            candidate_rc,
            candidate_vc,
            candidate_threshold,
            reference_rc,
            reference_vc,
            reference_threshold,
            candidate_output,
            candidate_trace,
            reference_output,
            reference_trace,
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
    if not flat:
        raise RuntimeError("K3 holdout produced no cases")

    def maximum(metric: str, field: str) -> dict[str, object]:
        row = max(flat, key=lambda item: float(item[metric][field]))
        return {
            "value": float(row[metric][field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    trace_row = max(
        flat,
        key=lambda item: int(item["route_trace"]["differing_bits"]),
    )
    return {
        "case_count": len(flat),
        "all_route_centroids_exact": all(
            bool(item["passes"]["route_centroid"]) for item in flat
        ),
        "all_raw_value_sums_exact": all(
            bool(item["passes"]["raw_value_sum"]) for item in flat
        ),
        "all_thresholds_exact": all(
            bool(item["passes"]["threshold"]) for item in flat
        ),
        "all_route_traces_equal": all(
            bool(item["passes"]["route_trace"]) for item in flat
        ),
        "all_outputs_exact": all(
            bool(item["passes"]["output"]) for item in flat
        ),
        "maximum_route_trace_differing_bits": {
            "value": int(trace_row["route_trace"]["differing_bits"]),
            "block_index": int(trace_row["block_index"]),
            "case": str(trace_row["name"]),
        },
        "route_centroid_max_abs": maximum("route_centroid", "max_abs"),
        "raw_value_sum_max_abs": maximum("raw_value_sum", "max_abs"),
        "threshold_max_abs": maximum("threshold", "max_abs"),
        "output_max_abs": maximum("output", "max_abs"),
        "k1_route_vs_comfy_route": {
            "rel_l2": maximum("k1_route_vs_comfy_route", "rel_l2"),
            "max_abs": maximum("k1_route_vs_comfy_route", "max_abs"),
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
    parser.add_argument("--tau", type=float, default=FROZEN_TAU)
    parser.add_argument("--blocks", type=int, nargs="*", default=(0, 25, 49))
    args = parser.parse_args()

    source = replay_fixture._probe_source_identity(_REPO_ROOT)
    source["probe_contract"] = PROBE_CONTRACT
    source["critical_source_sha256"] = {
        rel: replay_fixture._sha256_file(_REPO_ROOT / rel)
        for rel in (
            "tools/keyless_k3_selector_holdout.py",
            "tools/keyless_k3_selector_probe.py",
            "tools/keyless_real_h3_replay_probe.py",
            "sol_h3/keyless_selector.py",
            "sol_h3/keyless_route_summary.py",
        )
    }

    if args.expected_probe_contract:
        expected = args.expected_probe_contract.strip()
        if expected != PROBE_CONTRACT:
            parser.error(
                "stale Sol-H3 K3 holdout probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the complete Patcher stack through PR #25"
            )
    if source["tracked_worktree_dirty"]:
        parser.error("K3-v2 holdout requires a clean tracked Sol-H3 worktree")
    if K1_CONTRACT != REQUIRED_K1_CONTRACT:
        parser.error(f"unexpected K1 contract {K1_CONTRACT!r}")
    if K3_CONTRACT != REQUIRED_K3_CONTRACT:
        parser.error(f"unexpected K3 contract {K3_CONTRACT!r}")
    if float(args.tau) != FROZEN_TAU:
        parser.error(f"K3-v2 exact holdout requires frozen tau={FROZEN_TAU}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("K3-v2 holdout requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error(
            "K3-v2 holdout requires SM120, got "
            f"{torch.cuda.get_device_capability(device)}"
        )
    selector_provenance = calibration_probe._native_selector_provenance(device)

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
        raise RuntimeError(f"capture bundle lacks requested K3 blocks: {missing}")

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

    all_holdout_cases_pass = all(
        bool(block["all_cases_pass"]) for block in blocks
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "contract": EVIDENCE_CONTRACT,
        "mode": "keyless-k3-selector-holdout",
        "promotion_evidence": False,
        "thresholds_frozen": True,
        "exact_gate_frozen": True,
        "probe_source": source,
        "source_contracts": {
            "k1": K1_CONTRACT,
            "k3": K3_CONTRACT,
        },
        "evidence_lineage": {
            "k1_v9_sha256": K1_V9_EVIDENCE_SHA256,
            "k2_v3_holdout_sha256": K2_V3_HOLDOUT_SHA256,
            "k3_v2_calibration_sha256": CALIBRATION_EVIDENCE_SHA256,
            "historical_k3_v1_sha256": HISTORICAL_K3_V1_SHA256,
        },
        "k3_v2_exact_gate": dict(K3_V2_EXACT_GATE),
        "tau": FROZEN_TAU,
        "attention_scale": calibration_probe.SCALE,
        "selector_provenance": selector_provenance,
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
            ) in K3_V2_HOLDOUT_CASES
        ],
        "blocks": blocks,
        "observed_extrema": _extrema(blocks),
        "all_holdout_cases_pass": all_holdout_cases_pass,
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
            "frozen exact K3-v2 holdout only; no provider promotion",
            (
                "exact-block K remains the globally materialized exact K1 row route "
                "to isolate selector behavior"
            ),
            "candidate K1-v6 RC/VC use bounded-b8 route scratch",
            "the full exact K1 route exists only on the diagnostic/reference side",
            "the released SM120 selector/approximate mainloop is reused unchanged",
            "one sigma-1 real-H3 capture and blocks 0/25/49 only",
            (
                "holdout windows are disjoint from the frozen K3-v2 calibration "
                "windows for this pinned capture"
            ),
            (
                "no K4 fused selected-route CuTe transform, decoded-media, sampler, "
                "or end-to-end performance evidence"
            ),
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        replay_fixture._write_json_atomic(output_path, result)
        print(
            json.dumps(
                {
                    "all_holdout_cases_pass": all_holdout_cases_pass,
                    "diagnostic_report": str(output_path.expanduser().resolve()),
                    "evidence_complete": True,
                    "exact_gate_frozen": True,
                    "probe_contract": PROBE_CONTRACT,
                    "probe_git_commit": source["git_commit"],
                    "promotion_evidence": False,
                    "thresholds_frozen": True,
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

    if not all_holdout_cases_pass:
        raise RuntimeError("real-H3 K3-v2 holdout violated the frozen exact gate")


if __name__ == "__main__":
    main()
