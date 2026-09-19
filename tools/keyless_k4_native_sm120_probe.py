#!/usr/bin/env python3
"""Calibrate the native SM120 Keyless K4 candidate against released Sol numerics.

This probe is the first evidence layer after K4 boundary isolation.  It requires
that exact PR #28 isolation receipt, then runs the new candidate-owned K1/K3
summary+selector path and the Keyless SM120 exact-fragment transform against the
materialized exact-K1 released-Sol oracle on the same real H3 capture.

No acceptance envelope is frozen here and provider promotion remains disabled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

_TOOL_PATH = Path(__file__).resolve()
_REPO_ROOT = _TOOL_PATH.parents[1]
_DEFAULT_COMFY_ROOT = _TOOL_PATH.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
if _DEFAULT_COMFY_ROOT.joinpath("comfy").is_dir():
    sys.path.insert(0, str(_DEFAULT_COMFY_ROOT))

import torch  # noqa: E402

import keyless_k4_selected_route_holdout as k4_holdout  # noqa: E402
import keyless_real_h3_replay_probe as replay_fixture  # noqa: E402
from sol_h3.keyless_native import (  # noqa: E402
    CONTRACT as NATIVE_CONTRACT,
    PROMOTION_READY,
    run_native_candidate,
)
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
    route_trace_metrics,
    run_materialized_exact_selector_isolation,
    threshold_from_route_centroids,
)
from sol_h3._vendor.sol_attn import KEYLESS_FUSED_CONTRACT  # noqa: E402
from sol_h3.provenance import CONTRACT as SOURCE_CONTRACT, verify_source  # noqa: E402


PROBE_CONTRACT = "sol-h3-keyless-k4-native-sm120-probe-v1"
EVIDENCE_CONTRACT = "sol-h3-keyless-k4-native-sm120-calibration-v1"
BOUNDARY_ISOLATION_CONTRACT = "sol-h3-keyless-k4-boundary-isolation-v1"
BOUNDARY_ISOLATION_SHA256 = (
    "c9bd4e51e46e519c4612f0bbd8ae54cf56628049d3c29f618adf1ac95cb3b402"
)
FAILED_K4_HOLDOUT_SHA256 = (
    "0bcf0fa216517cb46ddae569725c71744efa9df8efe953092399cd36fd81295a"
)
REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
REQUIRED_K3_CONTRACT = "sol-h3-keyless-selector-k3-v2-k1-route-identity"
REQUIRED_NATIVE_CONTRACT = "sol-h3-keyless-native-sm120-v1"
REQUIRED_SOURCE_CONTRACT = "sana-sol-engine-sol-attn-64-rect-sm120-keyless-fused-v1"
SCALE = 128 ** -0.5
HEADS = 56
HEAD_DIM = 128


def _validate_boundary_isolation(path: Path) -> dict[str, object]:
    sha256 = replay_fixture._sha256_file(path)
    if sha256 != BOUNDARY_ISOLATION_SHA256:
        raise RuntimeError(
            "native K4 calibration requires the exact consumed boundary-isolation "
            f"evidence: {sha256} != {BOUNDARY_ISOLATION_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != BOUNDARY_ISOLATION_CONTRACT:
        raise RuntimeError("native K4 calibration received the wrong isolation contract")
    if payload.get("isolation_hypothesis_supported") is not True:
        raise RuntimeError("boundary isolation did not exonerate selected-route derivation")
    if payload.get("failed_holdout_gate_relaxed") is not False:
        raise RuntimeError("boundary isolation unexpectedly relaxed the failed K4 gate")
    lineage = payload.get("evidence_lineage")
    if not isinstance(lineage, dict) or lineage.get(
        "failed_k4_holdout_sha256"
    ) != FAILED_K4_HOLDOUT_SHA256:
        raise RuntimeError("boundary isolation does not bind the frozen failed K4 holdout")
    return payload


def _run_block(
    record,
    *,
    checkpoint: Path,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    device: torch.device,
    case_names: set[str] | None,
    timing_repeats: int,
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
    route_norm_bf16 = route_norm.to(device=device, dtype=torch.bfloat16)
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
    ) in k4_holdout.K4_COMPOSITION_HOLDOUT_CASES:
        if case_names is not None and name not in case_names:
            continue
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
        full_route = materialized_sol_reduction_v4_route_diagnostic(
            v_raw,
            route_norm_bf16,
            NORM_EPS,
            v_rope,
        )

        qb = q.unsqueeze(0).contiguous()
        kb = full_route.unsqueeze(0).contiguous()
        vb = v_raw.unsqueeze(0).contiguous()
        candidate_rc, candidate_vc = route_summary_sol_reduction(
            v_raw,
            route_norm_bf16,
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
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        started = time.perf_counter()
        candidate_output, candidate_trace = run_native_candidate(
            q,
            v_raw,
            route_norm_bf16,
            NORM_EPS,
            v_rope,
            scale=SCALE,
            tau=1.0,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
            debug_route_trace=True,
        )
        torch.cuda.synchronize(device)
        debug_ms = (time.perf_counter() - started) * 1000.0
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_delta = max(0, peak_allocated - allocated_before)

        warm_ms: list[float] = []
        if timing_repeats:
            # The non-debug specialization compiles once here; only subsequent
            # executions enter the timing sample.
            _ = run_native_candidate(
                q,
                v_raw,
                route_norm_bf16,
                NORM_EPS,
                v_rope,
                scale=SCALE,
                tau=1.0,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
                debug_route_trace=False,
            )
            torch.cuda.synchronize(device)
            for _index in range(timing_repeats):
                started = time.perf_counter()
                _ = run_native_candidate(
                    q,
                    v_raw,
                    route_norm_bf16,
                    NORM_EPS,
                    v_rope,
                    scale=SCALE,
                    tau=1.0,
                    sink_start=sink_start,
                    sink_tokens=sink_tokens,
                    debug_route_trace=False,
                )
                torch.cuda.synchronize(device)
                warm_ms.append((time.perf_counter() - started) * 1000.0)

        candidate_b = candidate_output.unsqueeze(0)
        trace = route_trace_metrics(
            candidate_trace,
            reference_trace,
            kv_rows=v_rows,
        )
        full_route_bytes = int(v_raw.numel() * v_raw.element_size())
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
                "route_trace": trace,
                "candidate_vs_materialized_sol": tensor_metrics(
                    candidate_b,
                    reference_output,
                ),
                "candidate_vs_materialized_sol_scale": tensor_scale_diagnostics(
                    candidate_b,
                    reference_output,
                ),
                "memory": {
                    "full_materialized_route_bytes": full_route_bytes,
                    "candidate_peak_allocated_delta_bytes": peak_delta,
                    "candidate_global_route_tensor": False,
                },
                "timing_ms": {
                    "debug_compile_or_execute": debug_ms,
                    "warm_repeats": warm_ms,
                    "warm_median": (
                        statistics.median(warm_ms) if warm_ms else None
                    ),
                    "timing_is_production_claim": False,
                },
            }
        )

        del (
            q_raw,
            v_raw,
            q_rope,
            v_rope,
            q,
            full_route,
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
            candidate_trace,
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
        raise RuntimeError("native K4 calibration produced no cases")

    def maximum(metric: str, field: str) -> dict[str, object]:
        row = max(flat, key=lambda item: float(item[metric][field]))
        return {
            "value": float(row[metric][field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def max_scale(field: str) -> dict[str, object]:
        rows = [
            item
            for item in flat
            if item["candidate_vs_materialized_sol_scale"].get(field) is not None
        ]
        row = max(
            rows,
            key=lambda item: float(
                item["candidate_vs_materialized_sol_scale"][field]
            ),
        )
        return {
            "value": float(row["candidate_vs_materialized_sol_scale"][field]),
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
        "all_route_traces_exact": all(
            bool(item["route_trace"]["equal"]) for item in flat
        ),
        "route_trace_differing_bits": sum(
            int(item["route_trace"]["differing_bits"]) for item in flat
        ),
        "route_centroid_max_abs": maximum("route_centroid", "max_abs"),
        "raw_value_sum_max_abs": maximum("raw_value_sum", "max_abs"),
        "threshold_max_abs": maximum("threshold", "max_abs"),
        "candidate_vs_materialized_sol": {
            "rel_l2": maximum("candidate_vs_materialized_sol", "rel_l2"),
            "mean_abs": maximum("candidate_vs_materialized_sol", "mean_abs"),
            "max_abs": maximum("candidate_vs_materialized_sol", "max_abs"),
            "max_abs_over_want_abs_max": max_scale("max_abs_over_want_abs_max"),
            "mean_abs_over_want_mean_abs": max_scale("mean_abs_over_want_mean_abs"),
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
        "maximum_candidate_peak_allocated_delta_bytes": max(
            int(item["memory"]["candidate_peak_allocated_delta_bytes"])
            for item in flat
        ),
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
    parser.add_argument("--boundary-isolation-evidence", required=True)
    parser.add_argument("--keyless-repo")
    parser.add_argument("--expected-receipt-sha256")
    parser.add_argument("--expected-probe-contract")
    parser.add_argument("--output-json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, nargs="*", default=(0, 25, 49))
    parser.add_argument("--case-names", nargs="*")
    parser.add_argument("--timing-repeats", type=int, default=0)
    args = parser.parse_args()

    if args.timing_repeats < 0 or args.timing_repeats > 10:
        parser.error("--timing-repeats must be in [0,10]")
    case_names = set(args.case_names) if args.case_names else None
    known_cases = {item[0] for item in k4_holdout.K4_COMPOSITION_HOLDOUT_CASES}
    if case_names is not None:
        unknown = sorted(case_names.difference(known_cases))
        if unknown:
            parser.error(f"unknown native K4 case names: {unknown}")

    source = replay_fixture._probe_source_identity(_REPO_ROOT)
    source["probe_contract"] = PROBE_CONTRACT
    source["critical_source_sha256"] = {
        rel: replay_fixture._sha256_file(_REPO_ROOT / rel)
        for rel in (
            "tools/keyless_k4_native_sm120_probe.py",
            "sol_h3/keyless_native.py",
            "sol_h3/keyless_route_summary.py",
            "sol_h3/keyless_selector.py",
            "sol_h3/_vendor/sol_attn/interface.py",
            "sol_h3/_vendor/sol_attn/sm120/kernel.py",
            "sol_h3/_vendor/sol_attn/sm120/mainloop.py",
            "sol_h3/provenance.py",
            "sol_h3/sol_manifest.json",
        )
    }
    if args.expected_probe_contract:
        expected = args.expected_probe_contract.strip()
        if expected != PROBE_CONTRACT:
            parser.error(
                "stale native K4 probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the complete Patcher stack through the native K4 PR"
            )
    if source["tracked_worktree_dirty"]:
        parser.error("native K4 evidence requires a clean tracked worktree")
    if PROMOTION_READY is not False:
        parser.error("native K4 calibration must run before provider promotion")
    if K1_CONTRACT != REQUIRED_K1_CONTRACT:
        parser.error(f"unexpected K1 contract {K1_CONTRACT!r}")
    if K3_CONTRACT != REQUIRED_K3_CONTRACT:
        parser.error(f"unexpected K3 contract {K3_CONTRACT!r}")
    if NATIVE_CONTRACT != REQUIRED_NATIVE_CONTRACT:
        parser.error(f"unexpected native K4 contract {NATIVE_CONTRACT!r}")
    if SOURCE_CONTRACT != REQUIRED_SOURCE_CONTRACT:
        parser.error(f"unexpected packaged source contract {SOURCE_CONTRACT!r}")
    if KEYLESS_FUSED_CONTRACT != REQUIRED_SOURCE_CONTRACT:
        parser.error(f"unexpected vendored Keyless ABI {KEYLESS_FUSED_CONTRACT!r}")
    manifest = verify_source()
    if manifest.get("contract") != REQUIRED_SOURCE_CONTRACT:
        parser.error("verified source manifest does not expose the native Keyless contract")

    isolation_path = Path(args.boundary_isolation_evidence).resolve()
    _validate_boundary_isolation(isolation_path)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("native K4 calibration requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error("native K4 calibration requires SM120")

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
            raise RuntimeError("teacher checkpoint SHA-256 does not match capture provenance")

    selected = set(args.blocks)
    by_block = {int(record.block_index): record for record in records}
    missing = sorted(selected.difference(by_block))
    if missing:
        raise RuntimeError(f"capture bundle lacks requested native K4 blocks: {missing}")

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
                    case_names=case_names,
                    timing_repeats=args.timing_repeats,
                )
            )

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "contract": EVIDENCE_CONTRACT,
        "mode": "keyless-k4-native-sm120-calibration",
        "promotion_evidence": False,
        "thresholds_frozen": False,
        "provider_promotion_enabled": False,
        "probe_source": source,
        "source_contracts": {
            "k1": K1_CONTRACT,
            "k3": K3_CONTRACT,
            "native_k4": NATIVE_CONTRACT,
            "vendored_sm120": SOURCE_CONTRACT,
        },
        "evidence_lineage": {
            "boundary_isolation_sha256": BOUNDARY_ISOLATION_SHA256,
            "failed_k4_holdout_sha256": FAILED_K4_HOLDOUT_SHA256,
        },
        "boundary_isolation_evidence": str(isolation_path),
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
        "case_filter": sorted(case_names) if case_names is not None else None,
        "timing_repeats": args.timing_repeats,
        "blocks": blocks,
        "observed_extrema": _extrema(blocks),
        "memory_strategy": {
            "candidate_global_route_tensor": False,
            "selected_exact_route_storage": "bounded SM120 K shared-memory tile",
            "selected_exact_rms": "computed inside CTA from the selected raw-V tile",
            "k1_route_scratch": "bounded b8 BF16 spill/reload",
            "cuda_free_bytes_after_evidence": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "limitations": [
            "threshold-free native K4 calibration; no acceptance envelope is frozen",
            "provider promotion remains disabled",
            "source/contract structure and real-SM120 arithmetic are separate gates",
            "decoded-media parity and end-to-end sampler behavior are not measured",
            "timing fields are exploratory and not a production speed claim",
            "K5 mapped-neighbor/measure/selected-domain/Untwist/VDN/Flow compositions remain out of scope",
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
