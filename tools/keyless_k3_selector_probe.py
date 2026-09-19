#!/usr/bin/env python3
"""Recalibrate K3 against the proven K1-v6 route identity on real H3.

K1-v6 supplies routed centroids/value sums from raw V through its bounded-b8
spill/reload primitive. The reference materializes the exact K1 row route only
on the diagnostic side and feeds it through the released Sol reducer. Candidate
and reference then use the same materialized exact K1 route for selected exact
blocks, so this run isolates only K3 threshold/selector/approximate-summary
semantics after the upstream K1 mismatch was corrected.

The run is threshold-free calibration evidence only. No K3 gate is frozen by it.
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


PROBE_CONTRACT = "sol-h3-keyless-k3-selector-probe-v2"
EVIDENCE_CONTRACT = "sol-h3-keyless-k3-selector-calibration-v2"
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
HEADS = 56
HIDDEN = 5376
SCALE = HEAD_DIM ** -0.5



def _native_selector_provenance(device: torch.device) -> dict[str, object]:
    """Verify and record the exact packaged source/backend/compiler stack."""
    from importlib.metadata import PackageNotFoundError, version as distribution_version

    from sol_h3._vendor.sol_attn import get_sol_attn_backend
    from sol_h3.provenance import CONTRACT as SOURCE_CONTRACT
    from sol_h3.provenance import REVISION as SOURCE_REVISION
    from sol_h3.provenance import SOURCE as SOURCE_NAME
    from sol_h3.provenance import verify_source

    manifest = verify_source()
    backend = get_sol_attn_backend(device)
    if backend != "cute_sm120":
        raise RuntimeError(
            f"K3 selector calibration requires verified cute_sm120, got {backend!r}"
        )

    distributions = {}
    for name in (
        "nvidia-cutlass-dsl",
        "cuda-python",
        "triton",
        "torch",
    ):
        try:
            distributions[name] = distribution_version(name)
        except PackageNotFoundError:
            distributions[name] = None

    return {
        "source": SOURCE_NAME,
        "revision": SOURCE_REVISION,
        "contract": SOURCE_CONTRACT,
        "manifest_source": manifest.get("source"),
        "manifest_revision": manifest.get("revision"),
        "manifest_contract": manifest.get("contract"),
        "backend": backend,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "distributions": distributions,
    }


K3_CALIBRATION_CASES = (
    ("head-257x1537-nosink", "head", 257, 1537, 0, 0),
    ("quarter-385x2049-prefix128", "quarter", 385, 2049, 0, 128),
    ("middle-769x4097-offset192", "middle", 769, 4097, 1024, 192),
    ("tail-1025x8193-prefix256", "tail", 1025, 8193, 0, 256),
)


def _resolve_start(total_rows: int, span: int, anchor: str) -> int:
    if span <= 0 or span > total_rows:
        raise ValueError(f"K3 span {span} does not fit captured length {total_rows}")
    available = total_rows - span
    if anchor == "head":
        return 0
    if anchor == "quarter":
        return available // 4
    if anchor == "middle":
        return available // 2
    if anchor == "tail":
        return available
    raise ValueError(f"unsupported K3 anchor {anchor!r}")


def _case_extrema(blocks: list[dict[str, object]]) -> dict[str, object]:
    flat = [
        {"block_index": int(block["block_index"]), **case}
        for block in blocks
        for case in block["cases"]
    ]
    if not flat:
        raise RuntimeError("K3 calibration produced no cases")

    def maximum(metric: str, field: str) -> dict[str, object]:
        row = max(flat, key=lambda item: float(item[metric][field]))
        return {
            "value": float(row[metric][field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def maximum_scale(metric: str, field: str) -> dict[str, object]:
        def value(item):
            raw = item[metric].get(field)
            return float(raw) if raw is not None else float("-inf")

        row = max(flat, key=value)
        return {
            "value": value(row),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    def maximum_ulp(metric: str) -> dict[str, object]:
        def value(item):
            raw = item[metric]["worst"]["abs_error_in_want_bf16_ulps"]
            return float(raw) if raw is not None else float("inf")

        row = max(flat, key=value)
        return {
            "value": value(row),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    trace_diff = max(
        flat,
        key=lambda item: int(item["route_trace"]["differing_bits"]),
    )
    trace_fraction = max(
        flat,
        key=lambda item: float(item["route_trace"]["differing_bit_fraction"]),
    )
    return {
        "case_count": len(flat),
        "all_route_centroids_exact": all(
            float(item["route_centroid"]["max_abs"]) == 0.0 for item in flat
        ),
        "all_raw_value_sums_exact": all(
            float(item["raw_value_sum"]["max_abs"]) == 0.0 for item in flat
        ),
        "all_thresholds_exact": all(
            float(item["threshold"]["max_abs"]) == 0.0 for item in flat
        ),
        "all_route_traces_equal": all(
            bool(item["route_trace"]["equal"]) for item in flat
        ),
        "all_outputs_exact": all(
            float(item["output"]["max_abs"]) == 0.0 for item in flat
        ),
        "maximum_route_trace_differing_bits": {
            "value": int(trace_diff["route_trace"]["differing_bits"]),
            "block_index": int(trace_diff["block_index"]),
            "case": str(trace_diff["name"]),
        },
        "maximum_route_trace_differing_bit_fraction": {
            "value": float(trace_fraction["route_trace"]["differing_bit_fraction"]),
            "block_index": int(trace_fraction["block_index"]),
            "case": str(trace_fraction["name"]),
        },
        "route_centroid": {
            "rel_l2": maximum("route_centroid", "rel_l2"),
            "mean_abs": maximum("route_centroid", "mean_abs"),
            "max_abs": maximum("route_centroid", "max_abs"),
            "max_abs_over_want_abs_max": maximum_scale(
                "route_centroid_scale", "max_abs_over_want_abs_max"
            ),
            "mean_abs_over_want_mean_abs": maximum_scale(
                "route_centroid_scale", "mean_abs_over_want_mean_abs"
            ),
            "worst_bf16_ulps": maximum_ulp("route_centroid_scale"),
        },
        "threshold": {
            "rel_l2": maximum("threshold", "rel_l2"),
            "mean_abs": maximum("threshold", "mean_abs"),
            "max_abs": maximum("threshold", "max_abs"),
        },
        "output": {
            "rel_l2": maximum("output", "rel_l2"),
            "mean_abs": maximum("output", "mean_abs"),
            "max_abs": maximum("output", "max_abs"),
            "max_abs_over_want_abs_max": maximum_scale(
                "output_scale", "max_abs_over_want_abs_max"
            ),
            "mean_abs_over_want_mean_abs": maximum_scale(
                "output_scale", "mean_abs_over_want_mean_abs"
            ),
            "worst_bf16_ulps": maximum_ulp("output_scale"),
        },
        "raw_value_sum_max_abs": maximum("raw_value_sum", "max_abs"),
        "k1_route_vs_comfy_route": {
            "rel_l2": maximum("k1_route_vs_comfy_route", "rel_l2"),
            "max_abs": maximum("k1_route_vs_comfy_route", "max_abs"),
        },
    }


def _run_block(
    record,
    *,
    checkpoint: Path,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    device: torch.device,
    tau: float,
) -> dict[str, object]:
    if record.case.rope_freqs is None or not torch.is_tensor(record.case.rope_freqs):
        raise RuntimeError(f"capture block {record.block_index} has no exact H3 RoPE")
    if record.attention_input.ndim != 2 or record.attention_input.shape[1] != HIDDEN:
        raise RuntimeError("K3 capture attention input has invalid hidden geometry")

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

    for name, anchor, q_rows, v_rows, sink_start, sink_tokens in K3_CALIBRATION_CASES:
        span = max(q_rows, v_rows)
        start = _resolve_start(total_rows, span, anchor)

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
            tau=tau,
            scale=SCALE,
        )

        reference_rc, reference_vc, reference_threshold = prepare(
            qb,
            kb,
            vb,
            tau=tau,
            scale=SCALE,
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
            scale=SCALE,
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
            scale=SCALE,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        torch.cuda.synchronize(device)
        reference_ms = (time.perf_counter() - reference_started) * 1000.0

        trace = route_trace_metrics(
            candidate_trace,
            reference_trace,
            kv_rows=v_rows,
        )
        sink_blocks = sink_block_range(
            kv_rows=v_rows,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        cases.append(
            {
                "name": name,
                "anchor": anchor,
                "q_start": start,
                "q_rows": q_rows,
                "v_start": start,
                "v_rows": v_rows,
                "sink_start": sink_start,
                "sink_tokens": sink_tokens,
                "sink_blocks": list(sink_blocks),
                "route_centroid": tensor_metrics(candidate_rc, reference_rc),
                "route_centroid_scale": tensor_scale_diagnostics(
                    candidate_rc, reference_rc
                ),
                "raw_value_sum": tensor_metrics(candidate_vc, reference_vc),
                "k1_route_vs_comfy_route": tensor_metrics(k1_route, comfy_route),
                "k1_route_vs_comfy_route_scale": tensor_scale_diagnostics(
                    k1_route,
                    comfy_route,
                ),
                "threshold": tensor_metrics(
                    candidate_threshold, reference_threshold
                ),
                "route_trace": trace,
                "output": tensor_metrics(candidate_output, reference_output),
                "output_scale": tensor_scale_diagnostics(
                    candidate_output, reference_output
                ),
                "timing_ms": {
                    "candidate_first_observed": candidate_ms,
                    "reference_after_candidate": reference_ms,
                    "note": (
                        "debug-route-trace diagnostic timings include launch/cache effects "
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
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, nargs="*", default=(0, 25, 49))
    args = parser.parse_args()

    probe_source = replay_fixture._probe_source_identity(_REPO_ROOT)
    probe_source["probe_contract"] = PROBE_CONTRACT
    probe_source["critical_source_sha256"] = {
        rel: replay_fixture._sha256_file(_REPO_ROOT / rel)
        for rel in (
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
                "stale Sol-H3 K3 probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the complete Patcher stack through the K3-v2 PR"
            )
    if probe_source["tracked_worktree_dirty"]:
        parser.error("K3 calibration requires a clean tracked Sol-H3 worktree")
    if K1_CONTRACT != REQUIRED_K1_CONTRACT:
        parser.error(
            f"K3-v2 calibration requires proven K1-v6, got {K1_CONTRACT!r}; "
            "refresh the complete Patcher stack through the K3-v2 PR"
        )
    if K3_CONTRACT != REQUIRED_K3_CONTRACT:
        parser.error(f"unexpected K3 selector contract {K3_CONTRACT!r}")
    if not 0.0 <= args.tau <= 3.0:
        parser.error("--tau must be in [0,3]")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("K3 calibration requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error(
            "K3 calibration requires SM120, got "
            f"{torch.cuda.get_device_capability(device)}"
        )
    selector_provenance = _native_selector_provenance(device)

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
                    tau=float(args.tau),
                )
            )

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "contract": EVIDENCE_CONTRACT,
        "mode": "keyless-k3-selector-calibration",
        "promotion_evidence": False,
        "thresholds_frozen": False,
        "probe_source": probe_source,
        "source_contracts": {
            "k1": K1_CONTRACT,
            "k3": K3_CONTRACT,
        },
        "evidence_lineage": {
            "k1_v9_sha256": K1_V9_EVIDENCE_SHA256,
            "k2_v3_holdout_sha256": K2_V3_HOLDOUT_SHA256,
            "historical_k3_v1_sha256": HISTORICAL_K3_V1_SHA256,
        },
        "tau": float(args.tau),
        "attention_scale": SCALE,
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
        "calibration_case_definitions": [
            {
                "name": name,
                "anchor": anchor,
                "q_rows": q_rows,
                "v_rows": v_rows,
                "sink_start": sink_start,
                "sink_tokens": sink_tokens,
            }
            for name, anchor, q_rows, v_rows, sink_start, sink_tokens
            in K3_CALIBRATION_CASES
        ],
        "blocks": blocks,
        "observed_extrema": _case_extrema(blocks),
        "memory_strategy": {
            "capture_deserialize_device": str(device),
            "checkpoint_tensor_device": str(device),
            "capture_bundle_rehash_after_validated_load": False,
            "large_file_hash_page_cache_policy": "posix_fadvise_dontneed_when_available",
            "cuda_free_bytes_after_calibration": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "limitations": [
            "threshold-free K3 calibration only; no K3 acceptance threshold is frozen",
            (
                "exact-block K remains the globally materialized exact K1 row route "
                "to isolate selector behavior"
            ),
            "candidate K1-v6 RC/VC come from raw V through bounded-b8 route scratch",
            "the full exact K1 route exists only on the diagnostic/reference side",
            "the released SM120 selector/approximate mainloop is reused unchanged",
            "one sigma-1 real-H3 capture and blocks 0/25/49 only",
            "no production provider promotion, fused selected-route CuTe transform, decoded-media, sampler, or end-to-end performance evidence",
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        replay_fixture._write_json_atomic(output_path, result)
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


if __name__ == "__main__":
    main()
