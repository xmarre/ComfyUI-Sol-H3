#!/usr/bin/env python3
"""Isolate the failed K4 holdout boundary without relaxing its frozen gate.

This diagnostic reuses the exact failed K4 holdout cases. For each case it
compares:

1. the bounded raw-V selected-route composition from PR #26;
2. the *same* Python composition with selected route tiles served directly as
   slices of the fully materialized exact-K1 route oracle;
3. released native Sol using that fully materialized route.

If (1) and (2) are exact while both retain the same small difference from (3),
the failed K4 holdout is localized to Python emulation of native Sol numerical
execution rather than selected-route tile derivation or routing semantics.

This is diagnostic evidence only. It does not alter the failed frozen gate and
does not promote the provider.
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

import keyless_k4_selected_route_holdout as failed_holdout  # noqa: E402
import keyless_k4_selected_route_probe as k4_probe  # noqa: E402
import keyless_real_h3_replay_probe as replay_fixture  # noqa: E402
import sol_h3.keyless_sparse_composition as composition  # noqa: E402
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
    exact_heads_from_route_trace,
    selected_route_composition,
)


PROBE_CONTRACT = "sol-h3-keyless-k4-boundary-isolation-probe-v1"
EVIDENCE_CONTRACT = "sol-h3-keyless-k4-boundary-isolation-v1"
FAILED_HOLDOUT_SHA256 = (
    "0bcf0fa216517cb46ddae569725c71744efa9df8efe953092399cd36fd81295a"
)
FAILED_HOLDOUT_CONTRACT = "sol-h3-keyless-k4-selected-route-holdout-v1"
REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
REQUIRED_K3_CONTRACT = "sol-h3-keyless-selector-k3-v2-k1-route-identity"
REQUIRED_K4_CONTRACT = "sol-h3-keyless-k4-selected-route-composition-v1"
SCALE = 128 ** -0.5
HEADS = 56
HEAD_DIM = 128
HIDDEN = 5376
BLOCK_SIZE = 64
ROUTE_GROUP_BLOCKS = 64


def _validate_failed_holdout(path: Path) -> dict[str, object]:
    sha256 = replay_fixture._sha256_file(path)
    if sha256 != FAILED_HOLDOUT_SHA256:
        raise RuntimeError(
            "K4 boundary isolation requires the exact failed holdout evidence: "
            f"{sha256} != {FAILED_HOLDOUT_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != FAILED_HOLDOUT_CONTRACT:
        raise RuntimeError("K4 boundary isolation received the wrong holdout contract")
    if payload.get("all_holdout_cases_pass") is not False:
        raise RuntimeError("K4 boundary isolation requires the failed holdout receipt")
    if payload.get("promotion_evidence") is not False:
        raise RuntimeError("failed K4 holdout unexpectedly claims promotion")
    if payload.get("composition_gate_frozen") is not True:
        raise RuntimeError("failed K4 holdout did not freeze the composition gate")
    return payload


def _selected_tile_identity(
    v: torch.Tensor,
    full_route: torch.Tensor,
    route_trace: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_freqs: torch.Tensor,
) -> dict[str, int | float | bool]:
    q_tiles = int(route_trace.shape[1])
    blocks = (int(v.shape[0]) + BLOCK_SIZE - 1) // BLOCK_SIZE
    selected_blocks: set[int] = set()

    for q_tile in range(q_tiles):
        for block_start in range(0, blocks, ROUTE_GROUP_BLOCKS):
            block_count = min(ROUTE_GROUP_BLOCKS, blocks - block_start)
            exact = exact_heads_from_route_trace(
                route_trace,
                q_tile=q_tile,
                block_start=block_start,
                block_count=block_count,
            )
            selected_local = torch.nonzero(
                exact.any(dim=0),
                as_tuple=False,
            ).flatten()
            selected_blocks.update(
                block_start + int(offset)
                for offset in selected_local.tolist()
            )

    max_abs = 0.0
    mismatch_elements = 0
    compared_elements = 0
    for block in sorted(selected_blocks):
        start = block * BLOCK_SIZE
        stop = min(start + BLOCK_SIZE, int(v.shape[0]))
        got = materialized_sol_reduction_v4_route_diagnostic(
            v[start:stop],
            norm_weight,
            NORM_EPS,
            rope_freqs[:, start:stop],
        )
        want = full_route[start:stop]
        difference = (got.float() - want.float()).abs()
        max_abs = max(max_abs, float(difference.max().item()))
        mismatch_elements += int(torch.count_nonzero(got != want).item())
        compared_elements += int(got.numel())

    return {
        "selected_block_count": len(selected_blocks),
        "compared_elements": compared_elements,
        "mismatch_elements": mismatch_elements,
        "max_abs": max_abs,
        "exact": mismatch_elements == 0,
    }


def _full_route_python_composition(
    q: torch.Tensor,
    v: torch.Tensor,
    route_centroid: torch.Tensor,
    value_sum: torch.Tensor,
    route_trace: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_freqs: torch.Tensor,
    full_route: torch.Tensor,
) -> torch.Tensor:
    original = composition.materialized_sol_reduction_v4_route_diagnostic

    if not v.is_contiguous() or not full_route.is_contiguous():
        raise RuntimeError("K4 boundary isolation requires contiguous V/full-route")
    row_stride = int(v.stride(0))

    def route_slice_loader(
        raw_tile: torch.Tensor,
        _norm_weight: torch.Tensor,
        _eps: float,
        _rope_tile: torch.Tensor,
    ) -> torch.Tensor:
        if raw_tile.device != v.device or raw_tile.dtype != v.dtype:
            raise RuntimeError("K4 oracle route loader received incompatible raw tile")
        element_offset = int(raw_tile.storage_offset()) - int(v.storage_offset())
        if element_offset < 0 or element_offset % row_stride:
            raise RuntimeError("K4 oracle route loader could not resolve V row offset")
        row_start = element_offset // row_stride
        row_stop = row_start + int(raw_tile.shape[0])
        return full_route[row_start:row_stop]

    composition.materialized_sol_reduction_v4_route_diagnostic = route_slice_loader
    try:
        output, _metrics = selected_route_composition(
            q,
            v,
            route_centroid,
            value_sum,
            route_trace,
            norm_weight,
            NORM_EPS,
            rope_freqs,
            scale=SCALE,
        )
        return output
    finally:
        composition.materialized_sol_reduction_v4_route_diagnostic = original


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
    ) in failed_holdout.K4_COMPOSITION_HOLDOUT_CASES:
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
        route_norm_bf16 = route_norm.to(device=device, dtype=torch.bfloat16)
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
        native_output, reference_trace = run_materialized_exact_selector_isolation(
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

        bounded_output, bounded_storage = selected_route_composition(
            q,
            v_raw,
            candidate_rc,
            candidate_vc,
            reference_trace,
            route_norm_bf16,
            NORM_EPS,
            v_rope,
            scale=SCALE,
        )
        full_python_output = _full_route_python_composition(
            q,
            v_raw,
            candidate_rc,
            candidate_vc,
            reference_trace,
            route_norm_bf16,
            v_rope,
            full_route,
        )
        tile_identity = _selected_tile_identity(
            v_raw,
            full_route,
            reference_trace,
            route_norm_bf16,
            v_rope,
        )

        bounded_b = bounded_output.unsqueeze(0)
        full_python_b = full_python_output.unsqueeze(0)
        bounded_vs_full = tensor_metrics(bounded_b, full_python_b)
        full_vs_native = tensor_metrics(full_python_b, native_output)
        bounded_vs_native = tensor_metrics(bounded_b, native_output)

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
                "selected_route_tile_identity": tile_identity,
                "bounded_vs_full_route_python": bounded_vs_full,
                "bounded_vs_full_route_python_scale": tensor_scale_diagnostics(
                    bounded_b,
                    full_python_b,
                ),
                "full_route_python_vs_native_sol": full_vs_native,
                "full_route_python_vs_native_sol_scale": tensor_scale_diagnostics(
                    full_python_b,
                    native_output,
                ),
                "bounded_vs_native_sol": bounded_vs_native,
                "bounded_vs_native_sol_scale": tensor_scale_diagnostics(
                    bounded_b,
                    native_output,
                ),
                "bounded_route_storage": bounded_storage,
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


def _extrema(blocks: list[dict[str, object]]) -> dict[str, object]:
    flat = [
        {"block_index": int(block["block_index"]), **case}
        for block in blocks
        for case in block["cases"]
    ]
    if not flat:
        raise RuntimeError("K4 boundary isolation produced no cases")

    def maximum(metric: str, field: str) -> dict[str, object]:
        row = max(flat, key=lambda item: float(item[metric][field]))
        return {
            "value": float(row[metric][field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    return {
        "case_count": len(flat),
        "all_selected_route_tiles_exact": all(
            bool(item["selected_route_tile_identity"]["exact"]) for item in flat
        ),
        "selected_route_tile_max_abs": max(
            float(item["selected_route_tile_identity"]["max_abs"]) for item in flat
        ),
        "selected_route_tile_mismatch_elements": sum(
            int(item["selected_route_tile_identity"]["mismatch_elements"])
            for item in flat
        ),
        "all_bounded_vs_full_route_python_exact": all(
            float(item["bounded_vs_full_route_python"]["max_abs"]) == 0.0
            for item in flat
        ),
        "bounded_vs_full_route_python": {
            "rel_l2": maximum("bounded_vs_full_route_python", "rel_l2"),
            "max_abs": maximum("bounded_vs_full_route_python", "max_abs"),
            "mean_abs": maximum("bounded_vs_full_route_python", "mean_abs"),
        },
        "full_route_python_vs_native_sol": {
            "rel_l2": maximum("full_route_python_vs_native_sol", "rel_l2"),
            "max_abs": maximum("full_route_python_vs_native_sol", "max_abs"),
            "mean_abs": maximum("full_route_python_vs_native_sol", "mean_abs"),
        },
        "bounded_vs_native_sol": {
            "rel_l2": maximum("bounded_vs_native_sol", "rel_l2"),
            "max_abs": maximum("bounded_vs_native_sol", "max_abs"),
            "mean_abs": maximum("bounded_vs_native_sol", "mean_abs"),
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
    parser.add_argument("--failed-holdout-evidence", required=True)
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
            "tools/keyless_k4_boundary_isolation_probe.py",
            "tools/keyless_k4_selected_route_holdout.py",
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
                "stale K4 boundary-isolation probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the complete Patcher stack through the isolation PR"
            )
    if source["tracked_worktree_dirty"]:
        parser.error("K4 boundary-isolation evidence requires a clean worktree")
    if K1_CONTRACT != REQUIRED_K1_CONTRACT:
        parser.error(f"unexpected K1 contract {K1_CONTRACT!r}")
    if K3_CONTRACT != REQUIRED_K3_CONTRACT:
        parser.error(f"unexpected K3 contract {K3_CONTRACT!r}")
    if K4_CONTRACT != REQUIRED_K4_CONTRACT:
        parser.error(f"unexpected K4 contract {K4_CONTRACT!r}")

    failed_path = Path(args.failed_holdout_evidence).resolve()
    _validate_failed_holdout(failed_path)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("K4 boundary isolation requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error("K4 boundary isolation requires SM120")

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

    extrema = _extrema(blocks)
    isolation_supported = (
        bool(extrema["all_selected_route_tiles_exact"])
        and bool(extrema["all_bounded_vs_full_route_python_exact"])
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "contract": EVIDENCE_CONTRACT,
        "mode": "keyless-k4-boundary-isolation",
        "promotion_evidence": False,
        "thresholds_frozen": False,
        "failed_holdout_gate_relaxed": False,
        "probe_source": source,
        "source_contracts": {
            "k1": K1_CONTRACT,
            "k3": K3_CONTRACT,
            "k4": K4_CONTRACT,
        },
        "evidence_lineage": {
            "failed_k4_holdout_sha256": FAILED_HOLDOUT_SHA256,
            "k4_calibration_sha256": failed_holdout.CALIBRATION_EVIDENCE_SHA256,
            "k3_holdout_sha256": k4_probe.K3_HOLDOUT_SHA256,
            "k3_calibration_sha256": k4_probe.K3_CALIBRATION_SHA256,
            "k2_holdout_sha256": k4_probe.K2_HOLDOUT_SHA256,
            "k1_v9_sha256": k4_probe.K1_V9_SHA256,
        },
        "isolation_hypothesis_supported": isolation_supported,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "failed_holdout_evidence": str(failed_path),
        "failed_holdout_evidence_sha256": FAILED_HOLDOUT_SHA256,
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
            ) in failed_holdout.K4_COMPOSITION_HOLDOUT_CASES
        ],
        "blocks": blocks,
        "observed_extrema": extrema,
        "memory_strategy": {
            "candidate_global_route_tensor": False,
            "full_route_exists_on_oracle_side_only": True,
            "cuda_free_bytes_after_evidence": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "limitations": [
            "diagnostic isolation after a failed frozen K4 holdout",
            "the failed holdout gate is not changed or reinterpreted",
            "selector generation remains owned by the proven materialized K3 oracle",
            "the full exact K1 route exists only on the oracle side",
            "Python composition does not emulate native Sol fast-exp2/MMA reduction exactly",
            "no vendored CuTe source is changed",
            "no provider dispatch or fused Keyless receipt/history identity is enabled",
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
                    "failed_holdout_gate_relaxed": False,
                    "isolation_hypothesis_supported": isolation_supported,
                    "probe_contract": PROBE_CONTRACT,
                    "probe_git_commit": source["git_commit"],
                    "promotion_evidence": False,
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
