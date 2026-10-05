#!/usr/bin/env python3
"""Run a frozen K4 selected-route composition gate on disjoint cases.

The arithmetic path is intentionally the exact calibration implementation from
keyless_k4_selected_route_probe.py. This holdout changes only the predeclared
capture windows and applies a gate frozen from calibration evidence before these
holdout outputs are observed.

This remains diagnostic composition evidence. It does not promote the Keyless
provider, does not generate selector policy inside the candidate, and does not
modify the vendored CuTe kernel.
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

import keyless_k4_selected_route_probe as calibration_probe  # noqa: E402
import keyless_real_h3_replay_probe as replay_fixture  # noqa: E402
from sol_h3.keyless_route_summary import (  # noqa: E402
    SOL_REDUCTION_CONTRACT as K1_CONTRACT,
)
from sol_h3.keyless_selector import CONTRACT as K3_CONTRACT  # noqa: E402
from sol_h3.keyless_sparse_composition import (  # noqa: E402
    CONTRACT as K4_CONTRACT,
)


PROBE_CONTRACT = "sol-h3-keyless-k4-selected-route-holdout-probe-v1"
EVIDENCE_CONTRACT = "sol-h3-keyless-k4-selected-route-holdout-v1"
GATE_CONTRACT = "sol-h3-keyless-k4-selected-route-gate-v1"

REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
REQUIRED_K3_CONTRACT = "sol-h3-keyless-selector-k3-v2-k1-route-identity"
REQUIRED_K4_CONTRACT = "sol-h3-keyless-k4-selected-route-composition-v1"
REQUIRED_CALIBRATION_PROBE = "sol-h3-keyless-k4-selected-route-probe-v1"
REQUIRED_CALIBRATION_CONTRACT = "sol-h3-keyless-k4-selected-route-calibration-v1"

CALIBRATION_EVIDENCE_SHA256 = (
    "056f36afaac76081911e5225a23f8f9a05b635398c4b62e6c501412db156e3e1"
)
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

FROZEN_TAU = 1.0
K4_COMPOSITION_GATE = {
    "contract": GATE_CONTRACT,
    "calibration_evidence_sha256": CALIBRATION_EVIDENCE_SHA256,
    "calibration_case_count": 12,
    "route_centroid_max_abs": 0.0,
    "raw_value_sum_max_abs": 0.0,
    "threshold_max_abs": 0.0,
    "candidate_rel_l2_max": 1.5e-4,
    "candidate_max_abs_over_want_abs_max": 0.006,
    "candidate_mean_abs_over_want_mean_abs": 3.5e-6,
    "candidate_worst_bf16_ulps_max": 1.0,
    "all_bounded_route_tiles_below_full_route": True,
}

# These windows were fixed only after consuming the K4 calibration receipt.
# They are pairwise disjoint and disjoint from all four K4 calibration windows
# for the pinned 24,879-row capture. The 4097-row case crosses the physical
# 64-block route-group boundary, adding a structural case absent from K4
# calibration.
K4_COMPOSITION_HOLDOUT_CASES = (
    ("zero-449x1025-nosink", 0, 1, 449, 1025, 0, 0),
    ("five-sixteenth-641x641-prefix128", 5, 16, 641, 641, 0, 128),
    ("eight-sixteenth-513x4097-offset256", 8, 16, 513, 4097, 3072, 256),
    ("fifteen-sixteenth-1153x1793-prefix256", 15, 16, 1153, 1793, 0, 256),
)


def _validate_calibration_evidence(path: Path) -> dict[str, object]:
    sha256 = replay_fixture._sha256_file(path)
    if sha256 != CALIBRATION_EVIDENCE_SHA256:
        raise RuntimeError(
            "K4 holdout calibration evidence SHA-256 mismatch: "
            f"{sha256} != {CALIBRATION_EVIDENCE_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != REQUIRED_CALIBRATION_CONTRACT:
        raise RuntimeError("K4 holdout received the wrong calibration contract")
    probe_source = payload.get("probe_source")
    if not isinstance(probe_source, dict):
        raise RuntimeError("K4 calibration evidence lacks probe source metadata")
    if probe_source.get("probe_contract") != REQUIRED_CALIBRATION_PROBE:
        raise RuntimeError("K4 calibration evidence has the wrong probe contract")
    if payload.get("promotion_evidence") is not False:
        raise RuntimeError("K4 calibration evidence unexpectedly claims promotion")
    if payload.get("thresholds_frozen") is not False:
        raise RuntimeError("K4 calibration evidence unexpectedly froze thresholds")
    extrema = payload.get("observed_extrema")
    if not isinstance(extrema, dict) or extrema.get("case_count") != 12:
        raise RuntimeError("K4 calibration evidence does not contain 12 cases")
    return payload


def _run_block_with_holdout_cases(
    record,
    *,
    checkpoint: Path,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, object]:
    original = calibration_probe.k3_holdout.K3_V2_HOLDOUT_CASES
    calibration_probe.k3_holdout.K3_V2_HOLDOUT_CASES = (
        K4_COMPOSITION_HOLDOUT_CASES
    )
    try:
        return calibration_probe._run_block(
            record,
            checkpoint=checkpoint,
            checkpoint_kind=checkpoint_kind,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
        )
    finally:
        calibration_probe.k3_holdout.K3_V2_HOLDOUT_CASES = original


def _case_passes(case: dict[str, object]) -> dict[str, bool]:
    gate = K4_COMPOSITION_GATE
    rc = case["route_centroid"]
    vc = case["raw_value_sum"]
    threshold = case["threshold"]
    output = case["candidate_vs_materialized_sol"]
    scale = case["candidate_vs_materialized_sol_scale"]
    bounded = case["bounded_route_storage"]
    worst = scale["worst"]

    passes = {
        "route_centroid": (
            bool(rc["finite"])
            and float(rc["max_abs"]) <= float(gate["route_centroid_max_abs"])
        ),
        "raw_value_sum": (
            bool(vc["finite"])
            and float(vc["max_abs"]) <= float(gate["raw_value_sum_max_abs"])
        ),
        "threshold": (
            bool(threshold["finite"])
            and float(threshold["max_abs"]) <= float(gate["threshold_max_abs"])
        ),
        "candidate_finite": bool(output["finite"]) and bool(scale["finite"]),
        "candidate_rel_l2": (
            float(output["rel_l2"]) <= float(gate["candidate_rel_l2_max"])
        ),
        "candidate_max_abs_scale": (
            float(scale["max_abs_over_want_abs_max"])
            <= float(gate["candidate_max_abs_over_want_abs_max"])
        ),
        "candidate_mean_abs_scale": (
            float(scale["mean_abs_over_want_mean_abs"])
            <= float(gate["candidate_mean_abs_over_want_mean_abs"])
        ),
        "candidate_bf16_ulps": (
            worst.get("abs_error_in_want_bf16_ulps") is not None
            and float(worst["abs_error_in_want_bf16_ulps"])
            <= float(gate["candidate_worst_bf16_ulps_max"])
        ),
        "bounded_route_storage": (
            int(bounded["max_route_tile_bytes"])
            < int(bounded["full_materialized_route_bytes"])
        ),
    }
    passes["all"] = all(passes.values())
    return passes


def _attach_passes(blocks: list[dict[str, object]]) -> bool:
    all_pass = True
    for block in blocks:
        block_pass = True
        for case in block["cases"]:
            passes = _case_passes(case)
            case["passes"] = passes
            block_pass = block_pass and bool(passes["all"])
        block["all_cases_pass"] = block_pass
        all_pass = all_pass and block_pass
    return all_pass


def _fractional_interval(
    total_rows: int,
    numerator: int,
    denominator: int,
    span: int,
) -> tuple[int, int]:
    start = ((total_rows - span) * numerator) // denominator
    return start, start + span


def _validate_holdout_windows(total_rows: int) -> None:
    calibration = (
        (1902, 3951),
        (4568, 5081),
        (8753, 10290),
        (14588, 16125),
    )
    intervals = [
        _fractional_interval(
            total_rows,
            numerator,
            denominator,
            max(q_rows, v_rows),
        )
        for (
            _name,
            numerator,
            denominator,
            q_rows,
            v_rows,
            _sink_start,
            _sink_tokens,
        ) in K4_COMPOSITION_HOLDOUT_CASES
    ]
    for start, stop in intervals:
        if any(
            max(start, cal_start) < min(stop, cal_stop)
            for cal_start, cal_stop in calibration
        ):
            raise RuntimeError("K4 holdout window overlaps K4 calibration")
    for index, (start, stop) in enumerate(intervals):
        if any(
            max(start, other_start) < min(stop, other_stop)
            for other_index, (other_start, other_stop) in enumerate(intervals)
            if other_index != index
        ):
            raise RuntimeError("K4 holdout windows overlap each other")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("teacher", "keyless"),
        required=True,
    )
    parser.add_argument("--calibration-evidence", required=True)
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
                "stale K4 selected-route holdout probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the complete Patcher stack through the K4 holdout PR"
            )
    if source["tracked_worktree_dirty"]:
        parser.error("K4 selected-route holdout requires a clean tracked worktree")
    if K1_CONTRACT != REQUIRED_K1_CONTRACT:
        parser.error(f"unexpected K1 contract {K1_CONTRACT!r}")
    if K3_CONTRACT != REQUIRED_K3_CONTRACT:
        parser.error(f"unexpected K3 contract {K3_CONTRACT!r}")
    if K4_CONTRACT != REQUIRED_K4_CONTRACT:
        parser.error(f"unexpected K4 composition contract {K4_CONTRACT!r}")
    if calibration_probe.PROBE_CONTRACT != REQUIRED_CALIBRATION_PROBE:
        parser.error(
            "K4 calibration probe contract changed after the frozen calibration"
        )
    if float(args.tau) != FROZEN_TAU:
        parser.error(f"K4 holdout requires frozen tau={FROZEN_TAU}")

    calibration_path = Path(args.calibration_evidence).resolve()
    _validate_calibration_evidence(calibration_path)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("K4 selected-route holdout requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error(
            "K4 selected-route holdout requires SM120, got "
            f"{torch.cuda.get_device_capability(device)}"
        )

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

    total_rows = min(int(record.attention_input.shape[0]) for record in records)
    _validate_holdout_windows(total_rows)

    blocks = []
    with torch.cuda.device(device), torch.inference_mode():
        for block in sorted(selected):
            blocks.append(
                _run_block_with_holdout_cases(
                    by_block[block],
                    checkpoint=checkpoint,
                    checkpoint_kind=args.checkpoint_kind,
                    checkpoint_sha256=checkpoint_sha256,
                    device=device,
                )
            )

    all_holdout_cases_pass = _attach_passes(blocks)
    extrema = calibration_probe._extrema(blocks)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "contract": EVIDENCE_CONTRACT,
        "mode": "keyless-k4-selected-route-holdout",
        "promotion_evidence": False,
        "thresholds_frozen": True,
        "composition_gate_frozen": True,
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
            "k4_composition_calibration_sha256": CALIBRATION_EVIDENCE_SHA256,
        },
        "k4_composition_gate": dict(K4_COMPOSITION_GATE),
        "tau": FROZEN_TAU,
        "attention_scale": calibration_probe.SCALE,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_evidence": str(calibration_path),
        "calibration_evidence_sha256": CALIBRATION_EVIDENCE_SHA256,
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
            ) in K4_COMPOSITION_HOLDOUT_CASES
        ],
        "blocks": blocks,
        "observed_extrema": extrema,
        "all_holdout_cases_pass": all_holdout_cases_pass,
        "memory_strategy": {
            "capture_deserialize_device": str(device),
            "checkpoint_tensor_device": str(device),
            "candidate_route_storage": "one selected V64 route tile at a time",
            "candidate_global_route_tensor": False,
            "cuda_free_bytes_after_evidence": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "limitations": [
            "frozen K4 composition holdout only; no provider promotion",
            "the route trace is still supplied by the proven materialized K3 oracle",
            "the candidate does not yet own selector generation",
            "Python/Torch orchestration is not production-performance evidence",
            "the full exact K1 route exists only on the oracle side",
            "no vendored CuTe source is changed by this holdout layer",
            "no fused Keyless receipt/history identity is enabled",
            "one sigma-1 real-H3 capture and blocks 0/25/49 only",
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        replay_fixture._write_json_atomic(output_path, result)
        print(
            json.dumps(
                {
                    "all_holdout_cases_pass": all_holdout_cases_pass,
                    "composition_gate_frozen": True,
                    "diagnostic_report": str(output_path.expanduser().resolve()),
                    "evidence_complete": True,
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
        raise RuntimeError(
            "real-H3 K4 selected-route holdout violated the frozen composition gate"
        )


if __name__ == "__main__":
    main()
