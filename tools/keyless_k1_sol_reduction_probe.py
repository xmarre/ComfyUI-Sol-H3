#!/usr/bin/env python3
"""Recalibrate K1 against the design-required existing Sol reduction on real H3.

The earlier K1 v2 campaign used a PyTorch FP32 block-sum oracle.  K3 showed that
the released Sol BF16 reduction is selector-sensitive and therefore is not an
interchangeable oracle.  This probe tests a versioned K1 v5 candidate that:

- derives routed centroids from raw V without writing a global route tensor;
- matches the released Sol TensorDescriptor/program geometry for RC reduction;
- reuses the exact released _reduce_kv_kernel for VC arithmetic;
- then feeds the v5 RC/VC into the unchanged SM120 Sol selector.

No acceptance threshold is frozen by this probe.
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

import keyless_k3_selector_probe as k3_fixture  # noqa: E402
import keyless_real_h3_replay_probe as replay_fixture  # noqa: E402
from sol_h3.keyless_real_h3_replay import (  # noqa: E402
    tensor_metrics,
    tensor_scale_diagnostics,
)
from sol_h3.keyless_route_summary import (  # noqa: E402
    CONTRACT as K1_V2_CONTRACT,
    HEAD_DIM,
    NORM_EPS,
    SOL_REDUCTION_CONTRACT as K1_V5_CONTRACT,
    materialized_sol_reduction_v4_route_diagnostic,
    route_summary,
    route_summary_sol_reduction,
)
from sol_h3.keyless_selector import (  # noqa: E402
    CONTRACT as K3_CONTRACT,
    route_trace_metrics,
    run_materialized_exact_selector_isolation,
    threshold_from_route_centroids,
)


PROBE_CONTRACT = "sol-h3-keyless-k1-sol-reduction-probe-v4"
EVIDENCE_CONTRACT = "sol-h3-keyless-k1-sol-reduction-calibration-v4"
REQUIRED_K1_V2_CONTRACT = "sol-h3-keyless-route-summary-v2"
REQUIRED_K1_V5_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v5"
REQUIRED_K3_CONTRACT = "sol-h3-keyless-selector-k3-calibration-v1"
HEADS = 56
HIDDEN = 5376
SCALE = HEAD_DIM ** -0.5


def _variant(
    *,
    q: torch.Tensor,
    route: torch.Tensor,
    raw_v: torch.Tensor,
    rc: torch.Tensor,
    vc: torch.Tensor,
    threshold: torch.Tensor,
    reference_output: torch.Tensor,
    reference_trace: torch.Tensor,
    v_rows: int,
    sink_start: int,
    sink_tokens: int,
) -> dict[str, object]:
    output, trace = run_materialized_exact_selector_isolation(
        q,
        route,
        raw_v,
        rc,
        vc,
        threshold,
        scale=SCALE,
        sink_start=sink_start,
        sink_tokens=sink_tokens,
    )
    torch.cuda.synchronize(q.device)
    return {
        "route_trace": route_trace_metrics(
            trace,
            reference_trace,
            kv_rows=v_rows,
        ),
        "output": tensor_metrics(output, reference_output),
        "output_scale": tensor_scale_diagnostics(output, reference_output),
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
        raise RuntimeError("K1 Sol-reduction capture attention input has invalid geometry")

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

    for name, anchor, q_rows, v_rows, sink_start, sink_tokens in (
        k3_fixture.K3_CALIBRATION_CASES
    ):
        span = max(q_rows, v_rows)
        start = k3_fixture._resolve_start(total_rows, span, anchor)

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
        route = replay_fixture._comfy_position(v_raw, route_norm, v_rope)

        qb = q.unsqueeze(0).contiguous()
        rb = route.unsqueeze(0).contiguous()
        vb = v_raw.unsqueeze(0).contiguous()

        reference_rc, reference_vc, reference_threshold = prepare(
            qb,
            rb,
            vb,
            tau=tau,
            scale=SCALE,
            thresh_type="diag",
            valid_tokens=q_rows,
            valid_kv_tokens=v_rows,
        )
        reference_output, reference_trace = run_materialized_exact_selector_isolation(
            qb,
            rb,
            vb,
            reference_rc,
            reference_vc,
            reference_threshold,
            scale=SCALE,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        torch.cuda.synchronize(device)

        v2_rc, v2_vc = route_summary(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        v2_rc = v2_rc.unsqueeze(0).contiguous()
        v2_vc = v2_vc.unsqueeze(0).contiguous()

        v5_rc, v5_vc = route_summary_sol_reduction(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        v5_rc = v5_rc.unsqueeze(0).contiguous()
        v5_vc = v5_vc.unsqueeze(0).contiguous()
        v5_threshold = threshold_from_route_centroids(
            qb,
            v5_rc,
            kv_rows=v_rows,
            tau=tau,
            scale=SCALE,
        )


        # Diagnostic-only full-route materialization. K1 v5 intentionally keeps
        # the v4 row arithmetic, so the established v4 materializer is the exact
        # row-route oracle for this reduction-identity check. This is not a
        # production implementation path.
        v5_materialized_route = materialized_sol_reduction_v4_route_diagnostic(
            v_raw,
            route_norm.to(device=device, dtype=torch.bfloat16),
            NORM_EPS,
            v_rope,
        )
        v5_materialized_route_b = v5_materialized_route.unsqueeze(0).contiguous()
        (
            v5_materialized_rc,
            v5_materialized_vc,
            v5_materialized_threshold,
        ) = prepare(
            qb,
            v5_materialized_route_b,
            vb,
            tau=tau,
            scale=SCALE,
            thresh_type="diag",
            valid_tokens=q_rows,
            valid_kv_tokens=v_rows,
        )
        v5_materialized_selector = _variant(
            q=qb,
            route=rb,
            raw_v=vb,
            rc=v5_materialized_rc,
            vc=v5_materialized_vc,
            threshold=v5_materialized_threshold,
            reference_output=reference_output,
            reference_trace=reference_trace,
            v_rows=v_rows,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        v5_reduction_identity = {
            "row_route": tensor_metrics(v5_materialized_route_b, rb),
            "row_route_scale": tensor_scale_diagnostics(
                v5_materialized_route_b,
                rb,
            ),
            "materialized_existing_sol_rc": tensor_metrics(
                v5_materialized_rc,
                reference_rc,
            ),
            "materialized_existing_sol_rc_scale": tensor_scale_diagnostics(
                v5_materialized_rc,
                reference_rc,
            ),
            "fused_rc_vs_materialized_existing_sol_rc": tensor_metrics(
                v5_rc,
                v5_materialized_rc,
            ),
            "fused_rc_vs_materialized_existing_sol_rc_scale": (
                tensor_scale_diagnostics(
                    v5_rc,
                    v5_materialized_rc,
                )
            ),
            "materialized_threshold": tensor_metrics(
                v5_materialized_threshold,
                reference_threshold,
            ),
            "selector": v5_materialized_selector,
            "comfy_rms_rope_backend": (
                replay_fixture._comfy_rms_rope_backend_identity(
                    v_raw,
                    route_norm,
                    v_rope,
                )
            ),
        }

        full_route_bytes = int(v_raw.numel() * v_raw.element_size())
        allocation = replay_fixture._candidate_allocation(
            lambda: route_summary_sol_reduction(
                v_raw,
                route_norm.to(device=device, dtype=torch.bfloat16),
                NORM_EPS,
                v_rope,
            ),
            device,
        )
        allocation["full_materialized_route_bytes"] = full_route_bytes
        allocation["below_full_route"] = (
            allocation["temporary_peak_delta"] < full_route_bytes
        )

        variants = {
            "v5_full": _variant(
                q=qb,
                route=rb,
                raw_v=vb,
                rc=v5_rc,
                vc=v5_vc,
                threshold=v5_threshold,
                reference_output=reference_output,
                reference_trace=reference_trace,
                v_rows=v_rows,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            ),
            "v5_selector_reference_vc": _variant(
                q=qb,
                route=rb,
                raw_v=vb,
                rc=v5_rc,
                vc=reference_vc,
                threshold=v5_threshold,
                reference_output=reference_output,
                reference_trace=reference_trace,
                v_rows=v_rows,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            ),
            "v5_rc_only": _variant(
                q=qb,
                route=rb,
                raw_v=vb,
                rc=v5_rc,
                vc=reference_vc,
                threshold=reference_threshold,
                reference_output=reference_output,
                reference_trace=reference_trace,
                v_rows=v_rows,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            ),
            "v5_threshold_only": _variant(
                q=qb,
                route=rb,
                raw_v=vb,
                rc=reference_rc,
                vc=reference_vc,
                threshold=v5_threshold,
                reference_output=reference_output,
                reference_trace=reference_trace,
                v_rows=v_rows,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            ),
            "v5_vc_only": _variant(
                q=qb,
                route=rb,
                raw_v=vb,
                rc=reference_rc,
                vc=v5_vc,
                threshold=reference_threshold,
                reference_output=reference_output,
                reference_trace=reference_trace,
                v_rows=v_rows,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            ),
        }

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
                "v2_baseline": {
                    "route_centroid": tensor_metrics(v2_rc, reference_rc),
                    "route_centroid_scale": tensor_scale_diagnostics(
                        v2_rc, reference_rc
                    ),
                    "raw_value_sum": tensor_metrics(v2_vc, reference_vc),
                },
                "v5_candidate": {
                    "route_centroid": tensor_metrics(v5_rc, reference_rc),
                    "route_centroid_scale": tensor_scale_diagnostics(
                        v5_rc, reference_rc
                    ),
                    "raw_value_sum": tensor_metrics(v5_vc, reference_vc),
                    "threshold": tensor_metrics(v5_threshold, reference_threshold),
                    "allocation": allocation,
                },
                "v5_reduction_identity": v5_reduction_identity,
                "variants": variants,
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

    def max_trace(variant: str, field: str) -> dict[str, object]:
        row = max(
            flat,
            key=lambda item: float(
                item["variants"][variant]["route_trace"][field]
            ),
        )
        return {
            "value": float(row["variants"][variant]["route_trace"][field]),
            "block_index": int(row["block_index"]),
            "case": str(row["name"]),
        }

    return {
        "case_count": len(flat),
        "v2_baseline": {
            "route_centroid_rel_l2": max_metric(
                ("v2_baseline", "route_centroid"), "rel_l2"
            ),
            "raw_value_sum_rel_l2": max_metric(
                ("v2_baseline", "raw_value_sum"), "rel_l2"
            ),
        },
        "v5_candidate": {
            "route_centroid_rel_l2": max_metric(
                ("v5_candidate", "route_centroid"), "rel_l2"
            ),
            "raw_value_sum_rel_l2": max_metric(
                ("v5_candidate", "raw_value_sum"), "rel_l2"
            ),
            "raw_value_sum_max_abs": max_metric(
                ("v5_candidate", "raw_value_sum"), "max_abs"
            ),
            "threshold_rel_l2": max_metric(
                ("v5_candidate", "threshold"), "rel_l2"
            ),
            "all_allocations_below_full_route": all(
                bool(item["v5_candidate"]["allocation"]["below_full_route"])
                for item in flat
            ),
        },

        "v5_reduction_identity": {
            "row_route_rel_l2": max_metric(
                ("v5_reduction_identity", "row_route"),
                "rel_l2",
            ),
            "materialized_existing_sol_rc_rel_l2": max_metric(
                ("v5_reduction_identity", "materialized_existing_sol_rc"),
                "rel_l2",
            ),
            "fused_rc_vs_materialized_existing_sol_rc_rel_l2": max_metric(
                (
                    "v5_reduction_identity",
                    "fused_rc_vs_materialized_existing_sol_rc",
                ),
                "rel_l2",
            ),
            "materialized_threshold_rel_l2": max_metric(
                ("v5_reduction_identity", "materialized_threshold"),
                "rel_l2",
            ),
            "materialized_selector_route_differing_bits": max_metric(
                (
                    "v5_reduction_identity",
                    "selector",
                    "route_trace",
                ),
                "differing_bits",
            ),
            "materialized_selector_output_rel_l2": max_metric(
                (
                    "v5_reduction_identity",
                    "selector",
                    "output",
                ),
                "rel_l2",
            ),
        },
        "v5_full": {
            "route_differing_bits": max_trace("v5_full", "differing_bits"),
            "route_differing_bit_fraction": max_trace(
                "v5_full", "differing_bit_fraction"
            ),
            "output_rel_l2": max_metric(
                ("variants", "v5_full", "output"), "rel_l2"
            ),
            "output_max_abs": max_metric(
                ("variants", "v5_full", "output"), "max_abs"
            ),
        },
        "decomposition": {
            name: {
                "route_differing_bits": max_trace(name, "differing_bits"),
                "output_rel_l2": max_metric(
                    ("variants", name, "output"), "rel_l2"
                ),
            }
            for name in (
                "v5_selector_reference_vc",
                "v5_rc_only",
                "v5_threshold_only",
                "v5_vc_only",
            )
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
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, nargs="*", default=(0, 25, 49))
    args = parser.parse_args()

    source = replay_fixture._probe_source_identity(_REPO_ROOT)
    source["probe_contract"] = PROBE_CONTRACT
    source["critical_source_sha256"] = {
        rel: replay_fixture._sha256_file(_REPO_ROOT / rel)
        for rel in (
            "tools/keyless_k1_sol_reduction_probe.py",
            "sol_h3/keyless_route_summary.py",
            "sol_h3/keyless_selector.py",
            "sol_h3/_vendor/sol_attn/preprocess.py",
        )
    }
    if args.expected_probe_contract:
        expected = args.expected_probe_contract.strip()
        if expected != PROBE_CONTRACT:
            parser.error(
                "stale Sol-H3 K1 Sol-reduction probe: expected "
                f"{expected!r}, local probe implements {PROBE_CONTRACT!r}; "
                "refresh the Sol #23 Patcher overlay"
            )
    if source["tracked_worktree_dirty"]:
        parser.error("K1 Sol-reduction calibration requires a clean tracked worktree")
    if K1_V2_CONTRACT != REQUIRED_K1_V2_CONTRACT:
        parser.error(f"unexpected K1 v2 contract {K1_V2_CONTRACT!r}")
    if K1_V5_CONTRACT != REQUIRED_K1_V5_CONTRACT:
        parser.error(f"unexpected K1 v5 contract {K1_V5_CONTRACT!r}")
    if K3_CONTRACT != REQUIRED_K3_CONTRACT:
        parser.error(f"unexpected K3 calibration contract {K3_CONTRACT!r}")
    if not 0.0 <= args.tau <= 3.0:
        parser.error("--tau must be in [0,3]")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("K1 Sol-reduction calibration requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        parser.error(
            "K1 Sol-reduction calibration requires SM120, got "
            f"{torch.cuda.get_device_capability(device)}"
        )
    selector_provenance = k3_fixture._native_selector_provenance(device)

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
        raise RuntimeError(
            f"capture bundle lacks requested K1 Sol-reduction blocks: {missing}"
        )

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
        "mode": "keyless-k1-sol-reduction-calibration",
        "promotion_evidence": False,
        "thresholds_frozen": False,
        "probe_source": source,
        "source_contracts": {
            "k1_v2": K1_V2_CONTRACT,
            "k1_v5": K1_V5_CONTRACT,
            "k3": K3_CONTRACT,
        },
        "evidence_lineage": {
            "prior_k1_v4_reduction_boundary_sha256": (
                "c3d253c98db46e676f161cfa1393b197370821d6ae40a99c24f6343d4cc72ee4"
            ),
            "prior_k1_v3_calibration_sha256": (
                "435f1ac38921bc46f4cd1efe58fff5b79439d928e7f58c1d128cfe2bb3464345"
            ),
            "prior_k3_calibration_sha256": (
                "8017876a937fe8a25287a231ccf8a331422514f36241a80675d52ef39a96d71b"
            ),
        },
        "selector_provenance": selector_provenance,
        "tau": float(args.tau),
        "attention_scale": SCALE,
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
                "anchor": anchor,
                "q_rows": q_rows,
                "v_rows": v_rows,
                "sink_start": sink_start,
                "sink_tokens": sink_tokens,
            }
            for name, anchor, q_rows, v_rows, sink_start, sink_tokens
            in k3_fixture.K3_CALIBRATION_CASES
        ],
        "blocks": blocks,
        "observed_extrema": _extrema(blocks),
        "memory_strategy": {
            "capture_deserialize_device": str(device),
            "checkpoint_tensor_device": str(device),
            "capture_bundle_rehash_after_validated_load": False,
            "large_file_hash_page_cache_policy": "posix_fadvise_dontneed_when_available",
            "cuda_free_bytes_after_calibration": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
        },
        "limitations": [
            "threshold-free recalibration only; no K1 v5 or K3 acceptance threshold is frozen",
            "exact-block K remains the materialized Comfy route while K1 reduction parity is isolated",
            "a full K1-v4 route is materialized only in the diagnostic/oracle "
            "path to verify exact fused-reduction identity",
            "K1 v5 is experimental and not connected to production provider dispatch",
            "one sigma-1 real-H3 capture and blocks 0/25/49 only",
            "no fused selected-route CuTe production mainloop, decoded-media, "
            "sampler, or end-to-end performance evidence",
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
