from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "sol_h3" / "_vendor" / "sol_attn"
SOURCE_CONTRACT = "sana-sol-engine-sol-attn-64-rect-sm120-keyless-fused-v1"
PATCH_LABEL = "keyless-fused-sm120-v1"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_native_k4_candidate_is_not_promoted_before_sm120_evidence():
    source = _read("sol_h3/keyless_native.py")
    assert 'CONTRACT = "sol-h3-keyless-native-sm120-v1"' in source
    assert 'FUSED_RECEIPT_TAG = "sol_h3_keyless_fused_v1"' in source
    assert "PROMOTION_READY = False" in source
    assert "route_summary_sol_reduction(" in source
    assert "threshold_from_route_centroids(" in source
    assert "sol_attn_keyless(" in source
    assert "materialized_sol_reduction_v4_route_diagnostic" not in source
    assert "materialized_route" not in source


def test_native_sm120_routes_selected_k_inside_cta_and_keeps_pv_raw():
    source = _read("sol_h3/_vendor/sol_attn/sm120/mainloop.py")
    assert "keyless_enabled: bool = False" in source
    assert "route_keyless_smem_in_place(" in source
    assert "sK[row, pair, stage] = cutlass.BFloat16(" in source
    assert "sK[row, pair + 48, stage] = cutlass.BFloat16(" in source
    assert "sK[row, d, stage] = cutlass.BFloat16(" in source
    assert "sum_sq = cute.math.fma(value, value, sum_sq)" in source
    assert "cute.arch.shuffle_sync_down(sum_sq, offset)" in source
    assert "first_raw = cutlass.Float32(sK[row, pair, stage])" in source
    assert "second_raw = cutlass.Float32(sK[row, pair + 48, stage])" in source
    assert "gemm_smem_zero_acc(" in source
    assert "tma_atom_V" in source
    assert "tVgV[None, first_exact]" in source
    assert "route_inv_rms" not in source


def test_native_k4_probe_binds_isolation_and_stays_threshold_free():
    source = _read("tools/keyless_k4_native_sm120_probe.py")
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k4-native-sm120-probe-v1"' in source
    assert 'EVIDENCE_CONTRACT = "sol-h3-keyless-k4-native-sm120-calibration-v1"' in source
    assert (
        '"c9bd4e51e46e519c4612f0bbd8ae54cf56628049d3c29f618adf1ac95cb3b402"'
        in source
    )
    assert (
        '"0bcf0fa216517cb46ddae569725c71744efa9df8efe953092399cd36fd81295a"'
        in source
    )
    assert "PROMOTION_READY is not False" in source
    assert '"promotion_evidence": False' in source
    assert '"thresholds_frozen": False' in source
    assert '"provider_promotion_enabled": False' in source
    assert '"candidate_global_route_tensor": False' in source


def test_keyless_vendor_manifest_binds_exact_modified_bytes_and_patch():
    manifest = json.loads((ROOT / "sol_h3" / "sol_manifest.json").read_text())
    assert manifest["contract"] == SOURCE_CONTRACT

    patch = next(item for item in manifest["patches"] if item["label"] == PATCH_LABEL)
    patch_path = ROOT / patch["path"]
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == patch["sha256"]

    modified = (
        "__init__.py",
        "interface.py",
        "sm120/kernel.py",
        "sm120/mainloop.py",
    )
    for relative in modified:
        entry = manifest["files"][relative]
        assert PATCH_LABEL in entry.get("modifications", ())
        assert (
            hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest()
            == entry["packaged_sha256"]
        )


def test_provenance_and_interface_use_new_contract_without_relabeling_old_one():
    provenance = _read("sol_h3/provenance.py")
    interface = _read("sol_h3/_vendor/sol_attn/interface.py")
    assert f"CONTRACT = '{SOURCE_CONTRACT}'" in provenance
    assert f'KEYLESS_FUSED_CONTRACT = "{SOURCE_CONTRACT}"' in interface
    assert (
        'MAPPED_NEIGHBOR_CONTRACT = '
        '"sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4"'
        in interface
    )
    assert "keyless_enabled=False" in interface
    assert "sol_attn_keyless(" in interface
