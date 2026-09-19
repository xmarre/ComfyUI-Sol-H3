from __future__ import annotations

from pathlib import Path


def _probe_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_k1_sol_reduction_probe.py"
    ).read_text(encoding="utf-8")


def test_k1_sol_reduction_probe_contracts_are_versioned():
    source = _probe_source()
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k1-sol-reduction-probe-v3"' in source
    assert (
        'EVIDENCE_CONTRACT = "sol-h3-keyless-k1-sol-reduction-calibration-v3"'
        in source
    )
    assert (
        'REQUIRED_K1_V4_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v4"'
        in source
    )
    assert '"promotion_evidence": False' in source
    assert '"thresholds_frozen": False' in source


def test_k1_sol_reduction_probe_compares_v2_v4_and_existing_sol_reduction():
    source = _probe_source()
    assert "reference_rc, reference_vc, reference_threshold = prepare(" in source
    assert "v2_rc, v2_vc = route_summary(" in source
    assert "v4_rc, v4_vc = route_summary_sol_reduction(" in source
    assert "tensor_metrics(v4_rc, reference_rc)" in source
    assert "tensor_metrics(v4_vc, reference_vc)" in source


def test_k1_sol_reduction_probe_decomposes_selector_sensitivity():
    source = _probe_source()
    for name in (
        '"v4_full"',
        '"v4_selector_reference_vc"',
        '"v4_rc_only"',
        '"v4_threshold_only"',
        '"v4_vc_only"',
    ):
        assert name in source
    assert "route_trace_metrics(" in source


def test_k1_sol_reduction_probe_reuses_frozen_k3_case_geometry():
    source = _probe_source()
    assert "k3_fixture.K3_CALIBRATION_CASES" in source
    assert "k3_fixture._resolve_start" in source


def test_k1_sol_reduction_probe_binds_source_and_backend_provenance():
    source = _probe_source()
    assert "source = replay_fixture._probe_source_identity(_REPO_ROOT)" in source
    assert "selector_provenance = k3_fixture._native_selector_provenance(device)" in source
    assert "refresh the Sol #23 Patcher overlay" in source
    assert '"selector_provenance": selector_provenance' in source
    assert '"critical_source_sha256"' in source
    assert '"sol_h3/keyless_route_summary.py"' in source


def test_k1_sol_reduction_probe_isolates_row_route_from_fused_reduction():
    source = _probe_source()
    assert "materialized_sol_reduction_v4_route_diagnostic(" in source
    assert '"v4_reduction_isolation": v4_reduction_isolation' in source
    assert '"materialized_existing_sol_rc"' in source
    assert '"fused_rc_vs_materialized_existing_sol_rc"' in source
    assert '"comfy_rms_rope_backend"' in source
    assert '"a full K1-v4 route is materialized only in the diagnostic/oracle "' in source
    assert '"path to isolate row-route arithmetic from fused reduction"' in source


def test_k1_sol_reduction_probe_keeps_production_promotion_false():
    source = _probe_source()
    assert (
        '"K1 v4 is experimental and not connected to production provider dispatch"'
        in source
    )
    assert '"no fused selected-route CuTe production mainloop, decoded-media, "' in source
    assert '"sampler, or end-to-end performance evidence"' in source
