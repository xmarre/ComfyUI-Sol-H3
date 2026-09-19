from __future__ import annotations

from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_k2_exact_route_probe.py"
    ).read_text(encoding="utf-8")


def test_k2_exact_route_probe_contracts_are_versioned():
    source = _source()
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k2-exact-route-probe-v1"' in source
    assert (
        'EVIDENCE_CONTRACT = "sol-h3-keyless-k2-exact-route-calibration-v1"'
        in source
    )
    assert (
        'REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"'
        in source
    )
    assert (
        'REQUIRED_K2_CONTRACT = "sol-h3-keyless-exact-allselected-v3-k1-route-identity"'
        in source
    )
    assert '"promotion_evidence": False' in source
    assert '"thresholds_frozen": False' in source


def test_k2_exact_route_probe_uses_proven_k1_route_as_primary_oracle():
    source = _source()
    assert "materialized_sol_reduction_v4_route_diagnostic(" in source
    assert "dense_k1 = replay_fixture._dense_attention(q, k1_route, v_raw)" in source
    assert '"candidate_vs_dense_k1_route"' in source
    assert '"k1_route_vs_comfy_route"' in source
    assert '"dense_k1_route_vs_dense_comfy_route"' in source


def test_k2_exact_route_probe_reuses_real_h3_calibration_geometry():
    source = _source()
    assert "replay_fixture.V2_CALIBRATION_CASES" in source
    assert "replay_fixture._resolve_calibration_start" in source
    assert '"case_count": len(flat)' in source


def test_k2_exact_route_probe_binds_source_and_provenance():
    source = _source()
    assert "source = replay_fixture._probe_source_identity(_REPO_ROOT)" in source
    assert '"critical_source_sha256"' in source
    assert '"sol_h3/keyless_exact_attention.py"' in source
    assert '"sol_h3/keyless_route_summary.py"' in source
    assert '"k1_v9_sha256"' in source
    assert "refresh the complete Patcher stack through the K2 PR" in source


def test_k2_exact_route_probe_records_allocation_and_tail_timing():
    source = _source()
    assert "replay_fixture._candidate_allocation(" in source
    assert '"full_materialized_route_bytes"' in source
    assert '"below_full_route"' in source
    assert 'if name == "tail-1025x1537":' in source
    assert "replay_fixture._timings(" in source
    assert '"tail_timing_ms"' in source


def test_k2_exact_route_probe_keeps_promotion_false():
    source = _source()
    assert (
        '"K2 v3 is experimental and not connected to production provider dispatch"'
        in source
    )
    assert (
        '"no vendored CuTe K2 mainloop, decoded-media, sampler, or end-to-end evidence"'
        in source
    )
