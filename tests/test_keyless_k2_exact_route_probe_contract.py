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
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k2-exact-route-probe-v2"' in source
    assert (
        'CALIBRATION_EVIDENCE_CONTRACT = "sol-h3-keyless-k2-exact-route-calibration-v1"'
        in source
    )
    assert (
        'HOLDOUT_EVIDENCE_CONTRACT = "sol-h3-keyless-k2-exact-route-holdout-v1"'
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
    assert '"thresholds_frozen": thresholds_frozen' in source


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
    assert '"sol_h3/keyless_real_h3_replay.py"' in source
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


def test_k2_v3_holdout_is_frozen_before_execution():
    source = _source()
    assert "--holdout-v3" in source
    assert "--calibration-v3" in source
    assert "V3_HOLDOUT_CASES = (" in source
    assert '"k2_v3_calibration_sha256"' in source
    assert '"v3_envelope": k2_v3_envelope_dict() if args.holdout_v3 else None' in source
    assert "scale_aware_metric_within_limit(" in source
    assert '"all_holdout_cases_pass": holdout_pass' in source
    assert "real-H3 K2-v3 holdout exceeded the frozen arithmetic envelope" in source


def test_k2_v3_holdout_windows_are_disjoint_from_calibration_windows():
    total_rows = 24879
    calibration = (
        ("head", 63),
        ("head", 64),
        ("head", 65),
        ("quarter", 127),
        ("quarter", 128),
        ("quarter", 129),
        ("middle", 255),
        ("middle", 256),
        ("middle", 257),
        ("three_quarter", 511),
        ("tail", 511),
        ("tail", 1537),
    )
    holdout = (
        (1, 8, 95),
        (3, 8, 96),
        (5, 8, 97),
        (7, 8, 191),
        (1, 8, 192),
        (3, 8, 193),
        (5, 8, 385),
        (7, 8, 385),
        (5, 8, 1025),
        (3, 8, 1025),
    )

    def calibration_start(anchor, span):
        available = total_rows - span
        return {
            "head": 0,
            "quarter": available // 4,
            "middle": available // 2,
            "three_quarter": (available * 3) // 4,
            "tail": available,
        }[anchor]

    calibration_intervals = [
        (calibration_start(anchor, span), calibration_start(anchor, span) + span)
        for anchor, span in calibration
    ]
    for numerator, denominator, span in holdout:
        start = ((total_rows - span) * numerator) // denominator
        stop = start + span
        assert all(
            max(start, cal_start) >= min(stop, cal_stop)
            for cal_start, cal_stop in calibration_intervals
        )
