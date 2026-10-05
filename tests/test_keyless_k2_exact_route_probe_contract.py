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
    assert '"sixteenth-79x79", 1, 16, 79, 79' in source
    assert '"thirteen-sixteenth-319x321", 13, 16, 319, 321' in source
    assert '"twentyseven-thirtysecond-385x769", 27, 32, 385, 769' in source
    assert '"k2_v3_calibration_sha256"' in source
    assert '"v3_envelope": k2_v3_envelope_dict() if args.holdout_v3 else None' in source
    assert "scale_aware_metric_within_limit(" in source
    assert '"all_holdout_cases_pass": holdout_pass' in source
    assert "real-H3 K2-v3 holdout exceeded the frozen arithmetic envelope" in source


def test_k2_v3_holdout_windows_are_new_and_disjoint():
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
    historical_holdout = (
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
    v3_holdout = (
        (1, 16, 79),
        (3, 16, 80),
        (5, 16, 81),
        (7, 16, 159),
        (9, 16, 160),
        (11, 16, 161),
        (13, 16, 321),
        (15, 16, 257),
        (5, 32, 513),
        (27, 32, 769),
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

    def fractional_interval(numerator, denominator, span):
        start = ((total_rows - span) * numerator) // denominator
        return start, start + span

    calibration_intervals = [
        (calibration_start(anchor, span), calibration_start(anchor, span) + span)
        for anchor, span in calibration
    ]
    historical_intervals = [
        fractional_interval(numerator, denominator, span)
        for numerator, denominator, span in historical_holdout
    ]
    for numerator, denominator, span in v3_holdout:
        start, stop = fractional_interval(numerator, denominator, span)
        assert all(
            max(start, cal_start) >= min(stop, cal_stop)
            for cal_start, cal_stop in calibration_intervals
        )
        assert all(
            max(start, old_start) >= min(stop, old_stop)
            for old_start, old_stop in historical_intervals
        )
