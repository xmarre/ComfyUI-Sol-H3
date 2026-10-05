from __future__ import annotations

from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_k3_selector_holdout.py"
    ).read_text(encoding="utf-8")


def _fractional_interval(
    total_rows: int,
    numerator: int,
    denominator: int,
    span: int,
) -> tuple[int, int]:
    start = ((total_rows - span) * numerator) // denominator
    return start, start + span


def test_k3_v2_holdout_freezes_exact_gate_before_execution():
    source = _source()
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k3-selector-holdout-probe-v1"' in source
    assert 'EVIDENCE_CONTRACT = "sol-h3-keyless-k3-selector-holdout-v1"' in source
    assert 'EXACT_GATE_CONTRACT = "sol-h3-keyless-k3-v2-exact-gate-v1"' in source
    assert (
        '"ea2daae99e6c3df515b9d8a03a9fb4b8b2920601d843b7363fb63309d7eff836"'
        in source
    )
    assert '"calibration_case_count": 12' in source
    assert '"route_centroid_max_abs": 0.0' in source
    assert '"raw_value_sum_max_abs": 0.0' in source
    assert '"threshold_max_abs": 0.0' in source
    assert '"route_trace_differing_bits": 0' in source
    assert '"output_max_abs": 0.0' in source
    assert '"thresholds_frozen": True' in source
    assert '"exact_gate_frozen": True' in source


def test_k3_v2_holdout_uses_corrected_k1_identity_and_released_selector():
    source = _source()
    assert "candidate_rc, candidate_vc = route_summary_sol_reduction(" in source
    assert "k1_route = materialized_sol_reduction_v4_route_diagnostic(" in source
    assert "reference_rc, reference_vc, reference_threshold = prepare(" in source
    assert "candidate_output, candidate_trace = run_materialized_exact_selector_isolation(" in source
    assert "reference_output, reference_trace = run_materialized_exact_selector_isolation(" in source
    assert "route_trace_metrics(" in source
    assert "_exact_tensor_pass(" in source
    assert '"passes": passes' in source


def test_k3_v2_holdout_cases_are_predeclared_and_disjoint_from_calibration():
    source = _source()
    assert '"two-twentyfourth-385x2049-nosink", 2, 24, 385, 2049, 0, 0' in source
    assert '"three-sixteenth-513x513-prefix64", 3, 16, 513, 513, 0, 64' in source
    assert '"six-sixteenth-769x1537-offset192", 6, 16, 769, 1537, 512, 192' in source
    assert '"ten-sixteenth-1025x1537-prefix192", 10, 16, 1025, 1537, 0, 192' in source

    total_rows = 24879
    calibration = (
        (0, 1537),
        (5707, 5707 + 2049),
        (10391, 10391 + 4097),
        (16686, 16686 + 8193),
    )
    holdout = (
        (2, 24, 2049),
        (3, 16, 513),
        (6, 16, 1537),
        (10, 16, 1537),
    )
    holdout_intervals = [
        _fractional_interval(total_rows, numerator, denominator, span)
        for numerator, denominator, span in holdout
    ]

    for start, stop in holdout_intervals:
        assert all(
            max(start, cal_start) >= min(stop, cal_stop)
            for cal_start, cal_stop in calibration
        )

    for index, (start, stop) in enumerate(holdout_intervals):
        assert all(
            max(start, other_start) >= min(stop, other_stop)
            for other_index, (other_start, other_stop)
            in enumerate(holdout_intervals)
            if other_index != index
        )


def test_k3_v2_holdout_binds_source_provenance_and_lineage():
    source = _source()
    assert "source = replay_fixture._probe_source_identity(_REPO_ROOT)" in source
    assert '"critical_source_sha256"' in source
    assert '"tools/keyless_k3_selector_probe.py"' in source
    assert '"sol_h3/keyless_selector.py"' in source
    assert '"sol_h3/keyless_route_summary.py"' in source
    assert "K1_V9_EVIDENCE_SHA256" in source
    assert "K2_V3_HOLDOUT_SHA256" in source
    assert "CALIBRATION_EVIDENCE_SHA256" in source
    assert "HISTORICAL_K3_V1_SHA256" in source
    assert "refresh the complete Patcher stack through PR #25" in source


def test_k3_v2_holdout_is_fail_closed_and_not_promotion_evidence():
    source = _source()
    assert "float(args.tau) != FROZEN_TAU" in source
    assert 'backend' not in source or "_native_selector_provenance" in source
    assert '"promotion_evidence": False' in source
    assert '"all_holdout_cases_pass": all_holdout_cases_pass' in source
    assert "if not all_holdout_cases_pass:" in source
    assert "real-H3 K3-v2 holdout violated the frozen exact gate" in source
    assert "no K4 fused selected-route CuTe transform" in source
