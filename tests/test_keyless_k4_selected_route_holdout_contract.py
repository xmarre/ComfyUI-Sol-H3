from __future__ import annotations

from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_k4_selected_route_holdout.py"
    ).read_text(encoding="utf-8")


def _fractional_interval(
    total_rows: int,
    numerator: int,
    denominator: int,
    span: int,
) -> tuple[int, int]:
    start = ((total_rows - span) * numerator) // denominator
    return start, start + span


def test_k4_holdout_freezes_gate_before_execution():
    source = _source()
    assert (
        'PROBE_CONTRACT = "sol-h3-keyless-k4-selected-route-holdout-probe-v1"'
        in source
    )
    assert (
        'EVIDENCE_CONTRACT = "sol-h3-keyless-k4-selected-route-holdout-v1"'
        in source
    )
    assert 'GATE_CONTRACT = "sol-h3-keyless-k4-selected-route-gate-v1"' in source
    assert (
        '"056f36afaac76081911e5225a23f8f9a05b635398c4b62e6c501412db156e3e1"'
        in source
    )
    assert '"candidate_rel_l2_max": 1.5e-4' in source
    assert '"candidate_max_abs_over_want_abs_max": 0.006' in source
    assert '"candidate_mean_abs_over_want_mean_abs": 3.5e-6' in source
    assert '"candidate_worst_bf16_ulps_max": 1.0' in source
    assert '"thresholds_frozen": True' in source
    assert '"composition_gate_frozen": True' in source


def test_k4_holdout_reuses_identical_calibration_arithmetic():
    source = _source()
    assert "calibration_probe._run_block(" in source
    assert "calibration_probe._extrema(blocks)" in source
    assert "K4_COMPOSITION_HOLDOUT_CASES" in source
    assert "original = calibration_probe.k3_holdout.K3_V2_HOLDOUT_CASES" in source
    assert "finally:" in source


def test_k4_holdout_cases_are_predeclared_and_disjoint():
    source = _source()
    assert '"zero-449x1025-nosink", 0, 1, 449, 1025, 0, 0' in source
    assert (
        '"five-sixteenth-641x641-prefix128", 5, 16, 641, 641, 0, 128'
        in source
    )
    assert (
        '"eight-sixteenth-513x4097-offset256", 8, 16, 513, 4097, 3072, 256'
        in source
    )
    assert (
        '"fifteen-sixteenth-1153x1793-prefix256", 15, 16, 1153, 1793, 0, 256'
        in source
    )

    total_rows = 24879
    calibration = (
        (1902, 3951),
        (4568, 5081),
        (8753, 10290),
        (14588, 16125),
    )
    holdout_cases = (
        (0, 1, 1025),
        (5, 16, 641),
        (8, 16, 4097),
        (15, 16, 1793),
    )
    intervals = [
        _fractional_interval(total_rows, numerator, denominator, span)
        for numerator, denominator, span in holdout_cases
    ]

    for start, stop in intervals:
        assert all(
            max(start, cal_start) >= min(stop, cal_stop)
            for cal_start, cal_stop in calibration
        )
    for index, (start, stop) in enumerate(intervals):
        assert all(
            max(start, other_start) >= min(stop, other_stop)
            for other_index, (other_start, other_stop) in enumerate(intervals)
            if other_index != index
        )

    assert (4097 + 63) // 64 == 65


def test_k4_holdout_is_fail_closed_and_not_promotion_evidence():
    source = _source()
    assert "--calibration-evidence" in source
    assert "_validate_calibration_evidence(calibration_path)" in source
    assert "float(args.tau) != FROZEN_TAU" in source
    assert '"promotion_evidence": False' in source
    assert '"all_holdout_cases_pass": all_holdout_cases_pass' in source
    assert "if not all_holdout_cases_pass:" in source
    assert "violated the frozen composition gate" in source
    assert '"candidate_global_route_tensor": False' in source
    assert '"no fused Keyless receipt/history identity is enabled"' in source
