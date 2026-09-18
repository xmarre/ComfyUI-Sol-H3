from __future__ import annotations

from pathlib import Path


def _probe_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_real_h3_replay_probe.py"
    ).read_text(encoding="utf-8")


def test_probe_binds_expensive_run_to_patcher_safe_contract():
    source = _probe_source()
    assert 'PROBE_CONTRACT = "sol-h3-keyless-v2-holdout-v6"' in source
    assert '--expected-probe-contract' in source
    assert 'probe_source = _probe_source_identity(_REPO_ROOT)' in source
    assert 'stale Sol-H3 replay probe' in source


def test_probe_can_write_complete_json_without_stdout_redirection():
    source = _probe_source()
    assert '--output-json' in source
    assert '_write_json_atomic(output_path, result)' in source
    assert '"diagnostic_report"' in source


def test_diagnostic_only_does_not_change_recorded_gate():
    source = _probe_source()
    assert '"all_blocks_pass": all(' in source
    assert '"promotion_evidence": False' in source
    assert 'replay_requires_process_failure(' in source


def test_probe_emits_fp32_rope_isolation_fields():
    source = _probe_source()
    assert '"rope_dtype"' in source
    assert '"standalone_comfy_backend"' in source
    assert '"fp32_rope_from_comfy_norm_vs_comfy_fused_route"' in source
    assert '"native_fp32_rope_route_vs_comfy_route"' in source
    assert '"diagnostic_fp32_rope_route"' in source


def test_probe_marks_v1_envelope_as_historical_for_v2():
    source = _probe_source()
    assert '"arithmetic_envelope_scope"' in source
    assert "fresh v2 calibration" in source


def test_probe_exposes_threshold_free_v2_calibration_campaign():
    source = _probe_source()
    assert "--calibration-v2" in source
    assert 'V2_CALIBRATION_CONTRACT = "sol-h3-keyless-v2-calibration-v1"' in source
    assert 'V2_K1_CONTRACT = "sol-h3-keyless-route-summary-v2"' in source
    assert 'V2_K2_CONTRACT = "sol-h3-keyless-exact-allselected-v2"' in source
    assert "Refresh the complete " in source
    assert "V2_CALIBRATION_CASES = (" in source
    assert '"thresholds_frozen": False' in source
    assert '"historical_v1_envelope_evaluated": False' in source
    assert '"observed_extrema": _calibration_v2_extrema(calibration_blocks)' in source
    assert '"max_abs_over_want_abs_max": maximum_scale_ratio(' in source
    assert '"mean_abs_over_want_mean_abs": maximum_scale_ratio(' in source
    assert '"mode": "k1-k2-v2-real-h3-calibration"' in source



def test_probe_exposes_frozen_v2_holdout_campaign():
    source = _probe_source()
    assert '--holdout-v2' in source
    assert 'V2_HOLDOUT_CONTRACT = "sol-h3-keyless-v2-holdout-v1"' in source
    assert "V2_HOLDOUT_CASES = (" in source
    assert '"eighth-95x95", 1, 8, 95, 95' in source
    assert '"three-eighth-96x96", 3, 8, 96, 96' in source
    assert '"five-eighth-97x97", 5, 8, 97, 97' in source
    assert '"seven-eighth-191x191", 7, 8, 191, 191' in source
    assert '"five-eighth-383x385", 5, 8, 383, 385' in source
    assert '"seven-eighth-385x383", 7, 8, 385, 383' in source
    assert '"five-eighth-769x1025", 5, 8, 769, 1025' in source
    assert '"three-eighth-1025x769", 3, 8, 1025, 769' in source
    assert '"v2_envelope": v2_envelope_dict()' in source
    assert '"all_holdout_cases_pass": holdout_pass' in source
    assert '"thresholds_frozen": True' in source
    assert "real-H3 v2 holdout exceeded the frozen arithmetic envelope" in source


def test_holdout_windows_are_disjoint_from_calibration_windows_for_bound_capture():
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
