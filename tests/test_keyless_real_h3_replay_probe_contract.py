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
    assert 'PROBE_CONTRACT = "sol-h3-keyless-v2-calibration-campaign-v5"' in source
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
