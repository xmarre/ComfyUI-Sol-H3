from __future__ import annotations

from pathlib import Path


def _probe_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_real_h3_replay_probe.py"
    ).read_text(encoding="utf-8")


def test_probe_binds_expensive_run_to_exact_source_commit():
    source = _probe_source()
    assert '--expected-probe-commit' in source
    assert 'probe_source = _probe_source_identity(_REPO_ROOT)' in source
    assert 'stale Sol-H3 replay source' in source


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
