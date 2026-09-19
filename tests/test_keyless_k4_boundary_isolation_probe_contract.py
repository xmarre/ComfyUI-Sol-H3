from __future__ import annotations

from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_k4_boundary_isolation_probe.py"
    ).read_text(encoding="utf-8")


def test_k4_boundary_probe_binds_failed_holdout_without_relaxing_gate():
    source = _source()
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k4-boundary-isolation-probe-v1"' in source
    assert 'EVIDENCE_CONTRACT = "sol-h3-keyless-k4-boundary-isolation-v1"' in source
    assert (
        '"0bcf0fa216517cb46ddae569725c71744efa9df8efe953092399cd36fd81295a"'
        in source
    )
    assert '"failed_holdout_gate_relaxed": False' in source
    assert "payload.get("all_holdout_cases_pass") is not False" in source


def test_k4_boundary_probe_compares_same_python_path_with_two_route_sources():
    source = _source()
    assert "bounded_output, bounded_storage = selected_route_composition(" in source
    assert "full_python_output = _full_route_python_composition(" in source
    assert "composition.materialized_sol_reduction_v4_route_diagnostic" in source
    assert "return full_route[row_start:row_stop]" in source
    assert '"bounded_vs_full_route_python"' in source
    assert '"full_route_python_vs_native_sol"' in source
    assert '"bounded_vs_native_sol"' in source


def test_k4_boundary_probe_checks_selected_route_tiles_directly():
    source = _source()
    assert "def _selected_tile_identity(" in source
    assert "exact_heads_from_route_trace(" in source
    assert "torch.count_nonzero(got != want)" in source
    assert '"all_selected_route_tiles_exact"' in source
    assert '"selected_route_tile_mismatch_elements"' in source


def test_k4_boundary_probe_remains_diagnostic_only():
    source = _source()
    assert '"promotion_evidence": False' in source
    assert '"thresholds_frozen": False' in source
    assert '"candidate_global_route_tensor": False' in source
    assert '"full_route_exists_on_oracle_side_only": True' in source
    assert '"no vendored CuTe source is changed"' in source
    assert '"no provider dispatch or fused Keyless receipt/history identity is enabled"' in source
