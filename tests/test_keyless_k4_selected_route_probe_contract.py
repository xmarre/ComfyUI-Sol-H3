from __future__ import annotations

from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_k4_selected_route_probe.py"
    ).read_text(encoding="utf-8")


def _module_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "sol_h3"
        / "keyless_sparse_composition.py"
    ).read_text(encoding="utf-8")


def test_k4_probe_contract_and_lineage_are_explicit():
    source = _source()
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k4-selected-route-probe-v1"' in source
    assert (
        'EVIDENCE_CONTRACT = "sol-h3-keyless-k4-selected-route-calibration-v1"'
        in source
    )
    assert "K3_HOLDOUT_SHA256" in source
    assert "K3_CALIBRATION_SHA256" in source
    assert "K2_HOLDOUT_SHA256" in source
    assert "K1_V9_SHA256" in source
    assert '"promotion_evidence": False' in source
    assert '"thresholds_frozen": False' in source


def test_k4_probe_uses_proven_k3_trace_but_candidate_has_no_global_route():
    source = _source()
    module = _module_source()
    assert "reference_output, reference_trace = run_materialized_exact_selector_isolation(" in source
    assert "selected_route_composition(" in source
    assert "reference_trace," in source
    assert "materialized_sol_reduction_v4_route_diagnostic(" in module
    assert "raw_tile," in module
    assert '"candidate_global_route_tensor": False' in source
    assert "routing.materialize" not in module


def test_k4_probe_keeps_provider_promotion_closed():
    source = _source()
    assert '"no provider dispatch or fused receipt/history identity is enabled"' in source
    assert '"no vendored CuTe source is changed by this diagnostic layer"' in source
    assert (
        '"the route trace is supplied by the already-proven materialized K3 oracle"'
        in source
    )
