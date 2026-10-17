from __future__ import annotations

from pathlib import Path


def _probe_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "keyless_k3_selector_probe.py"
    ).read_text(encoding="utf-8")


def test_k3_probe_contract_and_evidence_identity_are_frozen():
    source = _probe_source()
    assert 'PROBE_CONTRACT = "sol-h3-keyless-k3-selector-probe-v2"' in source
    assert 'EVIDENCE_CONTRACT = "sol-h3-keyless-k3-selector-calibration-v2"' in source
    assert (
        'REQUIRED_K1_CONTRACT = "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"'
        in source
    )
    assert (
        'REQUIRED_K3_CONTRACT = "sol-h3-keyless-selector-k3-v2-k1-route-identity"'
        in source
    )
    assert '"promotion_evidence": False' in source
    assert '"thresholds_frozen": False' in source


def test_k3_probe_predeclares_real_h3_selector_cases():
    source = _probe_source()
    assert '("head-257x1537-nosink", "head", 257, 1537, 0, 0)' in source
    assert '("quarter-385x2049-prefix128", "quarter", 385, 2049, 0, 128)' in source
    assert '("middle-769x4097-offset192", "middle", 769, 4097, 1024, 192)' in source
    assert '("tail-1025x8193-prefix256", "tail", 1025, 8193, 0, 256)' in source


def test_k3_probe_compares_candidate_and_materialized_reference_selector():
    source = _probe_source()
    assert "candidate_rc, candidate_vc = route_summary_sol_reduction(" in source
    assert "k1_route = materialized_sol_reduction_v4_route_diagnostic(" in source
    assert "reference_rc, reference_vc, reference_threshold = prepare(" in source
    assert "qb," in source
    assert "kb," in source
    assert "candidate_output, candidate_trace = run_materialized_exact_selector_isolation(" in source
    assert "reference_output, reference_trace = run_materialized_exact_selector_isolation(" in source
    assert "route_trace_metrics(" in source
    assert "kv_rows=v_rows" in source


def test_k3_probe_binds_patcher_source_before_expensive_work():
    source = _probe_source()
    assert "--expected-probe-contract" in source
    assert "probe_source = replay_fixture._probe_source_identity(_REPO_ROOT)" in source
    assert "K3 calibration requires a clean tracked Sol-H3 worktree" in source
    assert "refresh the complete Patcher stack through the K3-v2 PR" in source
    assert '"critical_source_sha256"' in source
    assert '"sol_h3/keyless_selector.py"' in source
    assert '"sol_h3/keyless_route_summary.py"' in source


def test_k3_probe_does_not_claim_no_route_or_production_promotion():
    source = _probe_source()
    assert (
        '"exact-block K remains the globally materialized exact K1 row route "'
        in source
    )
    assert (
        '"candidate K1-v6 RC/VC come from raw V through bounded-b8 route scratch"'
        in source
    )
    assert (
        '"no production provider promotion, fused selected-route CuTe transform, decoded-media, sampler, or end-to-end performance evidence"'
        in source
    )



def test_k3_probe_verifies_native_source_and_backend_provenance():
    source = _probe_source()
    assert "selector_provenance = _native_selector_provenance(device)" in source
    assert "manifest = verify_source()" in source
    assert "backend = get_sol_attn_backend(device)" in source
    assert 'backend != "cute_sm120"' in source
    assert '"selector_provenance": selector_provenance' in source
    assert '"nvidia-cutlass-dsl"' in source
    assert '"cuda-python"' in source
    assert '"triton"' in source


def test_k3_v2_probe_records_corrected_lineage_and_exactness_controls():
    source = _probe_source()
    assert "K1_V9_EVIDENCE_SHA256" in source
    assert "K2_V3_HOLDOUT_SHA256" in source
    assert "HISTORICAL_K3_V1_SHA256" in source
    assert '"all_route_centroids_exact"' in source
    assert '"all_raw_value_sums_exact"' in source
    assert '"all_thresholds_exact"' in source
    assert '"all_route_traces_equal"' in source
    assert '"all_outputs_exact"' in source
    assert '"k1_route_vs_comfy_route"' in source
