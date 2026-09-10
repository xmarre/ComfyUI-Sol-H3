import pytest
import torch

from sol_h3.attention_measure import bind, digest, normalize


def request():
    return {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "q_rows": 30,
        "kv_rows": 30,
        "video_start": 6,
        "temporal": 3,
        "prefix_t": 1,
        "source_grid": [2, 3],
        "prefix_grid": [3, 4],
        "segments": [
            {"start": 0, "stop": 6, "mass_num": 1, "mass_den": 1},
            {"start": 6, "stop": 18, "mass_num": 6, "mass_den": 12},
            {"start": 18, "stop": 30, "mass_num": 1, "mass_den": 1},
        ],
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
    }


def test_contract_normalizes_measure_identity_and_rejects_boolean_counts():
    normalized = normalize(request())
    assert normalized["segments"][1]["mass_num"] == 1
    assert normalized["segments"][1]["mass_den"] == 2
    equivalent = request()
    equivalent["segments"][1]["mass_num"] = 1
    equivalent["segments"][1]["mass_den"] = 2
    assert digest(request()) == digest(equivalent)
    bad = request()
    bad["prefix_t"] = True
    with pytest.raises(TypeError):
        normalize(bad)


def test_binding_keeps_all_rows_and_forces_every_weighted_boundary_block_exact():
    plan = bind(request(), q_rows=30, kv_rows=30, device="cpu", block_size=8, existing_sink=(0, 1))
    assert plan.q_rows == plan.kv_rows == 30
    assert plan.exact_k_blocks == (0, 3)
    assert plan.key_log_measure.shape == (30,)
    weights = plan.key_log_measure.double().exp()
    assert weights[:6].sum().item() == pytest.approx(6.0)
    assert weights[6:18].sum().item() == pytest.approx(6.0)
    assert weights[18:24].sum().item() == pytest.approx(6.0)
    assert torch.count_nonzero(plan.key_log_measure[6:18]).item() == 12


def test_binding_rejects_stale_runtime_geometry():
    with pytest.raises(ValueError):
        bind(request(), q_rows=29, kv_rows=29, device="cpu")
