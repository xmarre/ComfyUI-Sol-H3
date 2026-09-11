from types import SimpleNamespace

import pytest
import torch

from sol_h3 import weighted_measure


core = pytest.importorskip("comfy.attention_measure")


def _fixture():
    rows = 202
    layout = SimpleNamespace(seq_len=rows, segments=((0, 10, "text"), (10, rows, "video")))
    request = {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
        "q_rows": rows,
        "kv_rows": rows,
        "video_start": 10,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid": [8, 8],
        "prefix_grid": [8, 16],
        "segments": [
            {"start": 0, "stop": 10, "mass_num": 1, "mass_den": 1},
            {"start": 10, "stop": 138, "mass_num": 1, "mass_den": 2},
            {"start": 138, "stop": rows, "mass_num": 1, "mass_den": 1},
        ],
    }
    common = dict(
        block_index=3,
        layout=layout,
        q_rows=rows,
        kv_rows=rows,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        head_dim=128,
        existing_sink=(0, 1),
        external_sequence=None,
        preprocess_identity="route-profile-test",
    )
    return rows, request, common


def _external_sequence():
    return {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 138,
        "sequence_rows": 202,
        "video_start": 10,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 64,
        "prefix_rows_per_frame": 128,
    }


def test_dense_and_sparse_routes_bind_distinct_plans_but_share_measure_buffer():
    rows, request, common = _fixture()
    options = {}
    weighted_measure.register(options, refresh_owned=True)
    state = SimpleNamespace(measure_plans={}, measure_biases={})

    sparse_plan = weighted_measure.prepare(
        state,
        options,
        request,
        **common,
        implementation_profile=weighted_measure.SPARSE_IMPLEMENTATION_PROFILE,
        numerical_route=weighted_measure.SPARSE_NUMERICAL_ROUTE,
    )
    dense_plan = weighted_measure.prepare(
        state,
        options,
        request,
        **common,
        implementation_profile=weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        numerical_route=weighted_measure.DENSE_NUMERICAL_ROUTE,
    )

    assert sparse_plan is not dense_plan
    assert sparse_plan.implementation_profile == "weighted_exact_blocks_v1"
    assert dense_plan.implementation_profile == "dense_exact_v1"
    assert sparse_plan.numerical_route == weighted_measure.SPARSE_NUMERICAL_ROUTE
    assert dense_plan.numerical_route == weighted_measure.DENSE_NUMERICAL_ROUTE
    assert sparse_plan.exact_k_block_range == dense_plan.exact_k_block_range == (0, 3)
    assert sparse_plan.key_log_measure is dense_plan.key_log_measure
    assert sparse_plan.key_log_measure.shape == (rows,)
    assert len(state.measure_plans) == 2
    assert len(state.measure_biases) == 1


def test_cached_plan_revalidates_current_layout_and_external_sequence():
    _, request, common = _fixture()
    options = {}
    weighted_measure.register(options, refresh_owned=True)
    state = SimpleNamespace(measure_plans={}, measure_biases={})
    external = _external_sequence()

    first = weighted_measure.prepare(
        state,
        options,
        request,
        **{**common, "external_sequence": external},
    )
    second = weighted_measure.prepare(
        state,
        options,
        request,
        **{**common, "external_sequence": dict(external)},
    )
    assert second is first
    assert len(state.measure_plans) == 1
    assert len(state.measure_biases) == 1

    stale_external = dict(external)
    stale_external["prefix_rows_per_frame"] = 64
    with pytest.raises(ValueError, match="external-sequence"):
        weighted_measure.prepare(
            state,
            options,
            request,
            **{**common, "external_sequence": stale_external},
        )

    stale_layout = SimpleNamespace(
        seq_len=202,
        segments=((0, 11, "text"), (11, 202, "video")),
    )
    with pytest.raises(ValueError, match="layout"):
        weighted_measure.prepare(
            state,
            options,
            request,
            **{**common, "layout": stale_layout, "external_sequence": external},
        )

    assert len(state.measure_plans) == 1
    assert len(state.measure_biases) == 1


def test_invalid_route_profile_pair_fails_before_capability_dispatch():
    _, request, common = _fixture()
    options = {}
    weighted_measure.register(options, refresh_owned=True)
    state = SimpleNamespace(measure_plans={}, measure_biases={})

    with pytest.raises(ValueError, match="route/profile pair"):
        weighted_measure.prepare(
            state,
            options,
            request,
            **common,
            implementation_profile=weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
            numerical_route=weighted_measure.SPARSE_NUMERICAL_ROUTE,
        )
    assert not state.measure_plans
    assert not state.measure_biases
