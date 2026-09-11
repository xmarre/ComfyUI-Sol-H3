from types import SimpleNamespace

import torch

from sol_h3 import runtime, sparse, weighted_measure
from sol_h3.contracts import Config
from sol_h3.runtime import BlockPatch, Request, _FORWARD


ROWS = 202
HEADS = 2
DIM = 128


def _contracts():
    layout = SimpleNamespace(
        seq_len=ROWS,
        segments=((0, 10, "text"), (10, ROWS, "video")),
    )
    external = {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 138,
        "sequence_rows": ROWS,
        "video_start": 10,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 64,
        "prefix_rows_per_frame": 128,
    }
    measure = {"api": 1, "operator": "key_log_measure"}
    return layout, external, measure


def _plan(profile, route):
    return SimpleNamespace(
        key_log_measure=torch.zeros(ROWS, dtype=torch.float32),
        exact_k_block_range=(0, 3),
        semantic_digest="measure-digest",
        owner_generation="owner-generation",
        implementation_profile=profile,
        numerical_route=route,
        q_rows=ROWS,
        kv_rows=ROWS,
        exact_range_digest="exact-range-digest",
        preprocess_digest="preprocess-digest",
    )


def _run_block(monkeypatch, cfg, *, sparse_impl, dense_impl, include_external=True):
    state = Request(cfg)
    layout, external, measure = _contracts()
    q, k, v = (
        torch.randn(1, HEADS, ROWS, DIM, dtype=torch.bfloat16)
        for _ in range(3)
    )

    monkeypatch.setattr(runtime, "_shape_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(weighted_measure, "preprocess_digest", lambda provider: "preprocess-digest")

    prepare_calls = []
    plans = []

    def prepare(state_arg, options, request, **kwargs):
        prepare_calls.append((state_arg, options, request, kwargs))
        assert kwargs["q_rows"] == ROWS
        assert kwargs["kv_rows"] == ROWS
        assert kwargs["existing_sink"] == (0, 1)
        assert kwargs["preprocess_identity"] == "preprocess-digest"
        profile = kwargs["implementation_profile"]
        route = kwargs["numerical_route"]
        plan = _plan(profile, route)
        plans.append(plan)
        return plan

    monkeypatch.setattr(weighted_measure, "prepare", prepare)
    monkeypatch.setattr(weighted_measure, "dense", dense_impl)
    monkeypatch.setattr(sparse, "attention", sparse_impl)
    monkeypatch.setattr(
        runtime,
        "reduce_kv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("weighted route must never enter legacy K/V reduction")
        ),
    )

    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=None))])
    token = _FORWARD.set((model, state, 0, set(), []))

    def original_block(call_args):
        options = call_args["transformer_options"]
        override = options["optimized_attention_override"]
        return override(
            None,
            q,
            k,
            v,
            HEADS,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=False,
            transformer_options=options,
        )

    receipts = []
    options = {
        "minimax_h3_layout": layout,
        weighted_measure.ATTENTION_MEASURE_KEY: measure,
        "attention_backend_receipts_v1": receipts,
    }
    if include_external:
        options["vdn_h3_external_sequence_v1"] = external
    try:
        result = BlockPatch(0, cfg)(
            {"transformer_options": options, "layout": layout},
            {"original_block": original_block},
        )
    finally:
        _FORWARD.reset(token)

    return result, state, plans, prepare_calls, receipts


def test_weighted_runtime_keeps_all_kv_rows_and_separates_q_prefix_from_k_sink(monkeypatch):
    calls = []

    def sparse_impl(q, k, v, prefix, config, state, **kwargs):
        calls.append((q.shape, k.shape, v.shape, prefix, kwargs))
        assert prefix == 10
        assert k.shape[2] == v.shape[2] == ROWS
        assert kwargs["key_bias"].shape == (ROWS,)
        assert kwargs["exact_k_blocks"] == (0, 3)
        assert kwargs["calibration_identity"] == "measure-digest"
        assert kwargs["dense_attention"] is not None
        state.sparse_calls += 1
        return torch.zeros(1, ROWS, HEADS * DIM, dtype=q.dtype)

    def dense_impl(*args, **kwargs):
        raise AssertionError("hot weighted sparse route must not take dense fallback")

    result, state, plans, _, receipts = _run_block(
        monkeypatch,
        Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0),
        sparse_impl=sparse_impl,
        dense_impl=dense_impl,
    )

    assert result.shape == (1, ROWS, HEADS * DIM)
    assert len(calls) == 1
    assert [(p.implementation_profile, p.numerical_route) for p in plans] == [(
        weighted_measure.SPARSE_IMPLEMENTATION_PROFILE,
        weighted_measure.SPARSE_NUMERICAL_ROUTE,
    )]
    assert receipts[0][3][4:6] == (
        weighted_measure.SPARSE_IMPLEMENTATION_PROFILE,
        weighted_measure.SPARSE_NUMERICAL_ROUTE,
    )
    assert state.external_mixed_weighted_measure_calls == 1
    assert state.external_mixed_weighted_measure_q_rows == ROWS
    assert state.external_mixed_weighted_measure_kv_rows == ROWS
    assert state.external_mixed_measure_calls == 0


def test_weighted_dense_warmup_preserves_all_rows_and_binds_dense_receipt(monkeypatch):
    dense_calls = []

    def sparse_impl(*args, **kwargs):
        raise AssertionError("dense warmup must not dispatch sparse attention")

    def dense_impl(q, k, v, heads, plan, *, scale=None, output_heads=False):
        dense_calls.append((q.shape, k.shape, v.shape, heads, plan, scale, output_heads))
        assert k.shape[2] == v.shape[2] == ROWS
        assert plan.key_log_measure.shape == (ROWS,)
        assert output_heads is False
        return torch.zeros(1, ROWS, HEADS * DIM, dtype=q.dtype)

    result, state, plans, _, receipts = _run_block(
        monkeypatch,
        Config(exact=False, backend="sol", dense_evaluations=1, dense_layers=0),
        sparse_impl=sparse_impl,
        dense_impl=dense_impl,
    )

    assert result.shape == (1, ROWS, HEADS * DIM)
    assert len(dense_calls) == 1
    assert dense_calls[0][4] is plans[0]
    assert [(p.implementation_profile, p.numerical_route) for p in plans] == [(
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )]
    assert receipts[0][3][4:6] == (
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )
    assert state.dense_calls == 1
    assert state.external_mixed_weighted_measure_calls == 1
    assert state.external_mixed_weighted_measure_kv_rows == ROWS


def test_weighted_kernel_unavailable_rebinds_dense_fallback_receipt(monkeypatch):
    dense_calls = []

    def sparse_impl(*args, **kwargs):
        raise sparse.KernelUnavailable("test-unavailable")

    def dense_impl(q, k, v, heads, plan, *, scale=None, output_heads=False):
        dense_calls.append((k.shape[2], plan, output_heads))
        return torch.zeros(1, ROWS, HEADS * DIM, dtype=q.dtype)

    result, state, plans, _, receipts = _run_block(
        monkeypatch,
        Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0),
        sparse_impl=sparse_impl,
        dense_impl=dense_impl,
    )

    assert result.shape == (1, ROWS, HEADS * DIM)
    assert len(plans) == 2
    assert (plans[0].implementation_profile, plans[0].numerical_route) == (
        weighted_measure.SPARSE_IMPLEMENTATION_PROFILE,
        weighted_measure.SPARSE_NUMERICAL_ROUTE,
    )
    assert (plans[1].implementation_profile, plans[1].numerical_route) == (
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )
    assert dense_calls == [(ROWS, plans[1], False)]
    assert receipts[0][3][4:6] == (
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )
    assert state.external_mixed_weighted_measure_calls == 1
    assert state.external_mixed_weighted_measure_q_rows == ROWS
    assert state.external_mixed_weighted_measure_kv_rows == ROWS
    assert state.external_mixed_measure_calls == 0
    assert state.fallbacks["kernel_unavailable:test-unavailable"] == 1


def test_generic_weighted_measure_does_not_require_vdn_external_sequence(monkeypatch):
    calls = []

    def sparse_impl(q, k, v, prefix, config, state, **kwargs):
        calls.append((prefix, k.shape[2], kwargs["exact_k_blocks"]))
        state.sparse_calls += 1
        return torch.zeros(1, ROWS, HEADS * DIM, dtype=q.dtype)

    def dense_impl(*args, **kwargs):
        raise AssertionError("generic all-row weighted route unexpectedly fell back to dense")

    result, state, plans, _, _ = _run_block(
        monkeypatch,
        Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0),
        sparse_impl=sparse_impl,
        dense_impl=dense_impl,
        include_external=False,
    )

    assert result.shape == (1, ROWS, HEADS * DIM)
    assert calls == [(10, ROWS, (0, 3))]
    assert plans[0].numerical_route == weighted_measure.SPARSE_NUMERICAL_ROUTE
    assert state.external_mixed_weighted_measure_calls == 1
