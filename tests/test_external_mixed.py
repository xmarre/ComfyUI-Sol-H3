from types import SimpleNamespace

import torch

from sol_h3 import runtime, sparse
from sol_h3.contracts import Config
from sol_h3.runtime import BlockPatch, Request, _FORWARD


def _layout(rows):
    return SimpleNamespace(
        seq_len=rows,
        segments=[(0, 3, "text"), (3, rows, "video")],
        signature=("mixed", rows),
    )


def _contract(**changes):
    value = {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 7,
        "sequence_rows": 9,
        "video_start": 3,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 2,
        "prefix_rows_per_frame": 4,
    }
    value.update(changes)
    return value


def test_external_mixed_contract_derives_only_the_global_sink_prefix():
    assert runtime._external_sequence_prefix(_contract(), _layout(9), 9) == (3, None)
    # Production Flow keeps the native low-grid carrier layout in the payload
    # while the wrapped transformer block executes the larger mixed stream.
    assert runtime._external_sequence_prefix(_contract(), _layout(7), 9) == (3, None)
    assert runtime._external_sequence_prefix(_contract(api=1), _layout(9), 9) == (None, "external_sequence_native")
    assert runtime._external_sequence_prefix(_contract(sequence_rows=10), _layout(9), 9) == (
        None,
        "external_sequence_contract",
    )
    assert runtime._external_sequence_prefix(_contract(video_start=4), _layout(9), 9) == (
        None,
        "external_sequence_contract",
    )
    assert runtime._external_sequence_prefix(_contract(native_sequence_rows=8), _layout(7), 9) == (
        None,
        "external_sequence_contract",
    )


def test_valid_external_mixed_sequence_uses_sol_then_native_grid_resumes(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None))])
    token = _FORWARD.set((model, state, 0, set(), []))
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    calls = []

    def fake_attention(q, k, v, prefix, config, state, **kwargs):
        calls.append((q.shape[2], k.shape[2], prefix))
        state.sparse_calls += 1
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)

    monkeypatch.setattr(sparse, "attention", fake_attention)

    try:
        q9 = torch.zeros(1, 1, 9, 128, dtype=torch.bfloat16)
        opts = {"vdn_h3_external_sequence_v1": _contract()}

        def mixed_block(args):
            to = args["transformer_options"]
            out = to["optimized_attention_override"](
                lambda q, *a, **kw: q.transpose(1, 2).reshape(1, q.shape[2], -1),
                q9,
                q9,
                q9,
                1,
                skip_reshape=True,
                transformer_options=to,
            )
            assert out.shape == (1, 9, 128)
            return {"img": args["img"]}

        # Match the real Flow wrapper: the block attention sees nine mixed rows,
        # while the surrounding model payload still carries the seven-row native layout.
        BlockPatch(0, cfg)(
            {"img": torch.zeros(9, 128), "layout": _layout(7), "transformer_options": opts},
            {"original_block": mixed_block},
        )

        q7 = torch.zeros(1, 1, 7, 128, dtype=torch.bfloat16)

        def native_block(args):
            to = args["transformer_options"]
            out = to["optimized_attention_override"](
                lambda q, *a, **kw: q.transpose(1, 2).reshape(1, q.shape[2], -1),
                q7,
                q7,
                q7,
                1,
                skip_reshape=True,
                transformer_options=to,
            )
            assert out.shape == (1, 7, 128)
            return {"img": args["img"]}

        BlockPatch(0, cfg)(
            {"img": torch.zeros(7, 128), "layout": _layout(7), "transformer_options": {}},
            {"original_block": native_block},
        )
    finally:
        _FORWARD.reset(token)

    assert calls == [(9, 9, 3), (7, 7, 3)]
    assert state.sparse_calls == 2
    assert state.external_mixed_sol_calls == 1
    assert state.external_mixed_q_rows == state.external_mixed_kernel_q_rows == 9
    assert state.fallbacks["external_sequence_native"] == 0
    assert state.fallbacks["external_sequence_layout"] == 0


def test_legacy_external_contract_stays_native(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None))])
    token = _FORWARD.set((model, state, 0, set(), []))
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    dense = []
    try:
        q = torch.zeros(1, 1, 9, 128, dtype=torch.bfloat16)

        def block(args):
            to = args["transformer_options"]
            to["optimized_attention_override"](
                lambda q, *a, **kw: dense.append(True) or q.transpose(1, 2).reshape(1, 9, 128),
                q,
                q,
                q,
                1,
                skip_reshape=True,
                transformer_options=to,
            )
            return {"img": args["img"]}

        BlockPatch(0, cfg)(
            {
                "img": torch.zeros(9, 128),
                "layout": _layout(9),
                "transformer_options": {"vdn_h3_external_sequence_v1": {"api": 1}},
            },
            {"original_block": block},
        )
    finally:
        _FORWARD.reset(token)
    assert dense == [True]
    assert state.sparse_calls == 0
    assert state.fallbacks["external_sequence_native"] == 1


def _measure_contract(**changes):
    value = {
        "api": 1,
        "mode": "prefix_kv_stratified_subsample",
        "video_start": 3,
        "sequence_rows": 9,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid_h": 1,
        "source_grid_w": 2,
        "prefix_grid_h": 2,
        "prefix_grid_w": 2,
        "source_rows_per_frame": 2,
        "prefix_rows_per_frame": 4,
        "expected_kv_rows": 7,
        "exact_prefix_queries_preserved": True,
        "suffix_kv_unchanged": True,
    }
    value.update(changes)
    return value


def test_external_mixed_measure_keeps_all_q_and_reduces_only_kv(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None))])
    token = _FORWARD.set((model, state, 0, set(), []))
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    calls = []

    def fake_attention(q, k, v, prefix, config, request, **kwargs):
        calls.append((q.shape[2], k.shape[2], v.shape[2], prefix))
        request.sparse_calls += 1
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)

    monkeypatch.setattr(sparse, "attention", fake_attention)
    try:
        q = torch.arange(9 * 128, dtype=torch.bfloat16).reshape(1, 1, 9, 128)
        opts = {
            "vdn_h3_external_sequence_v1": _contract(),
            "h3_flow_mixed_grid_attention_measure_v1": _measure_contract(),
        }

        def block(args):
            to = args["transformer_options"]
            out = to["optimized_attention_override"](
                lambda q, *a, **kw: q.transpose(1, 2).reshape(1, q.shape[2], -1),
                q,
                q,
                q,
                1,
                skip_reshape=True,
                transformer_options=to,
            )
            assert out.shape == (1, 9, 128)
            return {"img": args["img"]}

        BlockPatch(0, cfg)(
            {"img": torch.zeros(9, 128), "layout": _layout(7), "transformer_options": opts},
            {"original_block": block},
        )
    finally:
        _FORWARD.reset(token)

    assert calls == [(9, 7, 7, 3)]
    assert state.external_mixed_sol_calls == 1
    assert state.external_mixed_q_rows == state.external_mixed_kernel_q_rows == 9
    assert state.external_mixed_measure_calls == 1
    assert state.external_mixed_measure_q_rows == 9
    assert state.external_mixed_measure_kv_rows_before == 9
    assert state.external_mixed_measure_kv_rows_after == 7
    assert state.external_mixed_measure_removed_rows == 2


def test_external_mixed_measure_malformed_contract_fails_hard(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None))])
    token = _FORWARD.set((model, state, 0, set(), []))
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    try:
        q = torch.zeros(1, 1, 9, 128, dtype=torch.bfloat16)
        opts = {
            "vdn_h3_external_sequence_v1": _contract(),
            "h3_flow_mixed_grid_attention_measure_v1": _measure_contract(expected_kv_rows=8),
        }

        def block(args):
            to = args["transformer_options"]
            to["optimized_attention_override"](
                lambda q, *a, **kw: q.transpose(1, 2).reshape(1, q.shape[2], -1),
                q,
                q,
                q,
                1,
                skip_reshape=True,
                transformer_options=to,
            )
            return {"img": args["img"]}

        import pytest

        with pytest.raises(RuntimeError, match="row accounting"):
            BlockPatch(0, cfg)(
                {"img": torch.zeros(9, 128), "layout": _layout(7), "transformer_options": opts},
                {"original_block": block},
            )
    finally:
        _FORWARD.reset(token)
