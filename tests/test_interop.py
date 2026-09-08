"""CPU routing tests; mocked kernel dispatch is not GPU kernel validation."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sol_h3 import runtime, sparse
from sol_h3.contracts import Config, KEY
from sol_h3.interop import HistoryPolicy, RECEIPTS_KEY, VDN_KEY
from sol_h3.runtime import BlockPatch, DiffusionWrapper, Request, SamplingWrapper, _FORWARD, _REQUEST


def layout(rows=7):
    return SimpleNamespace(seq_len=rows, segments=[(0, 3, "text"), (3, rows, "video")], signature=(3, rows))


@pytest.fixture
def scope(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None))])
    token = _FORWARD.set((model, state, 0, set(), []))
    # Eligibility on CPU is otherwise deliberately native-only.
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    yield cfg, state
    _FORWARD.reset(token)


@pytest.mark.parametrize("name", ["sage", "sage3", "arbitrary_provider"])
def test_generic_dense_chain_and_independent_arithmetic_gate(scope, monkeypatch, name):
    cfg, state = scope
    q = torch.randn(1, 2, 7, 128, dtype=torch.bfloat16)
    dense_calls = []
    def provider(original, q, k, v, heads, **kw):
        dense_calls.append((q.shape[2], kw["_inside_attn_wrapper"]))
        out = F.scaled_dot_product_attention(q, k, v) + .25
        return out if kw.get("skip_output_reshape") else out.transpose(1, 2).reshape(1, q.shape[2], -1)
    provider.__name__ = name
    def kernel(q, k, v, **kw):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    opts = {"optimized_attention_override": provider, RECEIPTS_KEY: []}
    def block(args):
        to = args["transformer_options"]
        got = to["optimized_attention_override"](lambda *a, **kw: None, q, q, q, 2,
             skip_reshape=True, transformer_options=to)
        assert got.shape == (1, 7, 256)
        assert torch.allclose(got[:, :3], (F.scaled_dot_product_attention(q[:, :, :3], q, q) + .25).transpose(1, 2).reshape(1, 3, 256))
        return {"img": args["img"]}
    BlockPatch(0, cfg)({"img": torch.zeros(7, 256), "layout": layout(), "transformer_options": opts},
                       {"original_block": block})
    assert dense_calls == [(3, True)]  # verification never used approximate dense provider
    assert state.sparse_calls == 1 and state.gates
    assert opts[RECEIPTS_KEY] == [("sol_h3", 0, "sol")]


def test_mixed_grid_fallback_then_native_grid_resumes(scope, monkeypatch):
    cfg, state = scope
    calls = []
    def kernel(q, k, v, prefix, config, state, **kw):
        state.sparse_calls += 1
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)
    monkeypatch.setattr(sparse, "attention", kernel)
    q = torch.zeros(1, 1, 7, 128, dtype=torch.bfloat16)
    for api in (1, 2, None):
        opts = {} if api is None else {"vdn_h3_external_sequence_v1": {"api": api}}
        def block(args):
            to = args["transformer_options"]
            def dense(q, *a, **kw):
                calls.append(api)
                return q.transpose(1, 2).reshape(1, 7, -1)
            to["optimized_attention_override"](dense, q, q, q, 1, skip_reshape=True, transformer_options=to)
            return {"img": args["img"]}
        BlockPatch(0, cfg)({"img": torch.zeros(7, 128), "layout": layout(), "transformer_options": opts},
                           {"original_block": block})
    assert calls == [1, 2]
    assert state.sparse_calls == 1
    assert state.fallbacks["external_sequence_native"] == 2


def test_vdn_multicall_restricted_domain_and_native_fallbacks(scope, monkeypatch):
    cfg, state = scope
    sizes = []
    def kernel(q, k, v, prefix, config, state, **kw):
        sizes.append((q.shape[2], k.shape[2], prefix))
        state.sparse_calls += 1
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)
    monkeypatch.setattr(sparse, "attention", kernel)
    q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
    native_calls = []
    def block(args):
        provider = args["transformer_options"][VDN_KEY]
        def native():
            native_calls.append(True)
            return q
        for kind, aligned in [("global", False), ("local", False), ("anchor", False),
                              ("flex_masked", False), ("local", True), ("local", True)]:
            assert provider(native, q, q, q, kind=kind, scale=128 ** -.5, square_aligned=aligned).shape == q.shape
        return {"img": args["img"]}
    BlockPatch(0, cfg)({"img": torch.zeros(7, 128), "layout": layout(), "transformer_options": {}},
                       {"original_block": block})
    assert len(native_calls) == 4
    assert sizes == [(4, 4, 0)] * 2
    assert state.vdn_local_sol_calls == 2 and state.sparse_calls == 2


def test_repeated_scopes_actual_only_warmup(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=1, dense_layers=0)
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    def kernel(q, k, v, prefix, config, state, **kw):
        state.sparse_calls += 1
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)
    monkeypatch.setattr(sparse, "attention", kernel)
    q = torch.zeros(1, 1, 7, 128)
    class Executor:
        class_obj = SimpleNamespace(blocks=[object()])
        def __call__(self, x, timestep, context, options, **kw):
            if timestep == "forecast":
                return x
            def block(args):
                to = args["transformer_options"]
                to["optimized_attention_override"](lambda q, *a, **kw: q, q, q, q, 1,
                                                   skip_reshape=True, transformer_options=to)
                return {"img": args["img"]}
            return BlockPatch(0, cfg)({"img": torch.zeros(7, 128), "layout": layout(),
                                      "transformer_options": options}, {"original_block": block})
    stats = []
    def sample():
        for step in ("forecast", 900, "forecast", 800, "forecast", 700):
            DiffusionWrapper(cfg)(Executor(), [q], step, None, {KEY: cfg.metadata()})
        state = _REQUEST.get()
        stats.append((state.evaluations, state.dense_calls, state.sparse_calls))
    for _ in range(3):
        SamplingWrapper(cfg)(sample)
    assert stats == [(3, 1, 2)] * 3


def test_preprocessing_is_once_before_sparse_and_dense_prefix(scope, monkeypatch):
    cfg, state = scope
    q = torch.randn(1, 1, 7, 128, dtype=torch.bfloat16)
    transforms = []
    dense_inputs = []
    def dense_leaf(original, q, k, v, heads, **kw):
        dense_inputs.append(k.clone())
        return F.scaled_dot_product_attention(q, k, v)
    def transform(q, k, v, heads, **kw):
        transforms.append(True)
        return q, k * 2, v
    def provider(original, q, k, v, heads, **kw):
        return dense_leaf(original, *transform(q, k, v, heads, **kw), heads, **kw)
    provider.attention_preprocess_v1 = (transform, dense_leaf)
    def kernel(qb, kb, vb, **kw):
        return F.scaled_dot_product_attention(qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2)).transpose(1, 2)
    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    def block(args):
        to = args["transformer_options"]
        to["optimized_attention_override"](lambda *a, **kw: None, q, q, q, 1,
                                           skip_reshape=True, transformer_options=to)
        return {"img": args["img"]}
    BlockPatch(0, cfg)({"img": torch.zeros(7, 128), "layout": layout(),
        "transformer_options": {"optimized_attention_override": provider}}, {"original_block": block})
    assert transforms == [True]
    assert len(dense_inputs) == 1 and torch.equal(dense_inputs[0], q * 2)


def _stack_vdn_forward(state, cfg, base_branch):
    def forward(*args, **kwargs):
        # These names are intentionally captured: the compatibility adapter audits
        # the same closure shape produced by VDN's make_vdn_forward.
        return state, cfg, base_branch
    forward._vdn_forward = True
    return forward


def test_grouped_vdn_closure_has_stable_forecast_identity():
    sol_cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    vdn_cfg = {"radius": 1, "chunk": 5, "anchor_frames": "both", "linear_enabled": True}
    state = SimpleNamespace(cfg=vdn_cfg, softmax_backend="grouped")
    branch = SimpleNamespace(enable_text_state=True)
    forward = _stack_vdn_forward(state, vdn_cfg, branch)
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(forward=forward))],
        dtype=torch.bfloat16,
    )
    options = {"patches_replace": {"dit": {("double_block", 0): BlockPatch(0, sol_cfg)}}}
    request = Request(sol_cfg)
    token = _REQUEST.set(request)
    try:
        policy = HistoryPolicy(sol_cfg)
        first = policy(layout=layout(), options=options, model=model)
        second = policy(layout=layout(), options=options, model=model)
        assert first is not None and first == second
        vdn_cfg["radius"] = 2
        changed = policy(layout=layout(), options=options, model=model)
        assert changed is not None and changed != first
    finally:
        _REQUEST.reset(token)


def test_flex_and_unknown_vdn_history_stay_actual_only():
    sol_cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    vdn_cfg = {"radius": 1, "chunk": 5, "anchor_frames": "both"}
    state = SimpleNamespace(cfg=vdn_cfg, softmax_backend="flex")
    forward = _stack_vdn_forward(state, vdn_cfg, SimpleNamespace(enable_text_state=False))
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(forward=forward))],
        dtype=torch.bfloat16,
    )
    options = {"patches_replace": {"dit": {("double_block", 0): BlockPatch(0, sol_cfg)}}}
    token = _REQUEST.set(Request(sol_cfg))
    try:
        policy = HistoryPolicy(sol_cfg)
        assert policy(layout=layout(), options=options, model=model) is None

        def opaque_forward(*args, **kwargs):
            return None
        opaque_forward._vdn_forward = True
        model.blocks[0].attn.forward = opaque_forward
        assert policy(layout=layout(), options=options, model=model) is None
    finally:
        _REQUEST.reset(token)
