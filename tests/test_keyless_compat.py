from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sol_h3.contracts import Config
from sol_h3.interop import HistoryPolicy, RECEIPTS_KEY
from sol_h3 import keyless_compat
from sol_h3.runtime import BlockPatch, Request, _REQUEST
from sol_h3 import runtime, sparse


class Contract:
    api = 1
    architecture = "h3_keyless_core50_v1"
    core_blocks = 50
    token_refiner = "native_qkv"
    token_refiner_blocks = 2
    heads = 56
    head_dim = 128
    inner_dim = 7168
    hidden_size = 5376
    routing_source = "value"
    retrieval_source = "raw_projected_value"
    routing_norm = "rmsnorm"
    routing_norm_epsilon = 1e-5
    rope_policy = "h3_split_half_96_v1"
    qv_order = "q_effective;v"
    projection_attr = "qv_proj"
    checkpoint_format_version = 1
    provenance_identity = "keyless-test-artifact"

    def identity(self):
        return (
            self.api,
            self.architecture,
            self.checkpoint_format_version,
            self.qv_order,
            self.heads,
            self.head_dim,
            self.inner_dim,
            self.routing_source,
            self.retrieval_source,
            self.rope_policy,
            self.provenance_identity,
        )


def _attention(*, fake_qkv=False):
    attention = SimpleNamespace(
        qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
        q_norm=object(),
        route_norm=object(),
        forward=lambda *args, **kwargs: None,
    )
    if fake_qkv:
        attention.qkv_proj = object()
    return attention


def _keyless_model(*, fake_qkv=False):
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=_attention(fake_qkv=fake_qkv)) for _ in range(50)],
        token_refiner=SimpleNamespace(blocks=[object(), object()]),
        dtype=torch.bfloat16,
    )
    setattr(model, keyless_compat.KEYLESS_CONTRACT_KEY, Contract())
    return model


def _layout(seq_len=7, prefix=3):
    return SimpleNamespace(
        seq_len=seq_len,
        segments=[(0, prefix, "text"), (prefix, seq_len, "video")],
        signature=(prefix, seq_len),
    )


def _history_options(cfg, semantic):
    return {
        keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic,
        "patches_replace": {
            "dit": {
                ("double_block", index): BlockPatch(index, cfg)
                for index in range(50)
            }
        },
    }


def test_contract_accepts_qv_and_rejects_fake_qkv():
    model = _keyless_model()
    semantic = keyless_compat.keyless_semantic_identity(model)
    assert semantic[0] == keyless_compat.KEYLESS_CONTRACT_KEY
    assert semantic[-1] == "keyless-test-artifact"

    with pytest.raises(keyless_compat.KeylessCompatibilityError, match="qkv_proj"):
        keyless_compat.validate_keyless_contract(_keyless_model(fake_qkv=True))


def test_keyless_history_identity_is_architecture_bound_and_requires_runtime_marker():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    model = _keyless_model()
    semantic = keyless_compat.keyless_semantic_identity(model)
    options = _history_options(cfg, semantic)
    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        policy = HistoryPolicy(cfg)
        identity = policy(layout=_layout(), options=options, model=model)
        missing_marker = policy(
            layout=_layout(),
            options={k: v for k, v in options.items() if k != keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY},
            model=model,
        )
    finally:
        _REQUEST.reset(token)

    assert identity is not None
    assert identity[-1] == (
        "keyless_materialized_route",
        semantic,
        keyless_compat.KEYLESS_RECEIPT_TAG,
    )
    assert missing_marker is None


def test_foreign_keyless_provider_makes_sol_history_opaque():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    model = _keyless_model()
    semantic = keyless_compat.keyless_semantic_identity(model)
    options = _history_options(cfg, semantic)

    class ForeignProvider:
        api = 1

        def __call__(self, **_kwargs):
            return None

    options[keyless_compat.KEYLESS_PROVIDER_KEY] = ForeignProvider()
    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        assert HistoryPolicy(cfg)(layout=_layout(), options=options, model=model) is None
    finally:
        _REQUEST.reset(token)


def test_keyless_materialized_receipts_are_tagged_and_accepted():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    semantic = keyless_compat.keyless_semantic_identity(_keyless_model())
    sink = []
    options = {
        keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic,
        RECEIPTS_KEY: sink,
    }

    runtime.receipt(options, 0, "sol")
    runtime.receipt(options, 1, "dense_warmup")

    assert sink == [
        ("sol_h3", 0, "sol", (keyless_compat.KEYLESS_RECEIPT_TAG, semantic)),
        ("sol_h3", 1, "dense_warmup", (keyless_compat.KEYLESS_RECEIPT_TAG, semantic)),
    ]
    assert HistoryPolicy(cfg).accept_receipts(sink) is True


def test_unreviewed_keyless_routes_remain_actual_only():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    semantic = keyless_compat.keyless_semantic_identity(_keyless_model())
    sink = []
    options = {
        keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic,
        RECEIPTS_KEY: sink,
    }

    runtime.receipt(options, 0, "sol_external_mixed")
    assert sink[0][2] == "keyless_unvalidated:sol_external_mixed"
    assert HistoryPolicy(cfg).accept_receipts(sink) is False


def test_materialized_route_dense_reference_retrieves_raw_v_not_route():
    torch.manual_seed(7)
    q = torch.randn(1, 2, 5, 8)
    route = torch.randn(1, 2, 5, 8) * 3.0
    raw_v = torch.randn(1, 2, 5, 8) + 10.0

    got = sparse._dense_reference(q, route, raw_v, None)
    want = F.scaled_dot_product_attention(q, route, raw_v).transpose(1, 2)
    wrong_retrieval = F.scaled_dot_product_attention(q, route, route).transpose(1, 2)

    assert torch.allclose(got, want)
    assert not torch.allclose(got, wrong_retrieval)


def test_attention_preprocess_chain_treats_logical_k_as_route_and_preserves_raw_v():
    q = torch.ones(1, 2, 4, 8)
    route = torch.ones_like(q)
    raw_v = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)

    def preprocess(q_in, route_in, v_in, _heads, **_kwargs):
        return q_in, route_in * 2.0, v_in

    def provider(*_args, **_kwargs):
        raise AssertionError("leaf provider is not executed by preprocessing")

    provider.attention_preprocess_v1 = (preprocess, None)
    q_out, route_out, v_out, leaf = runtime._preprocess_chain(
        provider,
        q,
        route,
        raw_v,
        2,
        {},
    )
    assert leaf is None
    assert torch.equal(q_out, q)
    assert torch.equal(route_out, route * 2.0)
    assert torch.equal(v_out, raw_v)


def test_runtime_install_marks_only_returned_clone_and_rejects_keyless_vdn(monkeypatch):
    model_inner = _keyless_model()
    source = SimpleNamespace(
        get_model_object=lambda name: model_inner,
        model_options={"transformer_options": {}},
    )
    clean_clone = SimpleNamespace(
        model_options={"transformer_options": {}},
        object_patches={},
    )
    monkeypatch.setattr(keyless_compat, "_ORIGINAL_RUNTIME_INSTALL", lambda model, config: clean_clone)

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    installed = keyless_compat._runtime_install(source, cfg)
    semantic = keyless_compat.keyless_semantic_identity(model_inner)
    assert keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY not in source.model_options["transformer_options"]
    assert installed.model_options["transformer_options"][keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY] == semantic

    def vdn_forward(*args, **kwargs):
        return None

    vdn_forward._vdn_forward = True
    blocked_clone = SimpleNamespace(
        model_options={"transformer_options": {}},
        object_patches={"diffusion_model.blocks.4.attn.forward": vdn_forward},
    )
    monkeypatch.setattr(keyless_compat, "_ORIGINAL_RUNTIME_INSTALL", lambda model, config: blocked_clone)
    with pytest.raises(RuntimeError, match="does not yet authorize VDN"):
        keyless_compat._runtime_install(source, cfg)


def test_diffusion_wrapper_installs_one_request_stable_owned_provider(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    semantic = keyless_compat.keyless_semantic_identity(_keyless_model())
    captured = []

    def original(_self, _executor, _x, _timestep, _context, options, **_kwargs):
        captured.append(options)
        return options

    monkeypatch.setattr(keyless_compat, "_ORIGINAL_DIFFUSION_CALL", original)
    wrapper = SimpleNamespace(config=cfg)
    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        base = {keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic}
        keyless_compat._diffusion_call(wrapper, object(), None, None, None, base)
        keyless_compat._diffusion_call(wrapper, object(), None, None, None, base)
    finally:
        _REQUEST.reset(token)

    first = captured[0][keyless_compat.KEYLESS_PROVIDER_KEY]
    second = captured[1][keyless_compat.KEYLESS_PROVIDER_KEY]
    assert first is second
    assert first.api == 1
    assert first.identity == (keyless_compat.KEYLESS_PROVIDER_IDENTITY, 1, semantic)
    assert keyless_compat.KEYLESS_PROVIDER_KEY not in base


def test_diffusion_wrapper_preserves_foreign_keyless_provider(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    semantic = keyless_compat.keyless_semantic_identity(_keyless_model())
    foreign = object()
    captured = []

    def original(_self, _executor, _x, _timestep, _context, options, **_kwargs):
        captured.append(options)
        return options

    monkeypatch.setattr(keyless_compat, "_ORIGINAL_DIFFUSION_CALL", original)
    wrapper = SimpleNamespace(config=cfg)
    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        keyless_compat._diffusion_call(
            wrapper,
            object(),
            None,
            None,
            None,
            {
                keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic,
                keyless_compat.KEYLESS_PROVIDER_KEY: foreign,
            },
        )
    finally:
        _REQUEST.reset(token)
    assert captured[0][keyless_compat.KEYLESS_PROVIDER_KEY] is foreign


def test_owned_provider_scores_materialized_route_and_retrieves_raw_v(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    semantic = keyless_compat.keyless_semantic_identity(_keyless_model())
    provider = keyless_compat._KeylessSolProviderV1(semantic)
    captured = {}

    class Routing:
        block_index = 4
        value_domain = None
        routing_position_domain = None

        @staticmethod
        def materialize(v):
            return v * 2.0

    def fake_sparse(q, route, v, prefix, config, state, **kwargs):
        captured.update(q=q.clone(), route=route.clone(), v=v.clone(), prefix=prefix)
        return torch.zeros(1, q.shape[2], q.shape[1] * q.shape[3], dtype=q.dtype)

    monkeypatch.setattr(sparse, "attention", fake_sparse)
    monkeypatch.setattr(keyless_compat, "_sol_override_previous", lambda _options: None)

    q = torch.randn(4, 2, 128)
    raw_v = torch.randn_like(q)
    options = {
        keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic,
        "minimax_h3_layout": _layout(seq_len=4, prefix=2),
        RECEIPTS_KEY: [],
    }
    state = Request(cfg)
    token = runtime._FORWARD.set((object(), state, 0, {4}, []))
    try:
        out = provider(
            q=q,
            v=raw_v,
            heads=2,
            scale=128**-0.5,
            routing=Routing(),
            mask=None,
            log_measure=None,
            exact_blocks=None,
            query_domain=None,
            value_domain=None,
            dense_fallback=lambda: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
            transformer_options=options,
        )
        routes = runtime._FORWARD.get()[4]
    finally:
        runtime._FORWARD.reset(token)

    assert out.shape == q.shape
    torch.testing.assert_close(captured["route"].squeeze(0).transpose(0, 1), raw_v * 2.0)
    torch.testing.assert_close(captured["v"].squeeze(0).transpose(0, 1), raw_v)
    assert captured["prefix"] == 2
    assert routes == [(4, "sol")]
    assert state.eligible_calls == 1
    assert options[RECEIPTS_KEY][0][2] == "sol"


def test_owned_provider_defers_inherited_attention_composition(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    semantic = keyless_compat.keyless_semantic_identity(_keyless_model())
    provider = keyless_compat._KeylessSolProviderV1(semantic)
    sentinel = torch.randn(3, 2, 128)
    previous = object()
    monkeypatch.setattr(keyless_compat, "_sol_override_previous", lambda _options: previous)
    monkeypatch.setattr(
        sparse,
        "attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sparse path must not run")),
    )

    class Routing:
        block_index = 1
        value_domain = None
        routing_position_domain = None

        @staticmethod
        def materialize(_v):
            raise AssertionError("route must be owned by the existing materialized bridge")

    state = Request(cfg)
    token = runtime._FORWARD.set((object(), state, 0, {1}, []))
    try:
        out = provider(
            q=torch.randn(3, 2, 128),
            v=torch.randn(3, 2, 128),
            heads=2,
            scale=128**-0.5,
            routing=Routing(),
            mask=None,
            log_measure=None,
            exact_blocks=None,
            query_domain=None,
            value_domain=None,
            dense_fallback=lambda: sentinel,
            transformer_options={keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic},
        )
    finally:
        runtime._FORWARD.reset(token)
    assert out is sentinel


def test_owned_provider_refuses_unimplemented_exact_blocks():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    semantic = keyless_compat.keyless_semantic_identity(_keyless_model())
    provider = keyless_compat._KeylessSolProviderV1(semantic)

    class Routing:
        block_index = 2
        value_domain = None
        routing_position_domain = None

    state = Request(cfg)
    token = runtime._FORWARD.set((object(), state, 0, {2}, []))
    try:
        with pytest.raises(RuntimeError, match="exact_blocks"):
            provider(
                q=torch.randn(3, 2, 128),
                v=torch.randn(3, 2, 128),
                heads=2,
                scale=128**-0.5,
                routing=Routing(),
                mask=None,
                log_measure=None,
                exact_blocks=(0, 1),
                query_domain=None,
                value_domain=None,
                dense_fallback=lambda: None,
                transformer_options={keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: semantic},
            )
    finally:
        runtime._FORWARD.reset(token)
