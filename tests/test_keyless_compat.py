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


def _layout():
    return SimpleNamespace(
        seq_len=7,
        segments=[(0, 3, "text"), (3, 7, "video")],
        signature=(3, 7),
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
