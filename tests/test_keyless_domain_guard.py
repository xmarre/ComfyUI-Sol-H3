from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sol_h3 import keyless_compat, keyless_domain_guard


class _Contract:
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

    def identity(self):
        return (self.api, self.architecture, "domain-guard-test")


def _semantic():
    attention = SimpleNamespace(
        qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
        q_norm=object(),
        route_norm=object(),
    )
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=attention) for _ in range(50)],
        token_refiner=SimpleNamespace(blocks=[object(), object()]),
    )
    setattr(model, keyless_compat.KEYLESS_CONTRACT_KEY, _Contract())
    return keyless_compat.keyless_semantic_identity(model)


def _call(provider, routing, **overrides):
    kwargs = {
        "q": torch.randn(3, 2, 128),
        "v": torch.randn(3, 2, 128),
        "heads": 2,
        "scale": 128**-0.5,
        "routing": routing,
        "mask": None,
        "log_measure": None,
        "exact_blocks": None,
        "query_domain": None,
        "value_domain": None,
        "dense_fallback": lambda: (_ for _ in ()).throw(
            AssertionError("row domains must fail before dense fallback")
        ),
        "transformer_options": {
            keyless_compat.KEYLESS_RUNTIME_IDENTITY_KEY: provider.semantic,
        },
    }
    kwargs.update(overrides)
    return provider(**kwargs)


@pytest.mark.parametrize(
    ("provider_field", "routing_field", "message"),
    [
        ("query_domain", None, "query_domain"),
        ("value_domain", None, "value_domain"),
        (None, "value_domain", "routing.value_domain"),
        (None, "routing_position_domain", "routing.routing_position_domain"),
    ],
)
def test_explicit_keyless_row_domains_fail_before_dense_fallback(
    provider_field, routing_field, message
):
    provider = keyless_compat._KeylessSolProviderV1(_semantic())
    routing = SimpleNamespace(
        block_index=0,
        value_domain=object() if routing_field == "value_domain" else None,
        routing_position_domain=(
            object() if routing_field == "routing_position_domain" else None
        ),
    )
    overrides = {provider_field: object()} if provider_field is not None else {}

    with pytest.raises(RuntimeError, match=message.replace(".", r"\.")):
        _call(provider, routing, **overrides)


def test_domain_guard_delegates_ordinary_provider_call(monkeypatch):
    provider = keyless_compat._KeylessSolProviderV1(_semantic())
    routing = SimpleNamespace(
        block_index=0,
        value_domain=None,
        routing_position_domain=None,
    )
    sentinel = object()
    captured = {}

    def original(self, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(keyless_domain_guard, "_ORIGINAL_PROVIDER_CALL", original)
    got = _call(provider, routing)

    assert got is sentinel
    assert captured["routing"] is routing
    assert captured["query_domain"] is None
    assert captured["value_domain"] is None
