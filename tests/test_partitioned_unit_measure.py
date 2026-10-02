"""Unit key measure must retain its contract without an SDPA attention mask."""
import math
from types import SimpleNamespace

import pytest
import torch

from sol_h3 import partitioned_request as request
from sol_h3.contracts import Config
from sol_h3.interop import RECEIPTS_KEY
from sol_h3.runtime import Request, _FORWARD, _REQUEST


@pytest.mark.parametrize("kind", ["global", "local", "anchor"])
def test_unit_measure_dense_request_uses_unmasked_sdpa_and_preserves_receipt(monkeypatch, kind):
    generator = torch.Generator().manual_seed(62)
    q = torch.randn((4, 2, 128), generator=generator, dtype=torch.float64)
    k = torch.randn((11, 2, 128), generator=generator, dtype=torch.float64)
    v = torch.randn(k.shape, generator=generator, dtype=torch.float64)
    scale = 128 ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    state = Request(Config(backend="sol"))
    options = {RECEIPTS_KEY: []}
    masks = []
    sdpa = request.F.scaled_dot_product_attention

    def observe_sdpa(*args, **kwargs):
        masks.append(kwargs["attn_mask"])
        return sdpa(*args, **kwargs)

    # Exercise request ownership, contract validation, dense execution and
    # completion on CPU; SM120 eligibility itself is a separate GPU contract.
    monkeypatch.setattr(request, "_validate_thd", lambda *_args: None)
    monkeypatch.setattr(request.F, "scaled_dot_product_attention", observe_sdpa)
    monkeypatch.setattr(request, "_bias_cache", lambda *_args: pytest.fail("Unit measure allocated a bias cache"))
    request_token = _REQUEST.set(state)
    forward_token = _FORWARD.set((None, state, 0, None, []))
    try:
        got = request.partitioned_request_attention(
            q, k, v, transformer_options=options, block_index=0, kind=kind,
            scale=scale, sink_rows=3, prefix_k_range=(3, 7),
            prefix_log_key_measure=0.0, semantic_digest="a" * 64, force_dense=True,
        )
    finally:
        _FORWARD.reset(forward_token)
        _REQUEST.reset(request_token)

    assert masks == [None]
    assert state.partitioned_unit_measure_calls == 1
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
    _provider, block, route, fields = options[RECEIPTS_KEY][0]
    assert route == request.PARTITIONED_DENSE_ROUTE
    assert fields[8:10] == ((3, 7), 0.0)
    assert (block, fields) in state.partitioned_validated_receipts


@pytest.mark.parametrize("prefix_range,log_measure,digest,reason", [
    ((True, 7), 0.0, "a" * 64, "integer pair"),
    ([3, 7], 0.0, "a" * 64, "integer pair"),
    ((7, 7), 0.0, "a" * 64, "outside"),
    ((3, 12), 0.0, "a" * 64, "outside"),
    ((3, 7), 0.0, "invalid", "digest"),
    ((3, 7), math.nan, "a" * 64, "finite"),
    ((3, 7), 0.1, "a" * 64, "non-positive"),
])
def test_unit_measure_shortcut_does_not_bypass_contract_validation(prefix_range, log_measure, digest, reason):
    with pytest.raises(RuntimeError, match=reason):
        request._key_bias(
            SimpleNamespace(), kv_rows=11, prefix_range=prefix_range,
            log_measure=log_measure, device=torch.device("cpu"), semantic_digest=digest,
        )


@pytest.mark.parametrize("measure", [-0.0, 0.0])
def test_exact_zero_is_an_identity_without_cuda_allocation(measure):
    state = SimpleNamespace()
    assert request._key_bias(
        state, kv_rows=11, prefix_range=(3, 7), log_measure=measure,
        device=torch.device("cpu"), semantic_digest="a" * 64,
    ) is None
    assert state.partitioned_unit_measure_calls == 1


def test_nonzero_measure_keeps_the_bias_path(monkeypatch):
    class BiasPathReached(Exception):
        pass

    def bias_cache(_state):
        raise BiasPathReached

    monkeypatch.setattr(request, "_bias_cache", bias_cache)
    with pytest.raises(BiasPathReached):
        request._key_bias(
            SimpleNamespace(), kv_rows=11, prefix_range=(3, 7), log_measure=-1e-15,
            device=torch.device("cpu"), semantic_digest="a" * 64,
        )
