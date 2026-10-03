"""Continuation dense attention must use the same dispatcher as native VDN."""

import sys
from types import ModuleType

import pytest
import torch

from sol_h3.partitioned_request import _weighted_dense
from sol_h3.contracts import Config
from sol_h3.runtime import Request, _REQUEST


def _inputs(dtype):
    generator = torch.Generator().manual_seed(840)
    # Include non-contiguous THD inputs, as supplied by projection views.
    values = [torch.randn((rows, 2, 24), generator=generator, dtype=dtype)[..., ::3]
              for rows in (5, 11, 11)]
    return values


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
def test_partitioned_dense_delegates_union_mask_scale_and_layout_to_core(monkeypatch, dtype, biased):
    q, k, v = _inputs(dtype)
    bias = torch.linspace(-0.7, 0, k.shape[0]) if biased else None
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    if bias is not None:
        scores += bias.to(dtype).double()
    want = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.double()).to(dtype)
    calls = []

    def core_sdpa(qh, kh, vh, *, attn_mask, scale):
        calls.append((qh, kh, vh, attn_mask, scale))
        return torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask, scale=scale)

    ops = ModuleType("comfy.ops")
    ops.scaled_dot_product_attention = core_sdpa
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    got = _weighted_dense(q, k, v, bias, scale=scale)

    assert len(calls) == 1  # The previous source bypassed Core completely.
    qh, kh, vh, mask, received_scale = calls[0]
    for received, source in zip((qh, kh, vh), (q, k, v), strict=True):
        assert received.data_ptr() == source.data_ptr()
        assert received.shape == (1, source.shape[1], source.shape[0], source.shape[2])
    assert received_scale == scale
    if bias is None:
        assert mask is None
    else:
        torch.testing.assert_close(mask, bias.to(dtype).view(1, 1, 1, -1), rtol=0, atol=0)
    assert got.shape == q.shape and got.dtype == dtype and got.is_contiguous()
    tolerance = 0.02 if dtype == torch.bfloat16 else (1e-6 if dtype == torch.float32 else 1e-12)
    torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)


def test_standalone_dense_oracle_works_without_comfy(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.ops", None)
    q, k, v = _inputs(torch.float64)
    got = _weighted_dense(q, k, v, None, scale=0.25)
    want = torch.einsum(
        "hqk,khd->qhd", (torch.einsum("qhd,khd->hqk", q, k) * 0.25).softmax(-1), v)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_core_execution_failure_is_not_retried_on_raw_sdpa(monkeypatch):
    ops = ModuleType("comfy.ops")

    def fail(*_args, **_kwargs):
        raise torch.OutOfMemoryError("sentinel dense dispatcher failure")

    ops.scaled_dot_product_attention = fail
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    q, k, v = _inputs(torch.float32)
    with pytest.raises(torch.OutOfMemoryError, match="sentinel"):
        _weighted_dense(q, k, v, None, scale=0.25)


@pytest.mark.parametrize("core_available", [False, True])
def test_completed_dispatch_counter_is_request_local(monkeypatch, core_available):
    ops = ModuleType("comfy.ops")
    ops.scaled_dot_product_attention = lambda *a, **k: torch.nn.functional.scaled_dot_product_attention(*a, **k)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops if core_available else None)
    q, k, v = _inputs(torch.float32)
    first, second = Request(Config()), Request(Config())
    for state in (first, second):
        token = _REQUEST.set(state)
        try:
            _weighted_dense(q, k, v, None, scale=0.25)
        finally:
            _REQUEST.reset(token)
    counter = "partitioned_core_dense_calls" if core_available else "partitioned_torch_dense_calls"
    other = "partitioned_torch_dense_calls" if core_available else "partitioned_core_dense_calls"
    assert getattr(first, counter) == getattr(second, counter) == 1
    assert getattr(first, other, 0) == getattr(second, other, 0) == 0
