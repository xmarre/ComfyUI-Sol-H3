"""Native arithmetic checks keep their decisions with one host statistics read."""

import math
import sys
from types import ModuleType

import pytest
import torch

from sol_h3 import sparse
from sol_h3.contracts import Config
from sol_h3.runtime import Request, _REQUEST


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
def test_native_reference_uses_core_with_original_views_and_bias(monkeypatch, dtype, biased):
    generator = torch.Generator().manual_seed(866)
    q, k, v = [torch.randn(1, 2, rows, 24, generator=generator, dtype=dtype)[..., ::3]
               for rows in (5, 11, 11)]
    bias = torch.linspace(-0.7, 0, k.shape[2]) if biased else None
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def core(qh, kh, vh, *, attn_mask):
        calls.append((qh, kh, vh, attn_mask))
        return sdpa(qh, kh, vh, attn_mask=attn_mask)

    ops = ModuleType("comfy.ops")
    ops.scaled_dot_product_attention = core
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    got = sparse._dense_reference(q, k, v, None, key_bias=bias)
    mask = None if bias is None else bias.to(dtype).view(1, 1, 1, -1)
    want = sdpa(q, k, v, attn_mask=mask).transpose(1, 2)
    assert len(calls) == 1  # Previously the all-selected reference bypassed Core.
    for received, source in zip(calls[0][:3], (q, k, v), strict=True):
        assert received is source
    if bias is None:
        assert calls[0][3] is None
    else:
        torch.testing.assert_close(calls[0][3], mask, rtol=0, atol=0)
        assert bias.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got.shape == (1, 5, 2, 8)


def test_standalone_reference_and_explicit_provider_remain_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.ops", None)
    q = torch.randn(1, 2, 5, 8)
    got = sparse._dense_reference(q, q, q, None)
    want = torch.nn.functional.scaled_dot_product_attention(q, q, q).transpose(1, 2)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    supplied = torch.zeros(1, 5, 2, 8)
    assert sparse._dense_reference(q, q, q, lambda *_: supplied) is supplied
    with pytest.raises(RuntimeError, match="Dense SOL reference returned"):
        sparse._dense_reference(q, q, q, lambda *_: q)


def test_core_reference_error_propagates_without_retry_or_success_count(monkeypatch):
    ops = ModuleType("comfy.ops")
    calls = []

    def fail(*args, **kwargs):
        calls.append("core")
        raise torch.OutOfMemoryError("reference failure")

    def retry(*args, **kwargs):
        raise AssertionError("A failed Core reference must not be retried")

    ops.scaled_dot_product_attention = fail
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    monkeypatch.setattr(sparse.F, "scaled_dot_product_attention", retry)
    state = Request(Config())
    token = _REQUEST.set(state)
    try:
        q = torch.zeros(1, 1, 2, 8)
        with pytest.raises(torch.OutOfMemoryError, match="reference failure"):
            sparse._dense_reference(q, q, q, None)
    finally:
        _REQUEST.reset(token)
    assert calls == ["core"]
    assert getattr(state, "native_core_reference_calls", 0) == 0
    assert getattr(state, "native_torch_reference_calls", 0) == 0


@pytest.mark.parametrize("core_available", [False, True])
def test_arithmetic_checks_repeat_for_each_request_and_keep_layout_cache_local(monkeypatch, core_available):
    sdpa = torch.nn.functional.scaled_dot_product_attention
    ops = ModuleType("comfy.ops")
    ops.scaled_dot_product_attention = sdpa
    monkeypatch.setitem(sys.modules, "comfy.ops", ops if core_available else None)

    def kernel(q, k, v, **_kwargs):
        return sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda _device: kernel)
    q = torch.randn(1, 2, 5, 128, dtype=torch.bfloat16)
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    states = [Request(cfg), Request(cfg)]
    for state in states:
        token = _REQUEST.set(state)
        try:
            sparse.attention(q, q, q, 0, cfg, state)
            sparse.attention(q, q, q, 0, cfg, state)
        finally:
            _REQUEST.reset(token)
        assert len(state.gates) == 1
        assert state.sparse_calls == 2
        assert getattr(state, f"native_{'core' if core_available else 'torch'}_reference_calls") == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_statistics_equal_previous_scalar_results_with_one_host_read(monkeypatch, dtype):
    generator = torch.Generator().manual_seed(867)
    want = torch.randn(1, 2, 257, 8, generator=generator, dtype=dtype)[..., ::2]
    got = want.clone()
    got[0, 0, 3, 1] += 0.125
    delta = got.float() - want.float()
    peak = float(want.float().abs().max().item())
    previous = {
        "finite": True,
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(delta) /
                         torch.linalg.vector_norm(want.float()).clamp_min(1e-12)).item()),
        "reference_peak_abs": peak,
        "catastrophic_max_abs_limit": max(0.5, 4.0 * peak),
    }
    host_reads = []
    tolist = torch.Tensor.tolist

    def copy_statistics(tensor):
        host_reads.append(tuple(tensor.shape))
        return tolist(tensor)

    def scalar_read(_tensor):
        raise AssertionError("The gate must transfer its statistics together")

    monkeypatch.setattr(torch.Tensor, "tolist", copy_statistics)
    monkeypatch.setattr(torch.Tensor, "item", scalar_read)
    got_metrics = sparse.error_metrics(got, want)
    assert host_reads == [(5,)]
    assert got_metrics == previous
    assert sparse.arithmetic_gate_passes(got_metrics) == sparse.arithmetic_gate_passes(previous)


@pytest.mark.parametrize("side,value", [("got", math.nan), ("got", math.inf), ("want", -math.inf)])
def test_nonfinite_statistics_still_reject_the_gate(side, value):
    inputs = {"got": torch.ones(2, 3), "want": torch.ones(2, 3)}
    inputs[side][0, 1] = value
    metrics = sparse.error_metrics(**inputs)
    assert metrics["finite"] is False
    assert metrics["max_abs"] == metrics["mean_abs"] == metrics["rel_l2"] == math.inf
    assert math.isnan(metrics["reference_peak_abs"])
    assert not sparse.arithmetic_gate_passes(metrics)


def test_finite_inputs_with_overflowing_delta_are_rejected():
    got = torch.full((2, 3), torch.finfo(torch.float32).max)
    metrics = sparse.error_metrics(got, -got)
    assert not metrics["finite"]
    assert not sparse.arithmetic_gate_passes(metrics)


def test_zero_reference_has_unchanged_norm_floor_and_peak_limit():
    metrics = sparse.error_metrics(torch.zeros(2, 3), torch.zeros(2, 3))
    assert metrics == {"finite": True, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0,
                       "reference_peak_abs": 0.0, "catastrophic_max_abs_limit": 0.5}
    assert sparse.arithmetic_gate_passes(metrics)
