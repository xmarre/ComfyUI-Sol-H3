"""Weighted dense requests keep every key and an independent native oracle."""

import math
import sys
from types import ModuleType

import pytest
import torch

from sol_h3 import partitioned_request as request
from sol_h3.contracts import Config
from sol_h3.interop import RECEIPTS_KEY
from sol_h3.partitioned_history import _accept_partitioned_receipt
from sol_h3.runtime import Request, _FORWARD, _REQUEST


def _oracle(q, k, v, bias, scale):
    scores = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    if bias is not None:
        scores += bias.double()
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.double()).to(q.dtype)


def _host_runtime(monkeypatch):
    native_calls, kernel_calls = [], []
    ops = ModuleType("comfy.ops")

    def native(q, k, v, *, attn_mask, scale):
        native_calls.append(attn_mask.clone())
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)

    def kernel(q, k, v, **kwargs):
        kernel_calls.append(kwargs)
        return _oracle(q, k, v, kwargs["key_bias"], kwargs["scale"])

    def bias(_state, *, kv_rows, prefix_range, log_measure, **kwargs):
        if log_measure == 0.0:
            return None
        result = torch.zeros(kv_rows)
        result[slice(*prefix_range)] = log_measure
        return result

    ops.scaled_dot_product_attention = native
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    monkeypatch.setattr(request, "_validate_thd", lambda *_: None)
    monkeypatch.setattr(request, "_sm120_union", kernel)
    monkeypatch.setattr(request, "_key_bias", bias)
    return native_calls, kernel_calls


def _inputs(rows=65, kv_rows=193):
    gen = torch.Generator().manual_seed(1000)
    return [torch.randn((n, 2, 128), generator=gen, dtype=torch.bfloat16) for n in (rows, kv_rows, kv_rows)]


def _call(q, k, v, *, kind="local", measure=math.log(330 / 672), prefix_range=(11, 79), force_dense=True):
    options = {RECEIPTS_KEY: []}
    result = request.partitioned_request_attention(
        q, k, v, transformer_options=options, block_index=3, kind=kind,
        scale=128 ** -0.5, sink_rows=11, prefix_k_range=prefix_range,
        prefix_log_key_measure=measure, semantic_digest="a" * 64, force_dense=force_dense,
    )
    return result, options[RECEIPTS_KEY][0]


@pytest.mark.parametrize("kind", ["global", "local", "anchor"])
def test_weighted_dense_uses_all_keys_and_references_native_only_once(monkeypatch, kind):
    native, kernels = _host_runtime(monkeypatch)
    q, k, v = _inputs()
    state = Request(Config(backend="sol"))
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 0, None, []))
    try:
        outputs = [_call(q, k, v, kind=kind) for _ in range(2)]
        assert all(_accept_partitioned_receipt(receipt) for _, receipt in outputs)
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)
    assert len(kernels) == 2
    assert len(native) == 1
    assert all(call["sink_rows"] == len(k) and call["mapped_neighbor_intervals"] is None for call in kernels)
    rounded = torch.zeros(len(k))
    rounded[11:79] = torch.tensor(math.log(330 / 672), dtype=q.dtype).float()
    for call in kernels:
        torch.testing.assert_close(call["key_bias"], rounded, rtol=0, atol=0)
    assert len(state.gates) == 1 and state.partitioned_weighted_dense_calls == 2
    assert state.partitioned_core_dense_calls == 1
    assert outputs[0][1][3] == outputs[1][1][3]
    assert outputs[0][1][3][4] == "dense_sm120_forced"
    for output, _ in outputs:
        torch.testing.assert_close(output, _oracle(q, k, v, rounded, 128 ** -0.5), rtol=0, atol=0)


def test_weighted_dense_rechecks_requests_measure_range_and_layout(monkeypatch):
    native, kernels = _host_runtime(monkeypatch)
    q, k, v = _inputs()
    for _ in range(2):
        state = Request(Config(backend="sol"))
        token = _REQUEST.set(state)
        forward = _FORWARD.set((None, state, 0, None, []))
        try:
            _call(q, k, v)
            _call(q, k, v)
            _call(q, k, v, measure=-1.2)
            _call(q, k, v, prefix_range=(11, 80))
            # Preserve shape while changing projection strides.
            padded = torch.empty((len(q), 4, 128), dtype=q.dtype)
            strided = padded[:, ::2, :]
            strided.copy_(q)
            _call(strided, k, v)
        finally:
            _FORWARD.reset(forward)
            _REQUEST.reset(token)
        assert len(state.gates) == 4
    assert len(native) == 8 and len(kernels) == 10


def test_weighted_dense_warmup_receipt_and_bounded_metadata_cache(monkeypatch):
    native, _ = _host_runtime(monkeypatch)
    monkeypatch.setattr(request, "MAX_WEIGHTED_DENSE_GATE_ENTRIES", 2)
    q, k, v = _inputs()
    state = Request(Config(backend="sol", tau=3.0))
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 0, None, []))
    try:
        for end in (79, 80, 81, 79):
            _, receipt = _call(q, k, v, prefix_range=(11, end), force_dense=False)
            assert _accept_partitioned_receipt(receipt)
            assert receipt[3][4] == "dense_sm120_warmup"
        assert len(state.partitioned_weighted_dense_verified) == 2
        assert all(value is None for value in state.partitioned_weighted_dense_verified.values())
        assert state.dense_calls == 4
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)
    assert len(native) == 4  # An evicted layout needs its independent check again.


@pytest.mark.parametrize("field,value", [(4, "dense_forced"), (9, 0.0), (13, "unknown-kernel")])
def test_weighted_dense_history_rejects_wrong_kernel_mode_or_unit_measure(monkeypatch, field, value):
    _host_runtime(monkeypatch)
    q, k, v = _inputs()
    state = Request(Config(backend="sol"))
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 0, None, []))
    try:
        _, receipt = _call(q, k, v)
        fields = list(receipt[3])
        fields[field] = value
        malformed = (*receipt[:3], tuple(fields))
        state.partitioned_validated_receipts.add((receipt[1], malformed[3]))
        assert not _accept_partitioned_receipt(malformed)
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)


@pytest.mark.parametrize("failure", ["mismatch", "native", "kernel"])
def test_failed_weighted_dense_cannot_publish_completion_or_cache_success(monkeypatch, failure):
    native, _ = _host_runtime(monkeypatch)
    q, k, v = _inputs()
    if failure == "mismatch":
        monkeypatch.setattr(request, "_sm120_union", lambda *args, **kwargs: torch.zeros_like(q))
    else:
        def fail(*args, **kwargs):
            raise torch.OutOfMemoryError("sentinel weighted failure")
        if failure == "native":
            sys.modules["comfy.ops"].scaled_dot_product_attention = fail
        else:
            monkeypatch.setattr(request, "_sm120_union", fail)
    state = Request(Config(backend="sol"))
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 0, None, []))
    try:
        with pytest.raises(RuntimeError):
            _call(q, k, v)
        assert not getattr(state, "partitioned_weighted_dense_verified", {})
        assert not getattr(state, "partitioned_validated_receipts", set())
        assert not getattr(state, "partitioned_weighted_dense_calls", 0)
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)
    assert len(native) == int(failure == "mismatch")


@pytest.mark.gpu
@pytest.mark.parametrize("q_rows,kv_rows", [(65, 193), (672, 3011)])
def test_real_sm120_weighted_dense_request_and_native_gate(q_rows, kv_rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires real SM120")
    from sol_h3.provenance import verify_source
    verify_source()
    q, k, v = [t.cuda() for t in _inputs(q_rows, kv_rows)]
    state = Request(Config(backend="sol"))
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 0, None, []))
    try:
        with torch.inference_mode():
            got, receipt = _call(q, k, v)
            assert _accept_partitioned_receipt(receipt)
            again, _ = _call(q, k, v)
            torch.testing.assert_close(got, again, rtol=0, atol=0)
            rounded = torch.zeros(len(k), device=q.device)
            rounded[11:79] = torch.tensor(math.log(330 / 672), dtype=q.dtype).float()
            from sol_h3.sparse import arithmetic_gate_passes, error_metrics
            metrics = error_metrics(got, _oracle(q, k, v, rounded, 128 ** -0.5))
            assert arithmetic_gate_passes(metrics), metrics
        assert len(state.gates) == 1 and state.partitioned_weighted_dense_calls == 2
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)


def test_weighted_dense_may_extend_the_measure_over_the_global_sink(monkeypatch):
    native, kernels = _host_runtime(monkeypatch)
    q, k, v = _inputs()
    state = Request(Config(backend="sol"))
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 0, None, []))
    try:
        output, receipt = _call(q, k, v, prefix_range=(0, 79))
        assert _accept_partitioned_receipt(receipt)
        with pytest.raises(RuntimeError, match="overlaps the global sink"):
            _call(q, k, v, prefix_range=(5, 79))
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)
    assert request.PARTITIONED_SINK_MEASURE_API == 1
    rounded = torch.zeros(len(k))
    rounded[0:79] = torch.tensor(math.log(330 / 672), dtype=q.dtype).float()
    torch.testing.assert_close(kernels[-1]["key_bias"], rounded, rtol=0, atol=0)
    torch.testing.assert_close(output, _oracle(q, k, v, rounded, 128 ** -0.5), rtol=0, atol=0)
