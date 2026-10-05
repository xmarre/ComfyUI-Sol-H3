"""Dynamic machine functions retain live arguments and physical arithmetic gates."""

from contextlib import nullcontext

import pytest
import torch

from sol_h3 import partitioned_request as request, provenance, sparse
from sol_h3._vendor.sol_attn import interface, preprocess
from sol_h3.contracts import Config
from sol_h3.runtime import Request, _REQUEST


def _inputs(rows, kv_rows, *, strided=False, device="cpu"):
    values = []
    for count in (rows, kv_rows, kv_rows):
        if strided:
            value = torch.randn(count, 2, 3, 128, dtype=torch.bfloat16, device=device)[:, :, 1, :]
        else:
            value = torch.randn(count, 2, 128, dtype=torch.bfloat16, device=device)
        values.append(value)
    return values


def _reference(q, k, v, bias):
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * 128 ** -0.5
    if bias is not None:
        scores += bias
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()).bfloat16()


def _launch(q, k, v, *, bias=None, mapped=None):
    return request._sm120_union(
        q, k, v, tau=float("inf"), scale=128 ** -0.5, sink_rows=len(k),
        key_bias=bias, mapped_neighbor_intervals=mapped,
    )


def test_dynamic_compile_reuses_shapes_and_strides_but_rebinds_all_live_arguments(monkeypatch):
    compiles, launches = [], []
    monkeypatch.setattr(provenance, "verify_source", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(interface, "_validate_cute", lambda *args: None)
    monkeypatch.setattr(interface, "_stream", lambda _: None)
    monkeypatch.setattr(interface, "_compiled", {})
    monkeypatch.setattr(interface, "_to_cute_tensors", lambda tensors: tensors)

    def prepare(q, k, v, **kwargs):
        kc = torch.empty(1, (k.shape[1] + 63) // 64, q.shape[2], 128, dtype=q.dtype)
        return kc, torch.empty_like(kc), torch.empty(1, (q.shape[1] + 63) // 64, q.shape[2])

    def compile_(key, tensors, scale, start, end, stream, biased, mapped):
        compiles.append((biased, mapped))

        def compiled(*args, **kwargs):
            q, k, v, output = args[:4]
            launches.append((q.shape[1], k.shape[1], args[-2:]))
            output.copy_(_reference(q[0], k[0], v[0], args[7] if biased else None).unsqueeze(0))

        interface._compiled[key] = compiled
        return compiled, tensors

    monkeypatch.setattr(preprocess, "prepare", prepare)
    monkeypatch.setattr(interface, "_compile_sm120", compile_)
    state = Request(Config())
    token = _REQUEST.set(state)
    try:
        for biased, mapped in ((False, False), (True, False), (True, True)):
            for rows, kv_rows, strided in ((17, 75, False), (33, 131, True), (63, 193, False)):
                q, k, v = _inputs(rows, kv_rows, strided=strided)
                bias = torch.linspace(-0.7, 0.1, kv_rows) if biased else None
                descriptor = torch.zeros((rows + 63) // 64, 2, dtype=torch.int32) if mapped else None
                got = _launch(q, k, v, bias=bias, mapped=descriptor)
                torch.testing.assert_close(got, _reference(q, k, v, bias), rtol=0, atol=0)
    finally:
        _REQUEST.reset(token)
    assert compiles == [(False, False), (True, False), (True, True)]
    assert [row[:2] for row in launches] == [(17, 75), (33, 131), (63, 193)] * 3
    assert all(row[2] == (0, (row[1] + 63) // 64) for row in launches)
    assert state.partitioned_kernel_compile_calls == 3
    assert state.partitioned_kernel_compile_cache_hits == 6
    assert state.partitioned_kernel_compile_wall_s >= 0.0


def test_dynamic_tensor_type_keeps_broadcast_rank_dtype_and_leading_stride_distinct():
    dense = torch.empty(2, 3, 128, dtype=torch.bfloat16)
    assert request._dynamic_tensor_abi(dense) != request._dynamic_tensor_abi(dense[:1].expand(2, 3, 128))
    assert request._dynamic_tensor_abi(dense) != request._dynamic_tensor_abi(dense.float())
    assert request._dynamic_tensor_abi(dense) != request._dynamic_tensor_abi(dense.unsqueeze(0))
    with pytest.raises(RuntimeError, match="unit stride"):
        request._dynamic_tensor_abi(dense[..., ::2])


@pytest.mark.gpu
@pytest.mark.parametrize("biased", [False, True])
def test_real_sm120_dynamic_compile_reuses_rectangles_and_interleaved_views(monkeypatch, biased):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires real SM120 and packaged CuTe")
    monkeypatch.setattr(interface, "_compiled", {})
    original_compile = interface._compile_sm120
    compiles = []

    def compile_(*args, **kwargs):
        compiles.append(args[0])
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(interface, "_compile_sm120", compile_)
    with torch.inference_mode():
        for rows, kv_rows, strided in ((17, 75, False), (33, 131, True), (63, 193, False)):
            q, k, v = _inputs(rows, kv_rows, strided=strided, device="cuda")
            bias = torch.linspace(-0.7, 0.1, kv_rows, device="cuda") if biased else None
            got = _launch(q, k, v, bias=bias)
            metrics = sparse.error_metrics(got, _reference(q, k, v, bias))
            assert sparse.arithmetic_gate_passes(metrics), metrics
    assert len(compiles) == 1
