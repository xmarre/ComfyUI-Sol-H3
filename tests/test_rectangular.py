"""Rectangular host contracts; only gpu-marked cases execute Sana/CuTe."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sol_h3 import runtime, sparse
from sol_h3._vendor.sol_attn import interface
from sol_h3.contracts import Config
from sol_h3.interop import VDN_KEY_V2
from sol_h3.runtime import BlockPatch, Request, _FORWARD


def dense(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2)).transpose(1, 2)


@pytest.mark.parametrize('tq,tk', [(1, 193), (63, 257), (65, 130), (130, 130)])
def test_rectangular_gate_and_kv_cache(monkeypatch, tq, tk):
    calls = []
    def kernel(q, k, v, **kw):
        calls.append((q.shape[1], k.shape[1], kw['sink_tokens']))
        return dense(q, k, v)
    monkeypatch.setattr(sparse, 'load_kernel', lambda device: kernel)
    state = Request(Config(exact=False, backend='sol'))
    q = torch.randn(1, 2, tq, 128, dtype=torch.bfloat16)
    for length in (tk, tk, tk + 1):
        k, v = (torch.randn(1, 2, length, 128, dtype=q.dtype) for _ in range(2))
        out = sparse.attention(q, k, v, 65, state.config, state, recompute_prefix_queries=False)
        want = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, tq, -1)
        assert torch.equal(out, want)
    assert calls == [(tq, tk, tk), (tq, tk, 65), (tq, tk, 65),
                     (tq, tk + 1, tk + 1), (tq, tk + 1, 65)]
    assert len(state.gates) == 2 and state.sparse_calls == 3


def test_interface_kv_sink_validation(monkeypatch):
    class Tensor:
        ndim = 4
        dtype = torch.bfloat16
        device = torch.device('cuda')
        def __init__(self, rows):
            self.shape = (1, rows, 2, 128)
        def stride(self, _):
            return 1
    q, k = Tensor(3), Tensor(193)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (12, 0))
    assert interface._validate_inputs(q, k, k, 'diag', 193, 0) == (12, 0)
    with pytest.raises(ValueError, match='Tkv'):
        interface._validate_inputs(q, k, k, 'diag', 194, 0)
    with pytest.raises(ValueError, match='Tkv'):
        interface._validate_inputs(q, k, k, 'diag', 65, 129)
    with pytest.raises(ValueError, match='share'):
        interface._validate_inputs(q, k, Tensor(192), 'diag')


def test_public_rejects_unaudited_rectangular_backends(monkeypatch):
    q = torch.zeros(1, 3, 1, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 193, 1, 128, dtype=q.dtype)
    monkeypatch.setattr(interface, '_validate_inputs', lambda *a: (9, 0))
    monkeypatch.setattr(interface, '_cute_runtime_available', lambda: True)
    with pytest.raises(ValueError, match='requires cute_sm120'):
        interface.sol_attn(q, k, k)


def test_cute_host_output_lse_sink_and_compile_cache(monkeypatch):
    from sol_h3._vendor.sol_attn import preprocess
    monkeypatch.setattr(torch.cuda, 'device', lambda _: nullcontext())
    monkeypatch.setattr(interface, '_stream', lambda _: None)
    monkeypatch.setattr(interface, '_to_cute_tensors', lambda tensors: tensors)
    monkeypatch.setattr(interface, '_compiled', {})
    prepares, compiles = [], []
    def prepare(q, k, v, **kw):
        prepares.append((q.shape[1], k.shape[1], kw))
        kc = torch.empty(1, (k.shape[1] + 63)//64, 2, 128)
        return kc, kc, torch.empty(1, (q.shape[1] + 63)//64, 2)
    def compile_(key, tensors, scale, start, end, stream):
        compiles.append((key, [t.shape for t in tensors], start, end))
        def compiled(*args, **kwargs):
            args[3].copy_(args[0])
        interface._compiled[key] = compiled
        return compiled, tensors
    monkeypatch.setattr(preprocess, 'prepare', prepare)
    monkeypatch.setattr(interface, '_compile_sm120', compile_)
    for tq, tk in [(65, 193), (65, 193), (65, 257), (63, 257)]:
        q = torch.zeros(1, tq, 2, 128, dtype=torch.bfloat16)
        k = torch.zeros(1, tk, 2, 128, dtype=q.dtype)
        got = interface._sol_attn_cute(q, k, k, arch=(12, 0), scale=128**-.5,
                    tau=1., thresh_type='diag', kv_splits=1, sink_tokens=tk, sink_start=0,
                    valid_tokens=tq)
        assert got.shape == q.shape
    assert len(compiles) == 3
    for _, shapes, start, end in compiles:
        assert shapes[3] == shapes[0]
        assert shapes[7] == shapes[0][:3]  # LSE: [B,Tq,H]
        assert start == 0 and end == (shapes[1][1] + 63)//64
    assert all(kw['valid_tokens'] == tq and kw['valid_kv_tokens'] == tk
               for tq, tk, kw in prepares)


def test_diag_threshold_uses_all_kv_blocks(monkeypatch):
    from sol_h3._vendor.sol_attn import preprocess as p
    class Launch:
        def __init__(self):
            self.calls = []
        def __getitem__(self, grid):
            def call(*args, **kw):
                self.calls.append((grid, args))
            return call
    stats, threshold = Launch(), Launch()
    monkeypatch.setattr(p, '_reduce_kc_stats_kernel', stats)
    monkeypatch.setattr(p, '_diag_threshold_kernel', threshold)
    monkeypatch.setattr(p, 'TensorDescriptor', SimpleNamespace(from_tensor=lambda t, _: t))
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    kc = torch.zeros(1, 8, 2, 128, dtype=q.dtype)
    out = p._compute_diag_threshold(q, kc, tau=1., scale=128**-.5, valid_kv_tokens=449)
    assert out.shape == (1, 2, 2)
    assert stats.calls[0][1][3:6] == (8, 2, 8)
    assert threshold.calls[0][0] == (2, 2)
    assert threshold.calls[0][1][6:9] == (65, 2, 2)


def test_v2_direct_q_ignores_square_payload_and_preserves_domain(monkeypatch):
    cfg = Config(exact=False, backend='sol', dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    token = _FORWARD.set((SimpleNamespace(blocks=[object()]), state, 0, set(), []))
    monkeypatch.setattr(runtime, '_shape_reason', lambda *a, **kw: None)
    q = torch.randn(3, 2, 128, dtype=torch.bfloat16)
    k, v = (torch.randn(193, 2, 128, dtype=q.dtype) for _ in range(2))
    def attention(qc, kc, vc, sink, config, request, **kw):
        assert torch.equal(qc[0].transpose(0, 1), q)
        assert torch.equal(kc[0].transpose(0, 1), k)
        assert torch.equal(vc[0].transpose(0, 1), v)
        assert sink == 65 and kw['recompute_prefix_queries'] is False
        request.sparse_calls += 1
        return q.reshape(1, 3, -1)
    monkeypatch.setattr(sparse, 'attention', attention)
    def block(args):
        provider = args['transformer_options'][VDN_KEY_V2]
        def native():
            raise AssertionError('unexpected fallback')
        for payload in ({}, {'square_q': object(), 'query_positions': object()}):
            assert torch.equal(provider(native, q, k, v, kind='local', scale=128**-.5,
                                        sink_rows=65, **payload), q)
        for kind in ('global', 'anchor', 'flex_masked'):
            assert provider(lambda: kind, q, k, v, kind=kind, scale=128**-.5) == kind
        return {}
    try:
        BlockPatch(0, cfg)({'transformer_options': {}}, {'original_block': block})
    finally:
        _FORWARD.reset(token)
    assert state.vdn_rectangular_sol_calls == 2
    assert state.vdn_kernel_q_rows == state.vdn_requested_q_rows == 6
    assert state.vdn_square_expanded_calls == state.vdn_square_kernel_rows == 0


@pytest.mark.gpu
@pytest.mark.parametrize('tq,tk', [(1, 257), (63, 257), (65, 449), (129, 8193), (193, 193)])
def test_real_sm120_rectangular_all_selected_and_sparse_sinks(tq, tk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('requires real SM120')
    kernel = sparse.load_kernel(torch.device('cuda'))
    assert kernel.backend_name == 'cute_sm120'
    torch.manual_seed(53)
    q = torch.randn(1, tq, 2, 128, dtype=torch.bfloat16, device='cuda')
    k, v = (torch.randn(1, tk, 2, 128, dtype=q.dtype, device=q.device) for _ in range(2))
    with torch.inference_mode():
        got = kernel(q, k, v, tau=1., sink_tokens=tk)
        assert got.shape == q.shape
        assert sparse.arithmetic_gate_passes(sparse.error_metrics(got, dense(q, k, v)))
        # Force approximate far blocks; changing prefix values must contribute
        # through exact sinks, including the outward-rounded 65-row prefix.
        for sink in (0, 1, 65):
            out = kernel(q, k, v, tau=100., sink_tokens=min(sink, tk))
            assert out.shape == q.shape and torch.isfinite(out).all()


def test_exact_threshold_moments_are_kv_based(monkeypatch):
    from sol_h3._vendor.sol_attn import preprocess as p
    class Launch:
        def __init__(self):
            self.calls = []
        def __getitem__(self, grid):
            def call(*args, **kw):
                self.calls.append((grid, args))
            return call
    pool, threshold = Launch(), Launch()
    monkeypatch.setattr(p, '_pool_query_kernel', pool)
    monkeypatch.setattr(p, '_exact_fused_threshold_kernel', threshold)
    monkeypatch.setattr(p, 'TensorDescriptor', SimpleNamespace(from_tensor=lambda t, _: t))
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    kc = torch.arange(8, dtype=q.dtype).reshape(1, 8, 1, 1).expand(1, 8, 2, 128).contiguous()
    out = p._compute_exact_threshold(q, kc, tau=1., scale=128**-.5, valid_kv_tokens=449)
    assert out.shape == (1, 2, 2)
    assert pool.calls[0][0] == (2, 2)
    assert torch.equal(threshold.calls[0][1][1], kc.float().mean(1))
    assert torch.equal(threshold.calls[0][1][2].float(), torch.full((1, 2, 128, 128), 17.5))


def forced_route_reference(q, k, v, sink):
    """Independent compressed-softmax oracle with tau=inf and ordinal local blocks.

    Approximate blocks use BF16 K centroids / V sums, as Sana does. Actual
    block mass is its KV row count; exact blocks expand into individual rows.
    """
    parts = []
    for qs in range(0, q.shape[1], 64):
        keys, values, masses = [], [], []
        for ks in range(0, k.shape[1], 64):
            kb, vb = k[:, ks:ks+64], v[:, ks:ks+64]
            if abs(qs//64 - ks//64) <= 1 or ks < sink:
                keys.append(kb.float())
                values.append(vb.float())
                masses.extend([1.] * kb.shape[1])
            else:
                keys.append(kb.float().mean(1, keepdim=True).bfloat16().float())
                values.append(vb.float().sum(1, keepdim=True).bfloat16().float() / kb.shape[1])
                masses.append(float(kb.shape[1]))
        logits = torch.einsum('bqhd,bkhd->bhqk', q[:, qs:qs+64].float(), torch.cat(keys, 1)) * 128**-.5
        logits += torch.tensor(masses, device=q.device).log()
        parts.append(torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), torch.cat(values, 1)))
    return torch.cat(parts, 1).bfloat16()


@pytest.mark.gpu
@pytest.mark.parametrize('sink', [0, 1, 65, 193])
def test_real_sm120_sink_and_approximate_kv_tail_mass(sink):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('requires real SM120')
    kernel = sparse.load_kernel(torch.device('cuda'))
    torch.manual_seed(71)
    # Four Q blocks, eight KV blocks; both have tails. B=2 also audits LSE batch stride.
    q = torch.randn(2, 193, 2, 128, device='cuda', dtype=torch.bfloat16)
    k, v = (torch.randn(2, 449, 2, 128, device='cuda', dtype=q.dtype) for _ in range(2))
    with torch.inference_mode():
        got = kernel(q, k, v, tau=float('inf'), sink_tokens=sink)
        want = forced_route_reference(q, k, v, sink)
        metrics = sparse.error_metrics(got, want)
        assert sparse.arithmetic_gate_passes(metrics), metrics
