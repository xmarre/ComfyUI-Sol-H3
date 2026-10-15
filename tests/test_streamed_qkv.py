"""Projection ownership, global query coordinates and invocation lifetimes."""
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import sys
import weakref

import pytest
import torch
import torch.nn.functional as F

from sol_h3 import sparse, streamed
from sol_h3.contracts import Config
from sol_h3.runtime import Request
from sol_h3._vendor.sol_attn import interface


@pytest.mark.parametrize("rows", [1, 63, 64, 65, 129, 193])
def test_two_pass_producer_drops_each_projection_and_kv_before_out_projection(rows):
    heads, dim = 2, 128
    torch.manual_seed(rows)
    x = torch.randn(rows, 3 * heads * dim, dtype=torch.bfloat16)
    chunks, roots, starts, scopes, kv_refs = [], [], [], [], []

    def project(attn, x, rope, start, stop):
        assert all(ref() is None for ref in roots)
        packed = x[start:stop].clone()
        roots.append(weakref.ref(packed))
        chunks.append((start, stop))
        return tuple(part.view(1, stop - start, heads, dim) for part in packed.split(heads * dim, -1))

    def attend(q, k, v, heads, options):
        scope = streamed._STREAM.get()
        scope.check(q, k, v)
        starts.append(scope.query_start)
        scopes.append(scope)
        kv_refs.extend((weakref.ref(k), weakref.ref(v)))
        # Simulate an invocation-owned prepared KV object holding both inputs.
        scope.prepared_kv = (k, v)
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)

    def out_proj(out):
        assert all(ref() is None for ref in roots + kv_refs)
        assert streamed._STREAM.get() is None
        assert all(scope.k is scope.v is scope.prepared_kv is None for scope in scopes)
        return out

    attn = SimpleNamespace(heads=heads, head_dim=dim, out_proj=out_proj)
    with torch.inference_mode():
        got = streamed.attention(attn, x, None, {}, 64, project=project, attend=attend)
        q, k, v = (p.view(1, rows, heads, dim).transpose(1, 2) for p in x.split(heads * dim, -1))
        want = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(rows, -1)
    assert torch.equal(got, want)
    intervals = [(i, min(rows, i + 64)) for i in range(0, rows, 64)]
    assert chunks == intervals * 2
    assert starts == [i for i, _ in intervals]


def test_nested_stream_and_exception_restore_context_and_clear_workspace():
    original = object()
    token = streamed._STREAM.set(original)
    scopes = []
    attn = SimpleNamespace(heads=1, head_dim=128, out_proj=lambda x: x)
    x = torch.zeros(65, 384, dtype=torch.bfloat16)

    def project(attn, x, rope, start, stop):
        return tuple(t.view(1, stop - start, 1, 128) for t in x[start:stop].clone().split(128, -1))

    def fail(q, k, v, heads, options):
        scope = streamed._STREAM.get()
        scope.prepared_kv = (k, v)
        scopes.append(scope)
        raise RuntimeError("cancelled")

    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            streamed.attention(attn, x, None, {}, 64, project=project, attend=fail)
        assert streamed._STREAM.get() is original
        assert scopes[0].k is scopes[0].v is scopes[0].prepared_kv is None
    finally:
        streamed._STREAM.reset(token)


@pytest.mark.parametrize("fail_call", [1, 3])
def test_projection_failure_releases_owned_kv_with_traceback_retained(monkeypatch, fail_call):
    roots = []
    native_empty = torch.Tensor.new_empty
    native_like = torch.empty_like

    def empty(x, *args, **kwargs):
        out = native_empty(x, *args, **kwargs)
        roots.append(weakref.ref(out))
        return out

    def empty_like(*args, **kwargs):
        out = native_like(*args, **kwargs)
        roots.append(weakref.ref(out))
        return out

    monkeypatch.setattr(torch.Tensor, "new_empty", empty)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    calls = 0

    def project(attn, x, rope, start, stop):
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise RuntimeError("projection cancelled")
        return tuple(t.view(1, stop - start, 1, 128) for t in x[start:stop].clone().split(128, -1))

    x = torch.zeros(65, 384, dtype=torch.bfloat16)
    attn = SimpleNamespace(heads=1, head_dim=128, out_proj=lambda x: x)
    with pytest.raises(RuntimeError, match="projection cancelled") as error:
        streamed.attention(attn, x, None, {}, 64, project=project)
    assert error.value.__traceback__ is not None
    assert roots and all(ref() is None for ref in roots)
    assert streamed._STREAM.get() is None


@pytest.mark.parametrize("start,expected_prefix", [(0, 64), (64, 1), (128, 0)])
def test_stream_bridge_preserves_global_prefix_and_query_origin(monkeypatch, start, expected_prefix):
    calls, dense_rows, preparations = [], [], []

    def kernel(q, k, v, **kw):
        calls.append(kw)
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    def prepare(k, v):
        preparations.append(1)
        return object()

    kernel.prepare_kv = prepare
    state = Request(Config(exact=False, backend="sol"))
    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    q = torch.randn(1, 2, 64, 128, dtype=torch.bfloat16)
    k, v = (torch.randn(1, 2, 193, 128, dtype=q.dtype) for _ in range(2))
    workspace = streamed.QueryStream(193, k, v, query_start=start)

    def dense(q, k, v):
        dense_rows.append(q.shape[2])
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    for _ in range(2):
        sparse.attention(q, k, v, 65, state.config, state, dense_attention=dense,
                         query_start=start, kv_workspace=workspace)
    assert preparations == [1]
    assert dense_rows == ([expected_prefix] * 2 if expected_prefix else [])
    assert len(state.gates) == 1
    assert all(c.get("query_start", 0) == start for c in calls)
    assert all(c["prepared_kv"] is workspace.prepared_kv for c in calls)
    assert [c["sink_tokens"] for c in calls] == [193, 65, 65]


def test_prepared_kv_checks_storage_offset_and_view_geometry(monkeypatch):
    from sol_h3._vendor.sol_attn import preprocess
    monkeypatch.setattr(interface, "_validate_inputs", lambda *args: (12, 0))
    monkeypatch.setattr(interface, "_backend_for_arch", lambda arch: "cute_sm120")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    stream_id = [7]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=stream_id[0]))
    preparations = []
    monkeypatch.setattr(preprocess, "_reduce_kv", lambda k, v: preparations.append((k, v)) or (object(), object()))
    packed = torch.empty(1, 193, 2, 3 * 128, dtype=torch.bfloat16)
    k, v = packed[..., 128:256], packed[..., 256:]
    prepared = interface.PreparedKV(k, v)
    prepared.check(k.view_as(k), v.view_as(v))
    assert len(preparations) == 1
    with pytest.raises(ValueError, match="belong"):
        prepared.check(k.clone(), v)
    with pytest.raises(ValueError, match="belong"):
        prepared.check(packed[..., :128], v)
    stream_id[0] = 9
    with pytest.raises(ValueError, match="allocating CUDA stream"):
        prepared.check(k, v)
    stream_id[0] = 7
    k.add_(1)
    with pytest.raises(ValueError, match="modified"):
        prepared.check(k, v)


def test_bounded_statistics_match_independent_fp64_oracle_on_strided_outputs(monkeypatch):
    # Force several tiles without allocating a large test fixture.
    monkeypatch.setattr(sparse, "METRIC_CHUNK_ELEMENTS", 256)
    torch.manual_seed(219)
    want = torch.randn(2, 97, 3, 8, dtype=torch.bfloat16)[..., ::2]
    got = want.clone()
    got[1, 73, 2, 1] += 0.125
    sizes = [t.numel() for t in sparse._metric_tiles(want)]
    assert max(sizes) <= 256 and sum(sizes) == want.numel()
    delta = got.double() - want.double()
    metrics = sparse.error_metrics(got, want)
    assert metrics["mean_abs"] == pytest.approx(delta.abs().mean().item(), rel=1e-12)
    assert metrics["rel_l2"] == pytest.approx((delta.norm() / want.double().norm()).item(), rel=1e-12)
    assert metrics["max_abs"] == delta.abs().max().item()
    assert metrics["reference_peak_abs"] == want.abs().max().item()
    got[1, -1, -1, -1] = float("nan")
    assert not sparse.arithmetic_gate_passes(sparse.error_metrics(got, want))


def test_qualification_visits_every_query_and_does_not_retain_gate_outputs(monkeypatch):
    monkeypatch.setattr(sparse, "GATE_QUERY_CHUNK_SIZE", 64)
    starts, refs, prepares = [], [], []

    def kernel(q, k, v, **kwargs):
        if kwargs["sink_tokens"] == k.shape[1]:
            assert all(ref() is None for ref in refs)
            starts.append((kwargs.get("query_start", 0), q.shape[1]))
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()
        refs.append(weakref.ref(out))
        return out

    kernel.prepare_kv = lambda k, v: prepares.append(1) or object()
    monkeypatch.setattr(sparse, "load_kernel", lambda _: kernel)
    state = Request(Config(exact=False, backend="sol"))
    q = torch.randn(1, 2, 193, 128, dtype=torch.bfloat16)
    k, v = (torch.randn(1, 2, 257, 128, dtype=q.dtype) for _ in range(2))
    sparse.attention(q, k, v, 0, state.config, state, recompute_prefix_queries=False)
    assert starts == [(0, 64), (64, 64), (128, 64), (192, 1)]
    assert prepares == [1]
    assert all(ref() is None for ref in refs)
    assert len(state.gates) == 1


def test_first_streamed_invocation_qualifies_all_origins_without_repeating_later_blocks(monkeypatch):
    starts = []

    def kernel(q, k, v, **kwargs):
        if kwargs["sink_tokens"] == k.shape[1]:
            starts.append(kwargs.get("query_start", 0))
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    kernel.prepare_kv = lambda *args: object()
    monkeypatch.setattr(sparse, "load_kernel", lambda _: kernel)
    state = Request(Config(exact=False, backend="sol", qkv_chunk_size=64))
    x = torch.randn(193, 384, dtype=torch.bfloat16)
    attn = SimpleNamespace(heads=1, head_dim=128, out_proj=lambda x: x)

    def project(attn, x, rope, start, stop):
        return tuple(t.view(1, stop - start, 1, 128) for t in x[start:stop].clone().split(128, -1))

    def attend(q, k, v, heads, options):
        scope = streamed._STREAM.get()
        return sparse.attention(q, k, v, 0, state.config, state, query_start=scope.query_start, kv_workspace=scope)

    with torch.inference_mode():
        for _ in range(2):
            streamed.attention(attn, x, None, {}, 64, project=project, attend=attend)
            assert starts == [0, 64, 128, 192]
    assert [g["query_start"] for g in state.gates] == starts


def test_producer_admission_preserves_foreign_owners_and_projection_hooks(monkeypatch):
    class CudaFacade(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:0")

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 128
            self.qkv_proj = torch.nn.Linear(128, 384, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            return x

    class DiTBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

        def forward(self, x):
            return x

    native = ModuleType("comfy.ldm.minimax.model")
    native.Attention, native.DiTBlock = Attention, DiTBlock
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    block = DiTBlock()
    state = Request(Config(exact=False, backend="sol", qkv_chunk_size=64))
    state.streamed_native_verified[Attention.forward] = True
    x = torch.empty(513, 128, dtype=torch.bfloat16).as_subclass(CudaFacade)
    layout = SimpleNamespace(seq_len=513, segments=[(0, 65, "text"), (65, 513, "video")])
    args = {"img": x, "rope_freqs": object(), "layout": layout, "transformer_options": {}}
    with torch.inference_mode():
        assert streamed.ineligible_reason(block, args, state, None) is None
        block.attn.qkv_proj.weight_function = [lambda x: x]
        assert streamed.ineligible_reason(block, args, state, None) == "runtime_projection_patches"
        block.attn.qkv_proj.weight_function = []
        hook = block.attn.qkv_proj.register_forward_hook(lambda *args: None)
        assert streamed.ineligible_reason(block, args, state, None) == "module_hooks_or_compile"
        hook.remove()
        block.attn.forward = lambda x: x
        assert streamed.ineligible_reason(block, args, state, None) == "block_or_attention_owner"
        block.attn.forward = Attention.forward.__get__(block.attn)
        inherited = {**args, "attention": lambda x: x}
        assert streamed.ineligible_reason(block, inherited, state, None) == "inherited_attention"
        measured = {**args, "transformer_options": {"attention_measure_v1": {"api": 1}}}
        assert streamed.ineligible_reason(block, measured, state, None) == "external_sequence_or_transform"
        state.streamed_native_verified[Attention.forward] = False
        assert streamed.ineligible_reason(block, args, state, None) == "native_attention_source"


def test_history_distinguishes_producers_and_accepts_explicit_stream_receipts():
    from sol_h3.interop import HistoryPolicy
    cfg = Config(backend="sol")
    native = Config(backend="sol", qkv_chunk_size=0)
    assert cfg.metadata()["fingerprint"] != native.metadata()["fingerprint"]
    assert HistoryPolicy(cfg).accept_receipts([("sol_h3", 0, "sol_streamed"), ("sol_h3", 1, "dense_warmup_streamed")])


@pytest.mark.parametrize("rows,kv_rows", [(63, 129), (64, 193), (65, 129), (193, 257)])
@pytest.mark.parametrize("policy", ["sparse", "all_selected", "weighted", "mapped"])
@pytest.mark.parametrize("query_layout", ["compact", "projection"])
@pytest.mark.gpu
def test_real_cute_query_streaming_is_bitwise_equal_to_unsliced_kernel(rows, kv_rows, policy, query_layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("requires SM120/SM121")
    from sol_h3.provenance import verify_source
    verify_source()
    torch.manual_seed(370 + rows + kv_rows)
    heads = 56
    q = torch.randn(1, rows, heads, 128, device="cuda", dtype=torch.bfloat16)

    def query_carrier(values):
        if query_layout == "compact":
            return values
        count = values.shape[1]
        packed = values.new_empty(count, 3 * heads * 128)
        view = packed[:, :heads * 128].view(1, count, heads, 128)
        view.copy_(values)
        return view

    q = query_carrier(q)
    k, v = (torch.randn(1, kv_rows, heads, 128, device="cuda", dtype=q.dtype) for _ in range(2))
    bias = torch.linspace(-1.0, 0.5, kv_rows, device="cuda") if policy == "weighted" else None
    mapped = None
    if policy == "mapped":
        mapped = torch.tensor([[1, 3]] * ((rows + 63) // 64), device="cuda", dtype=torch.int32)
    kwargs = dict(tau=1.0, sink_start=0, sink_tokens=kv_rows if policy == "all_selected" else 65, key_bias=bias)
    with torch.inference_mode():
        baseline = interface.sol_attn(q, k, v, mapped_neighbor_intervals=mapped, **kwargs)
        prepared = interface.PreparedKV(k, v)
        output = torch.empty_like(q)
        for start in range(0, rows, 64):
            query = query_carrier(q[:, start:start + 64])
            mapping = None if mapped is None else mapped[start // 64:(start + query.shape[1] + 63) // 64]
            output[:, start:start + 64] = interface.sol_attn(
                query, k, v, query_start=start, prepared_kv=prepared,
                mapped_neighbor_intervals=mapping, **kwargs,
            )
        torch.cuda.synchronize()
    # Query blocks and K/V arithmetic are identical; no tolerance is needed.
    assert torch.equal(output, baseline)


@pytest.mark.parametrize("rows,chunk", [(1, 64), (65, 64), (193, 64), (513, 128)])
def test_vdn_raw_qkv_projection_matches_full_projection_and_owns_disjoint_storage(rows, chunk):
    torch.manual_seed(rows)
    linear = torch.nn.Linear(16, 3 * 2 * 8, bias=False, dtype=torch.bfloat16).eval()
    x = torch.randn(rows, 16, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = tuple(p.view(rows, 2, 8).clone() for p in linear(x).split(16, -1))
        projected = streamed.vdn_project_qkv(linear, x, 2, 8, chunk)
    for actual, want in zip(projected, expected):
        torch.testing.assert_close(actual, want, atol=0, rtol=0)
    assert len({t.untyped_storage().data_ptr() for t in projected}) == 3


def test_vdn_raw_qkv_projection_failed_chunk_does_not_retain_projection_temporary():
    import weakref
    roots = []
    calls = 0

    def projection(x):
        nonlocal calls
        calls += 1
        tmp = x.new_zeros((len(x), 384))
        roots.append(weakref.ref(tmp))
        if calls == 2:
            raise RuntimeError("projection failed")
        return tmp

    with pytest.raises(RuntimeError, match="projection failed"):
        streamed.vdn_project_qkv(projection, torch.empty(129, 128), 1, 128, 64)
    assert all(ref() is None for ref in roots)
