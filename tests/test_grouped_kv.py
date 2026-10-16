"""Grouped producer geometry, lifetime and real-kernel acceptance contracts."""

from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sol_h3 import grouped_kv as group
from sol_h3.contracts import Config
from sol_h3.runtime import Request, _FORWARD, _REQUEST


def inputs(sinks=3, windows=7):
    k, v = torch.randn(31, 2, 128), torch.randn(31, 2, 128)
    ko, vo = torch.empty(sinks + windows, 2, 128), torch.empty(sinks + windows, 2, 128)
    ko[:sinks].copy_(k[:sinks])
    vo[:sinks].copy_(v[:sinks])
    return k, v, ko, vo, torch.randperm(31)[:windows], sinks


def test_fallback_preserves_noncontiguous_order_and_prefix():
    values = inputs()
    k, v, ko, vo, index, sinks = values

    def consumer(kg, vg):
        torch.testing.assert_close(kg, torch.cat((k[:sinks], k[index])), atol=0, rtol=0)
        torch.testing.assert_close(vg, torch.cat((v[:sinks], v[index])), atol=0, rtol=0)
        assert group.current(kg.unsqueeze(0), vg.unsqueeze(0)) is None
        return kg + vg

    torch.testing.assert_close(group.execute(*values, consumer), ko + vo)


def test_invalid_geometry_fails_before_consumer():
    values = list(inputs())
    values[-1] += 1
    with pytest.raises(RuntimeError, match="geometry"):
        group.execute(*values, lambda *args: pytest.fail("consumer entered"))


def test_scopes_restore_on_nested_calls_and_failure(monkeypatch):
    state = Request(Config(backend="sol"))
    rt = _REQUEST.set(state)
    ft = _FORWARD.set((None, state, 2, set(), []))
    monkeypatch.setattr(group, "_can_fuse", lambda *args: True)

    def produce(k, v, ko, vo, idx, sinks):
        torch.index_select(k, 0, idx, out=ko[sinks:])
        torch.index_select(v, 0, idx, out=vo[sinks:])
        return SimpleNamespace(k=ko.unsqueeze(0), v=vo.unsqueeze(0), check=lambda *args: None)

    monkeypatch.setattr(group, "_produce", produce)
    outer, inner = inputs(), inputs(1, 13)

    def outside(k, v):
        parent = group.current(k.unsqueeze(0), v.unsqueeze(0))
        assert parent is not None

        def fail(ki, vi):
            assert group.current(ki.unsqueeze(0), vi.unsqueeze(0)) is not parent
            raise ValueError("consumer failed")

        with pytest.raises(ValueError, match="consumer failed"):
            group.execute(*inner, fail)
        assert group.current(k.unsqueeze(0), v.unsqueeze(0)) is parent
        with pytest.raises(RuntimeError, match="recursively"):
            group.execute(*outer, lambda *args: None)
        nested = _REQUEST.set(Request(Config()))
        try:
            assert group.current(k.unsqueeze(0), v.unsqueeze(0)) is None
        finally:
            _REQUEST.reset(nested)
        raise RuntimeError("outer failed")

    try:
        with pytest.raises(RuntimeError, match="outer failed"):
            group.execute(*outer, outside)
        assert group._ACTIVE.get() is None
        assert state.grouped_kv_fused_calls == 2
    finally:
        _FORWARD.reset(ft)
        _REQUEST.reset(rt)


def test_prepared_summary_views_check_mutation_and_singleton_batch_stride(monkeypatch):
    from sol_h3._vendor.sol_attn import interface

    monkeypatch.setattr(interface, "_validate_inputs", lambda *args: (12, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: SimpleNamespace(cuda_stream=4))
    k, v = torch.randn(17, 2, 128), torch.randn(17, 2, 128)
    summaries = (torch.zeros(1, 1, 2, 128), torch.zeros(1, 1, 2, 128))
    prepared = interface.PreparedKV(k.unsqueeze(0), v.unsqueeze(0), summaries=summaries)
    kb = k.transpose(0, 1).unsqueeze(0).transpose(1, 2)
    vb = v.transpose(0, 1).unsqueeze(0).transpose(1, 2)
    prepared.check(kb, vb)
    with pytest.raises(ValueError, match="does not belong"):
        prepared.check(kb.clone(), vb)
    k.add_(1)
    with pytest.raises(ValueError, match="modified"):
        prepared.check(kb, vb)


def test_actual_triton_gather_in_cpu_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1", "OMP_NUM_THREADS": "1"}
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "tools/check_grouped_kv_interpreter.py")], cwd=root, env=env, check=True, timeout=60
    )


def test_gather_compiles_to_sm120_without_gpu():
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from sol_h3.grouped_kv_triton import _gather_summaries

    signature = {name: "*bf16" for name in ("K", "V", "KO", "VO", "KC", "VC")}
    signature.update(Index="*i64", Rows="i32", Sinks="i32")
    source = ASTSource(
        _gather_summaries, signature, constexprs=dict(KRow=7168, KHead=128, VRow=7168, VHead=128, Heads=56, Blocks=256)
    )
    result = triton.compile(source, target=GPUTarget("cuda", 120, 32), options={"num_warps": 4, "num_stages": 1})
    assert ".target sm_120" in result.asm["ptx"]
    assert result.metadata.global_scratch_size == 0


@pytest.mark.gpu
def test_real_sm120_grouped_bf16_acceptance():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("requires SM120/SM121")
    from tools.grouped_attention_acceptance import parity_cases

    with torch.inference_mode():
        parity_cases()
