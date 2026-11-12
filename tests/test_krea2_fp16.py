"""CPU structural tests for the native-FP16 ComfyUI Kitchen Krea route."""
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn.functional as F

from sol_h3 import krea2, krea2_fp16
from sol_h3.sparse import KernelUnavailable


class FakeCuda(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")


def fp16_qkv(tokens=1024, heads=2):
    return torch.randn(1, heads, tokens, 128, dtype=torch.float16).as_subclass(FakeCuda)


def fake_kernel(calls, bad_gate=False, small_error=0.0):
    def sol_attn(q, k, v, **kwargs):
        calls.append((q.shape, kwargs))
        if q.shape[1] <= krea2_fp16.GATE_ROWS:
            if bad_gate:
                return torch.ones_like(q)
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            ).transpose(1, 2) * (1 + small_error)
        return torch.zeros_like(q)
    return SimpleNamespace(sol_attn=sol_attn)


def test_fp16_uses_exact_text_sinks_and_256_extra_tokens(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "comfy_kitchen", fake_kernel(calls))
    monkeypatch.setattr(krea2_fp16, "available", lambda device: True)
    q = fp16_qkv()
    settings = krea2.Settings(dense_evaluations=0, extra_tokens=256)
    state = krea2._Run(owner=object(), settings=settings)
    result = krea2_fp16.attention(q, q, q, 101, settings, state)

    assert result.shape == (1, 1024, 2 * 128)
    assert result.dtype == torch.float16
    torch.testing.assert_close(
        result[:, :101].reshape(1, 101, 2, 128),
        F.scaled_dot_product_attention(
            q[:, :, :101], q, q,
        ).transpose(1, 2),
        atol=0, rtol=0,
    )
    assert state.sparse_calls == 1
    assert len(state.gates) == 1
    assert state.gates[0]["gate"] == "all_selected_fp16_pilot"
    assert len(calls) == 2
    pilot, actual = calls
    assert pilot[1]["sink_blocks"] == [0, 8]
    assert pilot[1]["sink_q"] == [0, 8]
    assert pilot[1]["token_aug"] == 0
    assert actual[1]["sink_blocks"] == [0, 2]
    assert actual[1]["sink_q"] == [0, 2]
    assert actual[1]["token_aug"] == 256
    assert actual[1]["tau"] == 1.0
    assert state.sparse_verified

    krea2_fp16.attention(q, q, q, 101, settings, state)
    assert len(calls) == 3  # The arithmetic pilot is per-run/per-layout.


def test_bad_fp16_arithmetic_fails_closed(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "comfy_kitchen", fake_kernel(calls, bad_gate=True))
    monkeypatch.setattr(krea2_fp16, "available", lambda device: True)
    q = fp16_qkv()
    settings = krea2.Settings()
    state = krea2._Run(object(), settings)
    with pytest.raises(KernelUnavailable, match="arithmetic pilot failed"):
        krea2_fp16.attention(q, q, q, 65, settings, state)
    assert state.kernel_failure
    assert state.sparse_calls == 0
    assert len(calls) == 1


def test_fp16_backend_not_available_avoids_gpu_call(monkeypatch):
    monkeypatch.setattr(krea2_fp16, "available", lambda device: False)
    q = fp16_qkv(16)
    settings = krea2.Settings()
    state = krea2._Run(object(), settings)
    with pytest.raises(KernelUnavailable, match="unavailable"):
        krea2_fp16.attention(q, q, q, 8, settings, state)
    assert state.sparse_calls == 0
    assert state.kernel_failure is not None
    # A rejected capability remains a dense fallback for this request.
    with pytest.raises(KernelUnavailable):
        krea2_fp16.attention(q, q, q, 8, settings, state)



def test_fp16_pilot_accepts_upstream_supported_small_relative_error(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "comfy_kitchen", fake_kernel(calls, small_error=0.01))
    monkeypatch.setattr(krea2_fp16, "available", lambda device: True)
    q = fp16_qkv()
    settings = krea2.Settings()
    state = krea2._Run(object(), settings)
    krea2_fp16.attention(q, q, q, 65, settings, state)
    assert len(state.gates) == 1
    assert 0 < state.gates[0]["rel_l2"] < krea2_fp16.FP16_PILOT_REL_L2_LIMIT
    assert state.sparse_calls == 1
