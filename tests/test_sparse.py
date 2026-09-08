from types import ModuleType
import sys

import pytest
import torch
import torch.nn.functional as F

from sol_h3 import sparse
from sol_h3.contracts import Config
from sol_h3.runtime import Request


def test_real_loader_rejects_cpu():
    with pytest.raises(RuntimeError, match="SM120"):
        sparse.load_kernel(torch.device("cpu"))


def test_comfy_kitchen_loader_requires_available_api(monkeypatch):
    fake = ModuleType("comfy_kitchen")
    fake.sol_attn_is_available = lambda device: False
    fake.sol_attn = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "comfy_kitchen", fake)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    with pytest.raises(RuntimeError, match="no compiled sol_attn"):
        sparse.load_kernel(torch.device("cuda"))


def test_comfy_kitchen_loader_maps_prefix_rows_to_exact_blocks(monkeypatch):
    calls = []
    fake = ModuleType("comfy_kitchen")
    fake.sol_attn_is_available = lambda device: True

    def sol_attn(q, k, v, **kw):
        calls.append(kw)
        return v

    fake.sol_attn = sol_attn
    monkeypatch.setitem(sys.modules, "comfy_kitchen", fake)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    kernel = sparse.load_kernel(torch.device("cuda"))
    q = torch.zeros(1, 130, 2, 128)
    kernel(q, q, q, tau=1.0, sink_start=0, sink_tokens=65)
    assert calls == [{
        "tau": 1.0,
        "scale": None,
        "sink_blocks": [0, 2],
        "sink_q": [0, 0],
        "topk_ratio": 0.0,
        "tail": True,
        "token_aug": 0,
    }]
    assert kernel.backend_name == "comfy_kitchen.sol_attn"
    assert kernel.block_size == 64


def test_sink_geometry_rejects_nonprefix_and_out_of_range():
    assert sparse._sink_blocks(0, 0, 130) == [0, 0]
    assert sparse._sink_blocks(0, 64, 130) == [0, 1]
    assert sparse._sink_blocks(0, 65, 130) == [0, 2]
    with pytest.raises(RuntimeError, match="beginning at row zero"):
        sparse._sink_blocks(64, 64, 130)
    with pytest.raises(RuntimeError, match="outside"):
        sparse._sink_blocks(0, 131, 130)


def test_bridge_protects_prefix_and_uses_full_sink_gate(monkeypatch):
    # CPU bridge oracle only: deliberately fake the sparse output after the
    # all-selected call to verify that ALL prefix queries are overwritten.
    calls = []
    dense_calls = []

    def kernel(q, k, v, **kw):
        calls.append(kw)
        dense = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return dense if kw["sink_tokens"] == q.shape[1] else torch.full_like(dense, 9)

    def dense_attention(q, k, v):
        dense_calls.append(q.shape[2])
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    q, k, v = (torch.randn(1, 2, 11, 128).to(torch.bfloat16) for _ in range(3))
    config = Config(exact=False, backend="sol")
    state = Request(config)
    out = sparse.attention(q, k, v, 5, config, state, dense_attention=dense_attention).reshape(1, 11, 2, 128)
    want = F.scaled_dot_product_attention(q[:, :, :5], k, v).transpose(1, 2)
    assert torch.equal(out[:, :5], want)
    assert (out[:, 5:] == 9).all()
    assert [c["sink_tokens"] for c in calls] == [11, 5]
    assert all(c["sink_start"] == 0 for c in calls)
    assert dense_calls == [5]
    assert state.sparse_calls == 1
    sparse.attention(q, k, v, 5, config, state, dense_attention=dense_attention)
    assert len(calls) == 3  # same-request shape gate reused
    assert dense_calls == [5, 5]


def test_failed_arithmetic_never_counts_sparse(monkeypatch):
    monkeypatch.setattr(
        sparse,
        "load_kernel",
        lambda device: lambda q, k, v, **kw: torch.full_like(v, float("nan")),
    )
    state = Request(Config(exact=False, backend="sol"))
    q = torch.ones(1, 1, 3, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="arithmetic gate failed"):
        sparse.attention(q, q, q, 1, state.config, state)
    assert state.sparse_calls == 0
