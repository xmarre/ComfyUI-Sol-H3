from types import SimpleNamespace

import torch

from sol_h3 import runtime, sparse
from sol_h3.contracts import Config
from sol_h3.interop import VDN_KEY_V3
from sol_h3.runtime import BlockPatch, Request, _FORWARD


def test_vdn_v3_consumes_direct_rectangular_q_without_square_payload(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None))])
    token = _FORWARD.set((model, state, 0, set(), []))
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    seen = []

    def fake_attention(q, k, v, prefix, config, state, **kwargs):
        seen.append((q.shape[2], k.shape[2], prefix, kwargs.get("recompute_prefix_queries")))
        state.sparse_calls += 1
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)

    monkeypatch.setattr(sparse, "attention", fake_attention)
    q = torch.zeros(2, 1, 128, dtype=torch.bfloat16)
    k = torch.zeros(5, 1, 128, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    native_calls = []

    try:
        def block(args):
            provider = args["transformer_options"][VDN_KEY_V3]
            got = provider(
                lambda: native_calls.append(True) or q,
                q, k, v, kind="local", scale=128 ** -0.5,
                sink_rows=1,
            )
            assert got.shape == q.shape
            return {"img": args["img"]}

        BlockPatch(0, cfg)(
            {"img": torch.zeros(7, 128), "layout": SimpleNamespace(), "transformer_options": {}},
            {"original_block": block},
        )
    finally:
        _FORWARD.reset(token)

    assert native_calls == []
    assert seen == [(2, 5, 1, False)]
    assert state.vdn_local_sol_calls == 1
    assert state.vdn_rectangular_sol_calls == 1
    assert state.vdn_requested_q_rows == state.vdn_kernel_q_rows == 2
