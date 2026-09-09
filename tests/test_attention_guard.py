from types import SimpleNamespace

import torch

from sol_h3.contracts import Config
from sol_h3.runtime import BlockPatch, Request, _FORWARD


def test_sparse_dense_reference_keeps_attention_recursion_guard(monkeypatch):
    """Raw dense providers may call wrapped fallbacks; they must stay inside wrap_attn."""
    from sol_h3 import sparse, runtime

    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None))])
    state = Request(cfg)
    q = torch.zeros(1, 1, 7, 128, dtype=torch.bfloat16)
    layout = SimpleNamespace(
        seq_len=7,
        segments=[(0, 2, "text"), (2, 3, "audio"), (3, 7, "video")],
    )
    observed = []

    def fake_sparse(q, k, v, prefix, config, request, dense_attention=None):
        assert prefix == 3
        out = dense_attention(q, k, v)
        request.sparse_calls += 1
        return out.reshape(1, q.shape[2], -1)

    def raw_dense(q, k, v, heads, mask=None, **kwargs):
        observed.append(kwargs.get("_inside_attn_wrapper"))
        assert kwargs["skip_reshape"] is True
        assert kwargs["skip_output_reshape"] is True
        return q

    monkeypatch.setattr(sparse, "attention", fake_sparse)
    token = _FORWARD.set((model, state, 0, set(), []))
    try:

        def original_block(forwarded):
            override = forwarded["transformer_options"]["optimized_attention_override"]
            out = override(raw_dense, q, q, q, 1, skip_reshape=True, _inside_attn_wrapper=True)
            assert out.shape == (1, 7, 128)
            return {"img": forwarded["img"]}

        BlockPatch(0, cfg)(
            {"img": torch.zeros(7, 128), "transformer_options": {}, "layout": layout},
            {"original_block": original_block},
        )
    finally:
        _FORWARD.reset(token)

    assert observed == [True]
    assert state.sparse_calls == 1


def test_sparse_rectangular_prefix_recompute_uses_reduced_kv(monkeypatch):
    from sol_h3 import sparse

    class Kernel:
        backend_name = "test"
        source_tree_verified = True

        def __call__(self, q, k, v, **kwargs):
            return torch.zeros_like(q)

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    state.kernel = Kernel()
    state.kernel_device = torch.device("cpu")
    state.sparse_verified.add(
        (
            torch.device("cpu"),
            torch.bfloat16,
            (
                ((1, 9, 1, 128), (1152, 128, 128, 1)),
                ((1, 7, 1, 128), (896, 128, 128, 1)),
                ((1, 7, 1, 128), (896, 128, 128, 1)),
            ),
        )
    )
    q = torch.zeros(1, 1, 9, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 7, 128, dtype=torch.bfloat16)
    v = torch.zeros(1, 1, 7, 128, dtype=torch.bfloat16)
    observed = []

    def dense(qd, kd, vd):
        observed.append((qd.shape[2], kd.shape[2], vd.shape[2]))
        return torch.ones(1, qd.shape[2], qd.shape[1], qd.shape[3], dtype=qd.dtype)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: state.kernel)
    out = sparse.attention(q, k, v, 3, cfg, state, dense_attention=dense)
    assert out.shape == (1, 9, 128)
    assert observed == [(3, 7, 7)]



def test_history_policy_accepts_external_mixed_measure_receipts():
    from sol_h3.interop import HistoryPolicy

    policy = HistoryPolicy(object())
    assert policy.accept_receipts([("sol_h3", 0, "sol_external_mixed_measure")])
    assert policy.accept_receipts([
        ("sol_h3", 0, "dense_warmup"),
        ("sol_h3", 1, "sol_external_mixed_measure"),
    ])
    assert not policy.accept_receipts([("sol_h3", 0, "unknown_measure_route")])
