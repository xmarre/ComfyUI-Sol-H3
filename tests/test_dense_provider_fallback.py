from types import SimpleNamespace

import torch

from sol_h3 import runtime
from sol_h3.contracts import Config
from sol_h3.runtime import BlockPatch, Request, _FORWARD


def test_preprocessing_survives_unavailable_leaf_demotion(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=99, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=None))])
    q = torch.ones(1, 1, 7, 128, dtype=torch.bfloat16)
    transforms = []
    originals = []

    def broken_leaf(original, q, k, v, heads, **kw):
        raise ImportError("libstdc++.so.6: version `GLIBCXX_3.4.32' not found")

    def transform(q, k, v, heads, **kw):
        transforms.append(True)
        return q + 1, k + 2, v + 3

    def wrapper(original, q, k, v, heads, **kw):
        q, k, v = transform(q, k, v, heads, **kw)
        return broken_leaf(original, q, k, v, heads, **kw)

    wrapper.attention_preprocess_v1 = (transform, broken_leaf)

    def original(q, k, v, heads, **kw):
        originals.append((q.clone(), k.clone(), v.clone()))
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)

    monkeypatch.setattr(runtime, "_shape_reason", lambda *args, **kwargs: None)
    options = {"optimized_attention_override": wrapper}

    def block(args):
        to = args["transformer_options"]
        return {"img": to["optimized_attention_override"](
            original, q, q, q, 1, skip_reshape=True, transformer_options=to
        )}

    for evaluation in range(2):
        token = _FORWARD.set((model, state, evaluation, set(), []))
        try:
            BlockPatch(0, cfg)(
                {"img": torch.zeros(7, 128),
                 "layout": SimpleNamespace(seq_len=7, segments=[(0, 3, "text"), (3, 7, "video")]),
                 "transformer_options": options},
                {"original_block": block},
            )
        finally:
            _FORWARD.reset(token)

    assert transforms == [True, True]
    assert len(originals) == 2
    for q_seen, k_seen, v_seen in originals:
        assert torch.equal(q_seen, q + 1)
        assert torch.equal(k_seen, q + 2)
        assert torch.equal(v_seen, q + 3)
    assert state.disabled_dense_providers == {id(broken_leaf)}
    assert state.fallbacks["dense_provider_unavailable:binary_abi"] == 1
