from types import SimpleNamespace

import torch

from sol_h3.contracts import Config
from sol_h3.interop import HistoryPolicy
from sol_h3.runtime import BlockPatch, Request, _REQUEST


def test_dense_provider_demotion_changes_history_identity():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)

    def provider(*args, **kwargs):
        return None

    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda *args, **kwargs: None))],
        dtype=torch.bfloat16,
    )
    options = {
        "optimized_attention_override": provider,
        "patches_replace": {"dit": {("double_block", 0): BlockPatch(0, cfg)}},
    }
    layout = SimpleNamespace(
        seq_len=7,
        segments=[(0, 3, "text"), (3, 7, "video")],
        signature=(3, 7),
    )
    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        policy = HistoryPolicy(cfg)
        initial = policy(layout=layout, options=options, model=model)
        request.disabled_dense_providers.add(id(provider))
        demoted = policy(layout=layout, options=options, model=model)
    finally:
        _REQUEST.reset(token)

    assert initial is not None and demoted is not None
    assert demoted != initial
    assert demoted[-2][-1] == "unavailable_fallback"
