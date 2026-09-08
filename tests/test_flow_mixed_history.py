"""Real Flow mixed-grid closure identity without executing a GPU attention backend."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH") or not os.environ.get("FLOW_PATH"),
    reason="set COMFYUI_PATH and FLOW_PATH for real mixed-grid source contracts",
)


def test_real_flow_mixed_grid_rebuilt_closures_are_forecast_provable():
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["FLOW_PATH"])

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from h3_flow_regenerate.mixed_grid import MixedGridPlan, mixed_diffusion_wrapper
    from sol_h3.contracts import Config
    from sol_h3.interop import HistoryPolicy
    from sol_h3.runtime import BlockPatch, Request, _REQUEST

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    blocks = [
        SimpleNamespace(attn=SimpleNamespace(forward=lambda *a, **k: None))
        for _ in range(3)
    ]
    model = SimpleNamespace(blocks=blocks, dtype=torch.bfloat16)

    # 4-frame low-grid carrier (4x4) with a 2-frame exact target-grid prefix (8x8).
    # Only closure construction runs: the fake executor inspects the real Flow
    # replacements before any wrapped transformer block executes.
    prefix = torch.randn(1, 24, 2, 8, 8)
    plan = MixedGridPlan(
        prefix=prefix,
        temporal=4,
        source_h=4,
        source_w=4,
        prefix_noise=torch.randn_like(prefix),
    )
    video = torch.randn(1, 24, 4, 4, 4)
    audio = torch.randn(1, 32, 2, 3)
    context = torch.randn(1, 2, 128, dtype=torch.bfloat16)
    options = {
        "h3_flow_mixed_grid_v1": {"plan": plan, "metrics": SimpleNamespace()},
        "patches_replace": {
            "dit": {
                ("double_block", index): BlockPatch(index, cfg)
                for index in range(len(blocks))
            }
        },
    }

    identities = []

    class Executor:
        class_obj = model

        def __call__(self, x, timestep, inner_context, local, minimax_payload=None, **kwargs):
            layout = minimax_payload["layout"]
            identity = HistoryPolicy(cfg)(layout=layout, options=local, model=model)
            assert identity is not None
            identities.append(identity)
            return [torch.zeros_like(x[0]), torch.zeros_like(x[1])]

    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        for _ in range(2):
            out = mixed_diffusion_wrapper(
                Executor(),
                [video, audio],
                torch.tensor([500.0]),
                context,
                options,
                minimax_payload={},
            )
            assert tuple(out[0].shape) == tuple(video.shape)
            assert tuple(out[1].shape) == tuple(audio.shape)
    finally:
        _REQUEST.reset(token)

    # The real Flow wrapper recreates all block closures on each call. Their
    # backend-history identity must remain semantic/stable across those rebuilds.
    assert len(identities) == 2 and identities[0] == identities[1]
