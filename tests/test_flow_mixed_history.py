"""Real Flow replacement-chain identities without executing a GPU attention backend."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH") or not os.environ.get("FLOW_PATH"),
    reason="set COMFYUI_PATH and FLOW_PATH for real Flow source contracts",
)


def _model(count=3):
    return SimpleNamespace(
        blocks=[
            SimpleNamespace(attn=SimpleNamespace(forward=lambda *a, **k: None))
            for _ in range(count)
        ],
        dtype=torch.bfloat16,
    )


def _native_layout():
    return SimpleNamespace(
        seq_len=16,
        segments=((0, 4, "text"), (4, 16, "video")),
        signature=(4, 3, 4, 4, 1),
    )


def test_real_flow_marked_layout_wrapper_is_forecast_provable_and_stable():
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["FLOW_PATH"])

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from h3_flow_regenerate.attention import make_layout_block_wrapper, mark_layout_wrapper
    from sol_h3.contracts import Config
    from sol_h3.interop import HistoryPolicy
    from sol_h3.runtime import BlockPatch, Request, _REQUEST

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    model = _model()
    metrics = SimpleNamespace()

    def options():
        replacements = {
            ("double_block", index): BlockPatch(index, cfg)
            for index in range(len(model.blocks))
        }
        previous = replacements[("double_block", 0)]
        wrapper = make_layout_block_wrapper(0, metrics, previous)
        replacements[("double_block", 0)] = mark_layout_wrapper(
            wrapper,
            metrics=metrics,
            previous=previous,
            scope="layout",
        )
        return {"patches_replace": {"dit": replacements}}

    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        first = HistoryPolicy(cfg)(layout=_native_layout(), options=options(), model=model)
        second = HistoryPolicy(cfg)(layout=_native_layout(), options=options(), model=model)
    finally:
        _REQUEST.reset(token)

    # Flow can rebuild the marked wrapper while its semantics remain identical.
    assert first is not None and second is not None
    assert first == second
    assert first[6][0][0][0] == "h3_flow_layout_wrapper_v1"


def test_real_flow_marked_layout_wrapper_fails_closed_when_marker_link_disagrees():
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["FLOW_PATH"])

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from h3_flow_regenerate.attention import make_layout_block_wrapper, mark_layout_wrapper
    from sol_h3.contracts import Config
    from sol_h3.interop import HistoryPolicy
    from sol_h3.runtime import BlockPatch, Request, _REQUEST

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    model = _model()
    metrics = SimpleNamespace()
    previous = BlockPatch(0, cfg)
    wrapper = mark_layout_wrapper(
        make_layout_block_wrapper(0, metrics, previous),
        metrics=metrics,
        previous=previous,
        scope="layout",
    )
    wrapper._h3_flow_previous = object()
    replacements = {
        ("double_block", 0): wrapper,
        ("double_block", 1): BlockPatch(1, cfg),
        ("double_block", 2): BlockPatch(2, cfg),
    }

    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        identity = HistoryPolicy(cfg)(
            layout=_native_layout(),
            options={"patches_replace": {"dit": replacements}},
            model=model,
        )
    finally:
        _REQUEST.reset(token)

    assert identity is None


def test_real_flow_mixed_grid_rebuilt_closures_are_forecast_provable():
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["FLOW_PATH"])

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from h3_flow_regenerate.attention import make_layout_block_wrapper, mark_layout_wrapper
    from h3_flow_regenerate.mixed_grid import MixedGridPlan, mixed_diffusion_wrapper
    from sol_h3.contracts import Config
    from sol_h3.interop import HistoryPolicy
    from sol_h3.runtime import BlockPatch, Request, _REQUEST

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    model = _model()

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
    metrics = SimpleNamespace()
    replacements = {
        ("double_block", index): BlockPatch(index, cfg)
        for index in range(len(model.blocks))
    }
    previous = replacements[("double_block", 0)]
    replacements[("double_block", 0)] = mark_layout_wrapper(
        make_layout_block_wrapper(0, metrics, previous),
        metrics=metrics,
        previous=previous,
        scope="layout",
    )
    options = {
        "h3_flow_mixed_grid_v1": {"plan": plan, "metrics": SimpleNamespace()},
        "patches_replace": {"dit": replacements},
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

    # The real Flow mixed wrapper recreates all block closures on each call, while
    # block 0 retains Flow's generic marked layout wrapper underneath it. The full
    # production replacement chain must remain semantic/stable across rebuilds.
    assert len(identities) == 2 and identities[0] == identities[1]


def test_real_flow_high_continuation_does_not_restart_dense_evaluation_warmup():
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["FLOW_PATH"])

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from h3_flow_regenerate.runtime import _flow_stage_contract, _high_stage_contract
    from sol_h3.contracts import Config
    from sol_h3.interop import HistoryPolicy, dense_evaluation_warmup
    from sol_h3.runtime import BlockPatch, Request, _REQUEST

    cfg = Config(exact=False, backend="sol", dense_evaluations=1, dense_layers=0)
    model = _model()
    replacements = {
        ("double_block", index): BlockPatch(index, cfg)
        for index in range(len(model.blocks))
    }
    guider = SimpleNamespace(model_options={
        "transformer_options": {"patches_replace": {"dit": replacements}}
    })
    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        ordinary = guider.model_options["transformer_options"]
        ordinary_identity = HistoryPolicy(cfg)(layout=_native_layout(), options=ordinary, model=model)
        assert ordinary_identity is not None and ordinary_identity[1] == "dense"
        assert dense_evaluation_warmup(cfg, 0, ordinary)

        with _flow_stage_contract(guider, "high"), _high_stage_contract(guider):
            high = guider.model_options["transformer_options"]
            high_identity = HistoryPolicy(cfg)(layout=_native_layout(), options=high, model=model)
            assert high_identity is not None and high_identity[1] == "sol"
            assert not dense_evaluation_warmup(cfg, 0, high)

        # The one-call handoff probe deliberately remains a dense exact anchor.
        with _flow_stage_contract(guider, "probe"), _high_stage_contract(guider):
            probe = guider.model_options["transformer_options"]
            probe_identity = HistoryPolicy(cfg)(layout=_native_layout(), options=probe, model=model)
            assert probe_identity is not None and probe_identity[1] == "dense"
            assert dense_evaluation_warmup(cfg, 0, probe)
    finally:
        _REQUEST.reset(token)
