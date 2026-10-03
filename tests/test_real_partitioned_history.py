"""Recognize actual Flow closures with native Core layouts, without GPU inference."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH") or not os.environ.get("FLOW_PATH"),
    reason="set COMFYUI_PATH and FLOW_PATH for native partitioned history contracts",
)


@pytest.mark.parametrize("exact", [False, True])
def test_real_equal_grid_flow_closures_are_stable_and_preserve_history_boundaries(exact):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["FLOW_PATH"])
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.partitioned_stage import (
        PARTITIONED_STAGE_KEY, PartitionedStagePlan, PartitionedStageRuntime,
    )
    from h3_flow_regenerate.partitioned_transformer import (
        _partitioned_transformer_options, partitioned_diffusion_wrapper,
    )
    from sol_h3 import interop
    from sol_h3.contracts import Config
    from sol_h3.partitioned_history import (
        _partitioned_history_layout_valid, install_partitioned_history_bridge,
    )
    from sol_h3.runtime import BlockPatch, Request, _REQUEST

    install_partitioned_history_bridge()
    cfg = Config(exact=exact, backend="sol", dense_evaluations=1, dense_layers=0)
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda *a, **kw: None)) for _ in range(2)],
        dtype=torch.bfloat16,
    )
    video = torch.zeros((1, 24, 5, 8, 12))
    audio = torch.zeros((1, 32, 2, 3))
    prefix = video[:, :, :2].clone()
    plan = PartitionedStagePlan(
        prefix=prefix, temporal=5, source_h=8, source_w=12, prefix_noise=prefix.clone(),
    )
    metrics = H3FlowMetrics()
    stage = PartitionedStageRuntime(plan=plan, metrics=metrics)
    replacements = {("double_block", i): BlockPatch(i, cfg) for i in range(2)}
    options = {PARTITIONED_STAGE_KEY: stage, "patches_replace": {"dit": replacements}}
    context = torch.zeros((1, 2, 128), dtype=torch.bfloat16)
    identities, wrappers = [], []
    request = Request(cfg)

    class Executor:
        class_obj = model

        def __call__(self, x, _timestep, _context, local, minimax_payload=None, **kw):
            patch = local["patches_replace"]["dit"][("double_block", 0)]
            wrappers.append(patch)
            values = interop._closure_values(patch)
            packed = values["partitioned_layout"]
            bound = _partitioned_transformer_options(
                local, packed, values["partition_contract"], metrics, runtime=stage, block_index=0,
            )
            assert _partitioned_history_layout_valid(bound, packed)
            identity = interop.HistoryPolicy(cfg)(layout=packed, options=bound, model=model)
            assert identity is not None
            identities.append(identity)
            stale = SimpleNamespace(seq_len=packed.seq_len - 1, segments=packed.segments, signature=packed.signature)
            assert not _partitioned_history_layout_valid(bound, stale)
            return [torch.zeros_like(x[0]), torch.zeros_like(x[1])]

    def run():
        return partitioned_diffusion_wrapper(
            Executor(), [video, audio], torch.tensor([500.0]), context, options, minimax_payload={},
        )

    token = _REQUEST.set(request)
    try:
        run()
        run()
        assert wrappers[0] is not wrappers[1]
        assert identities[0] == identities[1]
        request.evaluations = 1
        run()
        assert identities[2] != identities[1]  # Dense-to-Sol remains a history boundary.
        assert identities[2][1] == "sol"
        run()
        assert identities[3] == identities[2]
    finally:
        _REQUEST.reset(token)
    assert options["patches_replace"]["dit"] is replacements
    assert all(isinstance(patch, BlockPatch) for patch in replacements.values())
    assert _REQUEST.get() is None
