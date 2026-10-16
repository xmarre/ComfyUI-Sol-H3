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


@pytest.mark.skipif(not os.environ.get("VDN_PATH"), reason="set VDN_PATH for composed grouped execution")
@pytest.mark.parametrize("mixed_grid", [False, True])
def test_real_flow_vdn_sol_grouped_binding_executes_and_releases_with_raw_linear(monkeypatch, mixed_grid):
    """Exercise the published Flow callback through VDN and Sol's real scope.

    CPU replaces CUDA production and the attention kernel only. Row planning,
    block-option transport, VDN raw-linear lifetime, dispatch and Sol scoping are
    the actual implementations. This is not a GPU arithmetic/performance test.
    """
    for name in ("COMFYUI_PATH", "FLOW_PATH", "VDN_PATH"):
        sys.path.insert(0, os.environ[name])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.quant_ops
    from torch import nn
    from vdn_h3 import partitioned_linear, partitioned_runtime, softmax_provider
    from vdn_h3.hybrid import VDNLayout, VDNState
    from vdn_h3.partitioned_sequence import PartitionedSequence
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.partitioned_prefix import PartitionedExactPrefixPlan
    from h3_flow_regenerate.partitioned_stage import PartitionedStagePlan, PartitionedStageRuntime
    from h3_flow_regenerate.partitioned_transformer import (
        _partitioned_transformer_options, _stage_partitioned_attention_override,
    )
    from sol_h3 import grouped_kv, partitioned_request as request
    from sol_h3.contracts import Config
    from sol_h3.runtime import Request, _REQUEST, _FORWARD

    torch.manual_seed(4141)
    plan = PartitionedSequence(3, 5, 2, 2, 3, 3 if mixed_grid else 2, 4 if mixed_grid else 3)
    target_h, target_w = plan.target_grid_h * 2, plan.target_grid_w * 2
    prefix = torch.zeros(1, 24, 2, target_h, target_w)
    metrics = H3FlowMetrics()
    stage = PartitionedStageRuntime(PartitionedStagePlan(prefix, 5, 4, 6, prefix.clone()), metrics)
    cfg = {"radius": 1, "chunk": 1, "anchor_frames": "none",
           "enable_softmax_gate": False, "linear_enabled": True}
    branch = SimpleNamespace(enable_text_state=True)
    vdn = VDNState("real-flow-grouped", cfg, [branch], 1, 128, retain_buffers=True)
    layout = VDNLayout(3, 33, 5, 6, (2, 3), 0, 1, 33, 1, 1, "none")
    values = {
        "state": vdn, "base_branch": branch,
        "qkv_proj": nn.Linear(8, 384, bias=False).requires_grad_(False),
        "out_proj": nn.Linear(128, 8, bias=False).requires_grad_(False),
        "q_norm": nn.RMSNorm(128), "k_norm": nn.RMSNorm(128),
        "heads": 1, "head_dim": 128, "block_index": 0, "cfg": cfg,
    }
    vdn.weights_on = lambda *_a, **_kw: {"to_out_linear.weight": torch.ones(8, 128) * 0.01}
    x = torch.randn(plan.sequence_rows, 8)
    raw_calls, kernel_scopes = [], []

    def linear(_branch, _weights, _xv, q, k, v, **kwargs):
        raw_calls.append(tuple(t.clone() for t in (q, k, v)))
        return (q + k + v).reshape(len(q), 128)

    monkeypatch.setattr(partitioned_linear, "partitioned_linear_readout", linear)
    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", lambda *_a, **_kw: None)
    monkeypatch.setattr(request, "_validate_thd", lambda *_a: None)
    monkeypatch.setattr(request, "device_descriptor", lambda _state, descriptor, device:
                        torch.tensor(descriptor.intervals, dtype=torch.int32, device=device))

    def key_bias(_state, *, kv_rows, prefix_range, log_measure, device, **_kwargs):
        if prefix_range is None or log_measure == 0:
            return None
        bias = torch.zeros(kv_rows, device=device)
        bias[prefix_range[0]:prefix_range[1]] = log_measure
        return bias
    monkeypatch.setattr(request, "_key_bias", key_bias)

    def kernel(q, k, v, *, key_bias, **kwargs):
        prepared = grouped_kv.current(k.unsqueeze(0), v.unsqueeze(0))
        kernel_scopes.append(prepared is not None)
        return request._weighted_dense(q, k, v, key_bias, scale=128**-0.5)
    monkeypatch.setattr(request, "_sm120_union", kernel)
    monkeypatch.setattr(grouped_kv, "_can_fuse", lambda *_a, **kw: not _a[-1])

    def produce(k, v, ko, vo, index, sinks):
        torch.index_select(k, 0, index, out=ko[sinks:])
        torch.index_select(v, 0, index, out=vo[sinks:])
        return SimpleNamespace(k=ko.unsqueeze(0), v=vo.unsqueeze(0), check=lambda *_a: None)
    monkeypatch.setattr(grouped_kv, "_produce", produce)

    sol = Request(Config(backend="sol", exact=False, dense_evaluations=0, dense_layers=0))
    override = _stage_partitioned_attention_override(stage, None, metrics)
    flow_plan = PartitionedExactPrefixPlan(3, 5, 2, 2, 3, plan.target_grid_h, plan.target_grid_w,
                                          same_grid_control=not mixed_grid)
    options = {"optimized_attention_override": override, "sol_h3_runtime_v1": sol.config.metadata()}
    bound = _partitioned_transformer_options(
        options, SimpleNamespace(seq_len=plan.sequence_rows), flow_plan.to_contract(),
        metrics, runtime=stage, block_index=0,
    )
    provider = bound[softmax_provider.PARTITIONED_PROVIDER_KEY]
    assert provider.grouped_kv_v1 is grouped_kv.execute
    assert provider._vdn_readonly_query_contract_v1 is True

    def evaluate(options):
        with vdn.runtime.execution(), torch.no_grad():
            token = vdn._layout.set(layout)
            rt = _REQUEST.set(sol)
            ft = _FORWARD.set((None, sol, 2, set(), []))
            try:
                return partitioned_runtime._partitioned_vdn_forward(
                    None, values, x, torch.zeros(1, plan.sequence_rows, 1, 1), options,
                )
            finally:
                _FORWARD.reset(ft)
                _REQUEST.reset(rt)
                vdn._layout.reset(token)

    baseline_options = dict(bound)
    baseline_options[softmax_provider.PARTITIONED_PROVIDER_KEY] = override.partitioned_attention_v1
    baseline = evaluate(baseline_options)
    assert sol.grouped_kv_fused_calls == 0
    kernel_scopes.clear()
    result = evaluate(bound)
    assert sol.grouped_kv_fused_calls == (2 if mixed_grid else 5)
    assert any(kernel_scopes)
    assert len(raw_calls) == 2
    for original, actual in zip(raw_calls[0], raw_calls[1]):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
    torch.testing.assert_close(result, baseline, atol=0, rtol=0)
    assert grouped_kv._ACTIVE.get() is None
    assert _REQUEST.get() is None and _FORWARD.get() is None
