"""Real Diff-Aid runtime-declaration timing through Spectrum + SOL + VDN + Untwist."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in (
        "COMFYUI_PATH", "SPECTRUM_PATH", "VDN_PATH", "UNTWIST_PATH", "DIFFAID_PATH"
    )),
    reason="set all native-interop dependency paths for the real Diff-Aid stack",
)


class _DiffAidTestPatcher:
    """Minimal ModelPatcher surface used by the real Diff-Aid H3 apply node."""

    def __init__(self, inner):
        self.inner = inner
        self.model_options = {"transformer_options": {}}

    def clone(self):
        cloned = type(self)(self.inner)
        cloned.model_options = copy.deepcopy(self.model_options)
        return cloned

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise KeyError(name)
        return self.inner

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def set_model_patch_replace(self, patch, namespace, block_kind, index):
        replacements = self.model_options.setdefault("transformer_options", {}).setdefault(
            "patches_replace", {}
        ).setdefault(namespace, {})
        replacements[(block_kind, index)] = patch


def _load_real_diffaid_package():
    root = Path(os.environ["DIFFAID_PATH"]).resolve()
    name = "_sol_h3_real_diffaid_test_package"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Diff-Aid package from {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.mark.parametrize("sol_outer", [False, True])
def test_real_diffaid_runtime_declaration_remains_forecast_safe(monkeypatch, sol_outer):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["SPECTRUM_PATH"])
    sys.path.insert(0, os.environ["VDN_PATH"])
    sys.path.insert(0, os.environ["UNTWIST_PATH"])

    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
    from comfy.patcher_extension import WrapperExecutor
    from comfyui_spectrum_h3.config import SpectrumH3Config
    from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
    from comfyui_spectrum_h3.minimax_h3 import diffusion_model_wrapper
    from comfyui_spectrum_h3.sampling import RUNTIME_KEY, RUN_ID_KEY, STEP_ID_KEY
    from flux_untwist.patches import make_minimax_h3_attention_override
    from types import SimpleNamespace
    from vdn_h3.hybrid import VDNState, VDNLayout, make_vdn_forward
    from sol_h3 import runtime, sparse
    from sol_h3.contracts import Config, KEY
    from sol_h3.interop import HISTORY_KEY, SPECTRUM_EXTERNAL_RUNTIME_KEY, HistoryPolicy
    from sol_h3.runtime import BlockPatch, SamplingWrapper, DiffusionWrapper, _REQUEST

    diffaid = _load_real_diffaid_package()
    diffaid_node = diffaid.NODE_CLASS_MAPPINGS["MiniMaxH3DiffAidSparsePatch"]()

    torch.manual_seed(61)
    model = MiniMaxH3Model(
        hidden_size=128,
        num_layers=3,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=256,
        latents_dim=2,
        audio_latents_dim=2,
        text_dim=128,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=4,
        rope_inv_freq_len=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        operations=torch.nn,
    ).eval().requires_grad_(False)
    model.rope.inv_freq.fill_(1.0)

    # Apply the real Diff-Aid node first, matching the production node order.
    # Its compatibility installer owns the per-call runtime declaration needed
    # to prove the activation-only replacement chain to SOL history preflight.
    patched, summary = diffaid_node.patch(
        _DiffAidTestPatcher(model),
        True,
        "1,3",
        0.5,
        0.0,
        0.95,
        0.0,
        "none",
        0.35,
        True,
    )
    assert "spectrum_h3_contract=v1:" in summary
    diffaid_wrapper = patched.model_options["model_function_wrapper"]
    diffaid_options = patched.model_options["transformer_options"]
    diffaid_replacements = diffaid_options["patches_replace"]["dit"]

    cfg = Config(exact=False, backend="sol", dense_evaluations=1, dense_layers=0)
    sol_replacements = {
        ("double_block", index): BlockPatch(
            index,
            cfg,
            diffaid_replacements.get(("double_block", index)),
        )
        for index in range(len(model.blocks))
    }
    opts = {
        **diffaid_options,
        KEY: cfg.metadata(),
        HISTORY_KEY: {"sol_h3": HistoryPolicy(cfg)},
        "patches_replace": {"dit": sol_replacements},
        "cond_or_uncond": [0],
        "uuids": ["positive"],
    }

    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)

    def oracle(q, k, v, **kw):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: oracle)

    frames = 4
    video = torch.randn(1, 2, frames, 4, 4)
    audio = torch.randn(1, 2, 2, 3)
    context = torch.randn(1, 2, 128, dtype=torch.bfloat16)
    packed = PackedLayout(2, frames, 4, 4, 3)

    vdn_cfg = {
        "radius": 0,
        "chunk": 1,
        "anchor_frames": "both",
        "enable_softmax_gate": True,
        "linear_enabled": False,
    }
    vdn_state = VDNState(
        "real-diffaid-stack-test",
        vdn_cfg,
        [SimpleNamespace(w={}, enable_text_state=False) for _ in model.blocks],
        1,
        128,
    )
    vdn_state.softmax_backend = "grouped"
    va, vb, _ = next(segment for segment in packed.segments if segment[2] == "video")
    aa, ab, _ = next(segment for segment in packed.segments if segment[2] == "audio")
    ta, tb, _ = next(segment for segment in packed.segments if segment[2] == "text")
    vdn_layout = VDNLayout(
        video_start=va,
        video_end=vb,
        audio_start=aa,
        audio_end=ab,
        num_frames=frames,
        tokens_per_frame=4,
        frame_size=(2, 2),
        text_start=ta,
        text_len=tb - ta,
        seq_len=packed.seq_len,
        radius=0,
        chunk=1,
        anchor_frames="both",
    )
    vdn_state._layout.set(vdn_layout)
    gate_weight = torch.randn(1, 128, dtype=torch.bfloat16) * 0.01
    gate_bias = torch.zeros(1, dtype=torch.bfloat16)
    monkeypatch.setattr(
        vdn_state,
        "weights_on",
        lambda *a: {
            "softmax_gate.up.weight": gate_weight,
            "softmax_gate.up.bias": gate_bias,
        },
    )
    for index, block in enumerate(model.blocks):
        monkeypatch.setattr(block.attn, "forward", make_vdn_forward(block.attn, vdn_state, index))

    wrappers = (
        [DiffusionWrapper(cfg), diffusion_model_wrapper]
        if sol_outer
        else [diffusion_model_wrapper, DiffusionWrapper(cfg)]
    )
    spectrum = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            bootstrap_first_forecast=False,
            offline_smoothing_replay=False,
        )
    )
    sigmas = torch.linspace(1, 0, 9)
    run = spectrum.start_run(sigmas, "sample_euler", supported_sampler=True)
    identities = []
    runtime_instances = []

    def sample():
        for sigma in sigmas[:-1]:
            decision = spectrum.begin_step(sigma.reshape(1))
            to = {
                **opts,
                RUNTIME_KEY: spectrum,
                RUN_ID_KEY: run,
                STEP_ID_KEY: decision["step_id"],
                # Diff-Aid normalizes its current timestep against this complete
                # schedule. Keep both in the same scale as production wrappers.
                "sample_sigmas": sigmas * 1000,
            }
            to["optimized_attention_override"] = make_minimax_h3_attention_override(None)
            to["minimax_h3_untwist_rope"] = {
                "enabled": True,
                "progress": float(sigma),
                "start_percent": 0.0,
                "end_percent": 1.0,
                "high_scale_start": 1.0,
                "high_scale_end": 1.0,
                "low_scale_start": 1.0,
                "low_scale_end": 1.0,
                "beta": 2.0,
                "reference_ranges": ((0, 1),),
                "rope_axis_count": 1,
                "rope_freqs_per_axis": 1,
                "scale_temporal_axis": False,
            }
            timestep = (sigma * 1000).reshape(1)

            def execute_after_diffaid(input_x, inner_timestep, **condition):
                local_to = condition["transformer_options"]
                raw_runtime = local_to.get(SPECTRUM_EXTERNAL_RUNTIME_KEY)
                entries = raw_runtime if isinstance(raw_runtime, (tuple, list)) else (raw_runtime,)
                instances = tuple(
                    entry.get("instance_id")
                    for entry in entries
                    if isinstance(entry, dict)
                    and entry.get("provider") == "comfyui-diffaid-patches"
                    and entry.get("schema_version") == 1
                )
                assert instances
                runtime_instances.append(instances)
                identity = HistoryPolicy(cfg)(layout=packed, options=local_to, model=model)
                assert identity is not None
                identities.append(identity)
                executor = WrapperExecutor.new_class_executor(model._forward, model, wrappers)
                return executor.execute(
                    input_x,
                    inner_timestep,
                    context,
                    local_to,
                    minimax_payload={"layout": packed, "seed": 61},
                )

            out = diffaid_wrapper(
                execute_after_diffaid,
                {
                    "input": [video, audio],
                    "timestep": timestep,
                    "c": {"transformer_options": to},
                },
            )
            assert all(torch.isfinite(part).all() for part in out)
            spectrum.finalize_step(run, decision["step_id"])

        state = _REQUEST.get()
        assert state.evaluations == spectrum.stats.actual_transformer_calls
        assert state.vdn_local_sol_calls > 0
        assert state.vdn_rectangular_sol_calls > 0
        assert state.vdn_square_expanded_calls == 0
        assert state.vdn_kernel_q_rows == state.vdn_requested_q_rows > 0
        assert spectrum.stats.forecast_model_calls > 0

    with torch.no_grad():
        SamplingWrapper(cfg)(sample)
    spectrum.end_run(run)

    assert runtime_instances
    assert len(set(runtime_instances)) == 1
    # Dense warmup -> SOL changes phase once. After that, the complete semantic
    # identity (real Diff-Aid declaration + dynamic Untwist + grouped VDN) is stable.
    assert len(set(identities[1:])) == 1
