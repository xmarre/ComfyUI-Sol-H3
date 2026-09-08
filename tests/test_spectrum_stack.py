"""Real native H3 and Spectrum capture ordering with an explicit CPU SOL oracle."""
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH") or not os.environ.get("SPECTRUM_PATH"),
    reason="set COMFYUI_PATH and SPECTRUM_PATH for real stack contracts")


@pytest.mark.parametrize("sol_outer", [False, True])
@pytest.mark.parametrize("vdn_mode", [None, "full", "grouped", "flex"])
def test_real_spectrum_capture_and_actual_warmup_both_wrapper_orders(monkeypatch, sol_outer, vdn_mode):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    sys.path.insert(0, os.environ["SPECTRUM_PATH"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
    from comfy.patcher_extension import WrapperExecutor
    from comfyui_spectrum_h3.config import SpectrumH3Config
    from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
    from comfyui_spectrum_h3.minimax_h3 import diffusion_model_wrapper
    from comfyui_spectrum_h3.sampling import RUNTIME_KEY, RUN_ID_KEY, STEP_ID_KEY
    from sol_h3 import runtime, sparse
    from sol_h3.contracts import Config, KEY
    from sol_h3.interop import HISTORY_KEY, HistoryPolicy
    from sol_h3.runtime import BlockPatch, SamplingWrapper, DiffusionWrapper, _REQUEST

    torch.manual_seed(37)
    model = MiniMaxH3Model(hidden_size=128, num_layers=3, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=128, ffn_hidden_size=256,
        latents_dim=2, audio_latents_dim=2, text_dim=128, timestep_input_dim=4,
        time_embed_hidden_size=8, time_embed_dim=4, rope_inv_freq_len=1,
        dtype=torch.bfloat16, device=torch.device("cpu"), operations=torch.nn).eval().requires_grad_(False)
    model.rope.inv_freq.fill_(1.)
    cfg = Config(exact=False, backend="sol", dense_evaluations=1, dense_layers=0)
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    def oracle(q, k, v, **kw):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    monkeypatch.setattr(sparse, "load_kernel", lambda device: oracle)
    frames = 1 if vdn_mode in (None, "full") else 4
    video = torch.randn(1, 2, frames, 4, 4)
    audio = torch.randn(1, 2, 2, 3)
    context = torch.randn(1, 2, 128, dtype=torch.bfloat16)
    packed = PackedLayout(2, frames, 4, 4, 3)
    if vdn_mode:
        if not os.environ.get("VDN_PATH"):
            pytest.skip("set VDN_PATH for real VDN hybrid dispatch")
        sys.path.insert(0, os.environ["VDN_PATH"])
        from types import SimpleNamespace
        from vdn_h3.hybrid import VDNState, VDNLayout, make_vdn_forward
        from vdn_h3 import window
        vdn_cfg = {"radius": 0, "chunk": 1, "anchor_frames": "both",
                   "enable_softmax_gate": True, "linear_enabled": False}
        vdn_state = VDNState("stack-test", vdn_cfg,
                             [SimpleNamespace(w={}, enable_text_state=False) for _ in model.blocks], 1, 128)
        vdn_state.softmax_backend = "flex" if vdn_mode == "flex" else "grouped"
        va, vb, _ = next(segment for segment in packed.segments if segment[2] == "video")
        aa, ab, _ = next(segment for segment in packed.segments if segment[2] == "audio")
        ta, tb, _ = next(segment for segment in packed.segments if segment[2] == "text")
        vdn_layout = VDNLayout(
            video_start=va, video_end=vb,
            audio_start=aa, audio_end=ab,
            num_frames=frames, tokens_per_frame=4, frame_size=(2, 2),
            text_start=ta, text_len=tb - ta, seq_len=packed.seq_len,
            radius=0, chunk=1, anchor_frames="both")
        vdn_state._layout.set(vdn_layout)
        gate_weight = torch.randn(1, 128, dtype=torch.bfloat16) * .01
        gate_bias = torch.zeros(1, dtype=torch.bfloat16)
        monkeypatch.setattr(vdn_state, "weights_on", lambda *a: {
            "softmax_gate.up.weight": gate_weight, "softmax_gate.up.bias": gate_bias})
        if vdn_mode == "flex":
            def unavailable_flex(*a, **kw):
                raise RuntimeError("CPU test explicitly exercises Flex-to-grouped fallback")
            monkeypatch.setattr(window, "window_softmax_flex", unavailable_flex)
        for index, block in enumerate(model.blocks):
            monkeypatch.setattr(block.attn, "forward", make_vdn_forward(block.attn, vdn_state, index))

    opts = {KEY: cfg.metadata(), HISTORY_KEY: {"sol_h3": HistoryPolicy(cfg)},
            "patches_replace": {"dit": {("double_block", i): BlockPatch(i, cfg) for i in range(3)}},
            "cond_or_uncond": [0], "uuids": ["positive"]}
    wrappers = [DiffusionWrapper(cfg), diffusion_model_wrapper] if sol_outer else [diffusion_model_wrapper, DiffusionWrapper(cfg)]
    counts = []
    for _chunk in range(2):
        spectrum = SpectrumH3Runtime(SpectrumH3Config(degree=1, max_history=4, warmup_steps=2,
            tail_actual_steps=0, bootstrap_first_forecast=False, offline_smoothing_replay=False))
        sigmas = torch.linspace(1, 0, 9)
        run = spectrum.start_run(sigmas, "sample_euler", supported_sampler=True)
        def sample():
            for sigma in sigmas[:-1]:
                decision = spectrum.begin_step(sigma.reshape(1))
                to = {**opts, RUNTIME_KEY: spectrum, RUN_ID_KEY: run, STEP_ID_KEY: decision["step_id"]}
                executor = WrapperExecutor.new_class_executor(model._forward, model, wrappers)
                out = executor.execute([video, audio], (sigma * 1000).reshape(1), context, to,
                                       minimax_payload={"layout": packed, "seed": 37})
                assert all(torch.isfinite(part).all() for part in out)
                spectrum.finalize_step(run, decision["step_id"])
            state = _REQUEST.get()
            counts.append((state.evaluations, spectrum.stats.actual_transformer_calls,
                           spectrum.stats.forecast_model_calls, state.sparse_calls))
            assert state.evaluations == spectrum.stats.actual_transformer_calls
            assert state.sparse_calls > 0
            if vdn_mode in (None, "full"):
                assert state.dense_calls == 3
            else:
                assert state.vdn_local_sol_calls > 0
                assert state.vdn_rectangular_sol_calls > 0
                assert state.vdn_square_expanded_calls == 0
                assert state.vdn_kernel_q_rows == state.vdn_requested_q_rows > 0
                assert state.fallbacks["vdn_global_native"] > 0
            if vdn_mode == "flex":
                # Flex can fail into grouped at runtime and the stack-compatible VDN
                # overlay intentionally does not own hybrid.py, so preflight cannot
                # prove that route. Spectrum must execute actuals rather than forecast
                # across an opaque transition; the grouped fallback still consumes SOL.
                assert spectrum.stats.forecast_model_calls == 0
            else:
                assert spectrum.stats.forecast_model_calls > 0
        with torch.no_grad():
            SamplingWrapper(cfg)(sample)
        spectrum.end_run(run)
    assert counts[0] == counts[1]
