"""Execute Sol completion receipts through the real Spectrum history consumer."""
import os
import sys

import pytest
import torch

from sol_h3 import partitioned_request as request
from sol_h3.contracts import Config
from sol_h3.interop import HistoryPolicy
from sol_h3.partitioned_history import install_partitioned_history_bridge
from sol_h3.runtime import Request, _FORWARD, _REQUEST


pytestmark = pytest.mark.skipif(
    not os.environ.get("SPECTRUM_PATH"), reason="set SPECTRUM_PATH for the real history consumer",
)


def _sample(monkeypatch, *, change_digest=False):
    sys.path.insert(0, os.environ["SPECTRUM_PATH"])
    from comfyui_spectrum_h3.backend_history import observe, prepare
    from comfyui_spectrum_h3.config import SpectrumH3Config
    from comfyui_spectrum_h3.runtime import SpectrumH3Runtime

    install_partitioned_history_bridge()
    config = Config(backend="sol", dense_evaluations=0)

    class Policy(HistoryPolicy):
        def __call__(self, **kwargs):
            # This fixture holds geometry and route fixed; completion validation
            # and Spectrum's history/reset/forecast paths are the real code.
            return ("partitioned-fixed-dense-route",)

    spectrum = SpectrumH3Runtime(SpectrumH3Config(
        warmup_steps=2, tail_actual_steps=1, window_size=2, flex_window=0,
        bootstrap_first_forecast=False, offline_smoothing_replay=False,
    ))
    sigmas = torch.linspace(1, 0, 9)
    run = spectrum.start_run(sigmas, "sample_res_multistep", supported_sampler=True)
    state = Request(config)
    modes, receipts = [], []
    generator = torch.Generator().manual_seed(812)
    q = torch.randn((3, 2, 128), generator=generator)
    k = torch.randn((11, 2, 128), generator=generator)
    v = torch.randn(k.shape, generator=generator)
    monkeypatch.setattr(request, "_validate_thd", lambda *_args: None)
    token = _REQUEST.set(state)
    try:
        for step, sigma in enumerate(sigmas[:-1]):
            decision = spectrum.begin_step(sigma.reshape(1))
            step_id = decision["step_id"]
            options, policy = prepare(
                spectrum, run, step_id, {"attention_backend_history_v1": {"sol_h3": Policy(config)}},
                None, None,
            )
            call, actual = spectrum.begin_model_call(
                run, step_id, topology=(("target_audio_rows", 1), ("target_video_rows", 2)),
                labels=((0, "positive"),), expected_shape=(1, 3, 256),
            )
            modes.append(actual)
            if actual:
                forward = _FORWARD.set((None, state, state.evaluations, None, []))
                try:
                    output = request.partitioned_request_attention(
                        q * (1 + float(sigma)), k, v, transformer_options=options, block_index=0,
                        kind="local", scale=128 ** -0.5, sink_rows=3, prefix_k_range=(3, 7),
                        prefix_log_key_measure=0.0,
                        semantic_digest=("b" if change_digest and step >= 5 else "a") * 64,
                        force_dense=True,
                    )
                    observe(spectrum, run, step_id, options, policy)
                    spectrum.observe_actual(run, step_id, call, output.reshape(1, 3, 256))
                    receipts.append(tuple(options["attention_backend_receipts_v1"]))
                finally:
                    _FORWARD.reset(forward)
                state.evaluations += 1
            else:
                prediction = spectrum.predict(run, step_id, call, device=q.device, dtype=q.dtype)
                assert prediction is not None and torch.isfinite(prediction).all()
            spectrum.finalize_step(run, step_id)
    finally:
        _REQUEST.reset(token)
    return modes, receipts, spectrum


def test_completed_partitioned_routes_accumulate_anchors_and_resume_forecasts(monkeypatch):
    modes, receipts, spectrum = _sample(monkeypatch)
    assert modes == [True, True, False, True, False, True, False, True]
    assert spectrum.stats.backend_history_resets == 0
    assert spectrum.forecaster.history_length >= 2
    assert all(receipt == receipts[0] for receipt in receipts)


def test_changed_numerical_geometry_still_resets_partitioned_history(monkeypatch):
    modes, _receipts, spectrum = _sample(monkeypatch, change_digest=True)
    assert spectrum.stats.backend_history_resets == 1
    assert modes[6] is True
