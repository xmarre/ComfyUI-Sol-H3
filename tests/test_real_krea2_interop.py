"""Exercise Sol-Krea2's dense ownership through real ComfyUI attention wrappers."""
import os
import sys

import pytest
import torch

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")
pytestmark = pytest.mark.skipif(not COMFYUI_PATH, reason="set COMFYUI_PATH for real ComfyUI integration")


def test_real_krea2_modelpatcher_preserves_inherited_dense_attention():
    sys.path.insert(0, COMFYUI_PATH)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    from comfy.ldm.krea2.model import SingleStreamDiT
    from comfy.ldm.modules.attention import optimized_attention_masked
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor, WrappersMP

    from sol_h3.krea2 import Settings, _ACTIVE, install

    inner = SingleStreamDiT(
        features=128, heads=1, kvheads=1, layers=1, txtdim=128,
        txtlayers=1, txtheads=1, channels=16, device="cpu",
        dtype=torch.float32, operations=torch.nn,
    )
    outer = torch.nn.Module()
    outer.diffusion_model = inner
    source = ModelPatcher(outer, torch.device("cpu"), torch.device("cpu"))

    observed = []
    def previous(func, *args, **kwargs):
        observed.append((args[0].shape, kwargs.get("skip_reshape")))
        return func(*args, **kwargs)

    source.model_options["transformer_options"]["optimized_attention_override"] = previous
    patcher = install(source, Settings(dense_evaluations=0, dense_layers=0, min_tokens=0))
    assert source.model_options["transformer_options"]["optimized_attention_override"] is previous

    q = torch.randn(1, 1, 4, 128, dtype=torch.float32)
    opts = patcher.model_options["transformer_options"]
    opts.update({"block_type": "single", "total_blocks": 1, "block_index": 0, "img_slice": [1, 4]})

    def invoke_attention():
        return optimized_attention_masked(
            q, q, q, 1, skip_reshape=True, transformer_options=opts,
        )

    def run_in_diffusion_scope():
        return WrapperExecutor.new_executor(
            invoke_attention, patcher.get_all_wrappers(WrappersMP.DIFFUSION_MODEL),
        ).execute()

    with torch.no_grad():
        expected = optimized_attention_masked(q, q, q, 1, skip_reshape=True, transformer_options={})
        actual = WrapperExecutor.new_executor(
            run_in_diffusion_scope, patcher.get_all_wrappers(WrappersMP.OUTER_SAMPLE),
        ).execute()

    torch.testing.assert_close(actual, expected)
    assert observed == [(q.shape, True)]
    assert _ACTIVE.get() is None
