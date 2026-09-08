"""Contract smoke for the Sol-Attn API shipped with the audited ComfyUI checkout."""
import inspect
import os
import sys

import pytest


comfy_path = os.environ.get("COMFYUI_PATH")
pytestmark = pytest.mark.skipif(not comfy_path, reason="set COMFYUI_PATH for current ComfyUI contract")


def test_current_comfy_kitchen_sol_attn_api():
    sys.path.insert(0, comfy_path)
    import comfy_kitchen as ck

    assert callable(ck.sol_attn)
    assert callable(ck.sol_attn_is_available)
    params = inspect.signature(ck.sol_attn).parameters
    assert tuple(params) == (
        "q", "k", "v", "tau", "scale", "sink_blocks", "sink_q", "key_bias",
        "topk_ratio", "tail", "block_len", "coarse_gate", "token_aug",
    )
    assert params["tau"].default == 1.0
    assert params["topk_ratio"].default == 0.0
    assert params["tail"].default is True
    assert params["token_aug"].default == 0
