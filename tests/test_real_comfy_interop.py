"""Current Comfy ModelPatcher/wrap_attn, with small CPU native H3 modules."""
import ast
import logging
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

comfy_path = os.environ.get("COMFYUI_PATH")
pytestmark = pytest.mark.skipif(not comfy_path, reason="set COMFYUI_PATH for real Comfy integration")


@pytest.fixture
def native():
    sys.path.insert(0, comfy_path)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor, WrappersMP
    inner = MiniMaxH3Model(hidden_size=8, num_layers=3, token_refiner_num_layers=0,
                          num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
                          text_dim=8, time_embed_hidden_size=8, time_embed_dim=4,
                          timestep_input_dim=8, dtype=torch.float32, device="cpu", operations=torch.nn)
    outer = torch.nn.Module()
    outer.diffusion_model = inner
    return ModelPatcher(outer, torch.device("cpu"), torch.device("cpu")), WrapperExecutor, WrappersMP


def test_real_modelpatcher_duplicate_owner_and_wrappers(native):
    from sol_h3.contracts import Config, KEY
    from sol_h3.runtime import install, _REQUEST
    patcher, Executor, Kinds = native
    a = install(patcher, Config(backend="sol", exact=False))
    b = install(a, Config())
    assert len(b.get_wrappers(Kinds.OUTER_SAMPLE, KEY)) == 1
    assert len(b.get_wrappers(Kinds.DIFFUSION_MODEL, KEY)) == 1
    assert a.model_options["transformer_options"][KEY]["exact"] is False
    def run():
        assert _REQUEST.get().config.exact
        return 7
    assert Executor.new_executor(run, b.get_all_wrappers(Kinds.OUTER_SAMPLE)).execute() == 7
    assert _REQUEST.get() is None


@pytest.mark.parametrize("mode", ["auto", "sageattn3"])
def test_actual_kj_sage_wrapper_and_comfy_recursion(native, monkeypatch, mode):
    kj = os.environ.get("KJNODES_PATH")
    if not kj:
        pytest.skip("set KJNODES_PATH for audited KJ Sage wrapper")
    from comfy.ldm.modules.attention import wrap_attn, attention_pytorch
    from sol_h3.contracts import Config
    from sol_h3.runtime import BlockPatch, Request, _FORWARD
    from sol_h3 import runtime, sparse
    source = Path(kj, "nodes/model_optimization_nodes.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "get_sage_func")
    ns = dict(torch=torch, logging=logging, wrap_attn=wrap_attn, attention_pytorch=attention_pytorch)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "kj-get-sage", "exec"), ns)
    def sage(q, k, v, **kw):
        if kw.get("tensor_layout", "HND") == "NHD":
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    module = ModuleType("sageattention" if mode == "auto" else "sageattn3")
    setattr(module, "sageattn" if mode == "auto" else "sageattn3_blackwell", sage)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sage_func = ns["get_sage_func"](mode, allow_compile=True)
    def previous(original, *args, **kwargs):
        return sage_func.__wrapped__(*args, **kwargs)
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    model = native[0].model.diffusion_model
    token = _FORWARD.set((model, state, 0, set(), []))
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **kw: None)
    def kernel(q, k, v, **kw):
        return sage(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    q = torch.randn(1, 1, 7, 128, dtype=torch.bfloat16)
    from types import SimpleNamespace
    layout = SimpleNamespace(seq_len=7, segments=[(0, 3, "text"), (3, 7, "video")])
    def block(args):
        got = attention_pytorch(q, q, q, 1, skip_reshape=True, transformer_options=args["transformer_options"])
        assert got.shape == (1, 7, 128)
        return {"img": args["img"]}
    try:
        with torch.no_grad():
            BlockPatch(0, cfg)({"img": torch.zeros(7, 128), "layout": layout,
                "transformer_options": {"optimized_attention_override": previous}}, {"original_block": block})
    finally:
        _FORWARD.reset(token)
    assert state.sparse_calls == 1


def test_real_modelpatcher_vdn_v2_object_patch_reaches_sol(monkeypatch):
    vdn_path = os.environ.get("VDN_PATH")
    if not vdn_path:
        pytest.skip("set VDN_PATH for real VDN ModelPatcher integration")
    sys.path.insert(0, comfy_path)
    sys.path.insert(0, vdn_path)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    from types import SimpleNamespace
    from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor, WrappersMP
    from vdn_h3.hybrid import VDNState, apply_vdn
    from sol_h3 import runtime, sparse
    from sol_h3.contracts import Config
    from sol_h3.runtime import install, _REQUEST

    torch.manual_seed(41)
    inner = MiniMaxH3Model(
        hidden_size=128, num_layers=3, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=128, ffn_hidden_size=256,
        latents_dim=2, audio_latents_dim=2, text_dim=128,
        timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
        rope_inv_freq_len=1, dtype=torch.bfloat16, device=torch.device("cpu"),
        operations=torch.nn).eval().requires_grad_(False)
    inner.rope.inv_freq.fill_(1.)
    outer = torch.nn.Module()
    outer.diffusion_model = inner
    patcher = ModelPatcher(outer, torch.device("cpu"), torch.device("cpu"))

    vdn_cfg = {"radius": 0, "chunk": 1, "anchor_frames": "both",
               "enable_softmax_gate": True, "linear_enabled": False}
    branches = [SimpleNamespace(w={}, enable_text_state=False) for _ in inner.blocks]
    vdn_state = VDNState("object-patch-test", vdn_cfg, branches, 1, 128)
    vdn_state.softmax_backend = "grouped"
    gate_weight = torch.randn(1, 128, dtype=torch.bfloat16) * .01
    gate_bias = torch.zeros(1, dtype=torch.bfloat16)
    monkeypatch.setattr(vdn_state, "weights_on", lambda *a: {
        "softmax_gate.up.weight": gate_weight,
        "softmax_gate.up.bias": gate_bias,
    })
    apply_vdn(patcher, vdn_state)
    assert all(getattr(patcher.object_patches[
        f"diffusion_model.blocks.{i}.attn.forward"], "_vdn_softmax_provider_api", 0) == 2
        for i in range(3))

    preprocess_calls = []
    def transform(q, k, v, heads, **kw):
        preprocess_calls.append((q.shape[2], heads))
        return q, k, v
    def inherited(original, *args, **kwargs):
        return original(*args, **kwargs)
    inherited.attention_preprocess_v1 = (transform, None)
    patcher.model_options["transformer_options"]["optimized_attention_override"] = inherited

    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    sol = install(patcher, cfg)
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **kw: None)
    def oracle(q, k, v, **kw):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    monkeypatch.setattr(sparse, "load_kernel", lambda device: oracle)

    video = torch.randn(1, 2, 4, 4, 4, dtype=torch.bfloat16)
    audio = torch.randn(1, 2, 2, 3, dtype=torch.bfloat16)
    context = torch.randn(1, 2, 128, dtype=torch.bfloat16)
    packed = PackedLayout(2, 4, 4, 4, 3)
    patched_outer = sol.patch_model(load_weights=False)
    patched = patched_outer.diffusion_model
    try:
        assert getattr(patched.blocks[0].attn.forward, "_vdn_softmax_provider_api", 0) == 2
        to = dict(sol.model_options["transformer_options"])
        to.update({"cond_or_uncond": [0], "uuids": ["positive"]})

        def actual():
            executor = WrapperExecutor.new_class_executor(
                patched._forward, patched,
                sol.get_all_wrappers(WrappersMP.DIFFUSION_MODEL))
            out = executor.execute(
                [video, audio], torch.ones(1), context, to,
                minimax_payload={"layout": packed, "seed": 41})
            assert all(torch.isfinite(part).all() for part in out)
            state = _REQUEST.get()
            assert state.sparse_calls > 0
            assert state.vdn_local_sol_calls > 0
            assert state.vdn_square_expanded_calls > 0
            assert state.vdn_square_kernel_rows > state.vdn_square_requested_rows > 0
            assert state.fallbacks["vdn_global_native"] > 0

        with torch.no_grad():
            WrapperExecutor.new_executor(
                actual, sol.get_all_wrappers(WrappersMP.OUTER_SAMPLE)).execute()
        assert preprocess_calls
    finally:
        sol.unpatch_model(unpatch_weights=False)
