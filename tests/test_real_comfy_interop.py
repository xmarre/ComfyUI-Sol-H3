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
    # Execute the unmodified current KJ function; only the unavailable Sage CUDA
    # extension is replaced by CPU SDPA. This is wrapper correctness evidence.
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
