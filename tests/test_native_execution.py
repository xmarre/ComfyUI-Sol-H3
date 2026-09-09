"""Real PyTorch block arithmetic from the audited native source excerpt.

CPU tests verify integration ordering/hooks. They replace only the unavailable
CUDA affine function with its literal reference, and make no kernel parity claim.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from sol_h3 import exact
from sol_h3.native_contract import function_digest


@pytest.fixture
def native(monkeypatch):
    path = Path(__file__).parent / "fixtures/native_block.py"
    spec = importlib.util.spec_from_file_location("comfy.ldm.minimax.model", path)
    module = importlib.util.module_from_spec(spec)
    module.nn = nn

    class Attention(nn.Module):
        def __init__(self, hidden, *args, **kwargs):
            super().__init__()
            self.proj = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x, rope_freqs=None, transformer_options=None):
            return self.proj(x)

    class MLP(nn.Module):
        def __init__(self, hidden, ffn, **kwargs):
            super().__init__()
            self.proj = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x):
            return self.proj(x)

    module.Attention, module.MLP = Attention, MLP
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def reference(h, shift, scale, segments, verified):
        return module._mod_scale_shift(h, shift, scale, segments)

    monkeypatch.setattr(exact, "affine", reference)
    return module


def test_native_manifest_matches_verbatim_excerpt(native):
    manifest = json.loads((Path(__file__).parents[1] / "sol_h3/native_manifest.json").read_text())
    for path, expected in manifest.items():
        function = native
        for name in path.split("."):
            function = getattr(function, name)
        assert function_digest(function) == expected


@pytest.mark.parametrize("indexed", [False, True])
def test_native_block_ordering_and_projection_hooks(native, indexed):
    torch.manual_seed(123)
    block = native.DiTBlock(8, 1, 8, 16, 4, 1e-5, 1e-5, operations=nn).eval()
    h, emb = torch.randn(9, 8), torch.randn(2, 4)
    rows = torch.tensor([0, 1, 2, 3, 4, 5]) if indexed else 3
    segments = [(0, 3, 1), (3, 9, rows)]
    calls = []
    # Represents dynamically applied adapter projection hooks. Changing the
    # hook contribution between runs must be observed, never cached.
    delta = [0.125]

    def adapter(module, args, output):
        calls.append("adapter")
        return output + delta[0]

    handle = block.adaln_proj.linear.register_forward_hook(adapter)
    try:
        for strength in (0.125, -0.25):
            delta[0] = strength
            with torch.no_grad():
                want = block(h.clone(), emb, segments, None)
                got = exact.execute_block(
                    block,
                    {"img": h.clone(), "t_emb": emb, "mod_segments": segments,
                     "rope_freqs": None, "transformer_options": {}},
                    set(),
                )["img"]
            assert torch.equal(got, want)
        assert len(calls) == 4
    finally:
        handle.remove()


def test_block_hooks_rejected(native):
    block = native.DiTBlock(8, 1, 8, 16, 4, 1e-5, 1e-5, operations=nn)
    hook = block.register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(RuntimeError, match="hooks"):
            exact.execute_block(block, {}, set())
    finally:
        hook.remove()


@pytest.mark.parametrize("kind", ["pre", "post"])
def test_global_forward_hooks_rejected(native, kind):
    block = native.DiTBlock(8, 1, 8, 16, 4, 1e-5, 1e-5, operations=nn)
    module_hooks = torch.nn.modules.module
    register = (
        module_hooks.register_module_forward_pre_hook
        if kind == "pre"
        else module_hooks.register_module_forward_hook
    )
    handle = register(lambda *args: None)
    try:
        with pytest.raises(RuntimeError, match="global forward hooks"):
            exact.execute_block(block, {}, set())
    finally:
        handle.remove()


def test_exact_cuda_capability_error_is_explicit():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        exact.affine(torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), [(0, 1, 0)], set())
