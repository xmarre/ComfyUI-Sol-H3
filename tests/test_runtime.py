import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sol_h3 import runtime
from sol_h3.contracts import KEY, Config
from sol_h3.runtime import BlockPatch, DiffusionWrapper, Request, SamplingWrapper, _FORWARD, _REQUEST, install


@pytest.fixture
def patcher(monkeypatch):
    class Native:
        def __init__(self):
            self.blocks = [SimpleNamespace(attn=SimpleNamespace(forward=lambda: None)) for _ in range(3)]
            self.use_adaln_curves = False
    class Patcher:
        def __init__(self):
            self.inner = Native()
            self.model_options = {"transformer_options": {}}
            self.wrappers = {}
            self.object_patches = {}
        def get_model_object(self, name):
            return self.inner
        def clone(self):
            other = copy.copy(self)
            other.model_options = copy.deepcopy(self.model_options)
            other.wrappers = copy.deepcopy(self.wrappers)
            other.object_patches = dict(self.object_patches)
            return other
        def add_wrapper_with_key(self, kind, key, wrapper):
            self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)
        def remove_wrappers_with_key(self, kind, key):
            self.wrappers.get(kind, {}).pop(key, None)
        def set_model_patch_replace(self, patch, name, block, i):
            self.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault(name, {})[(block, i)] = patch
    native = ModuleType("comfy.ldm.minimax.model")
    native.MiniMaxH3Model = Native
    ext = ModuleType("comfy.patcher_extension")
    ext.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", DIFFUSION_MODEL="diffusion_model")
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.model", native)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", ext)
    return Patcher()


@pytest.mark.parametrize("first,second", [
    (Config(), Config(backend="sol")),
    (Config(backend="sol"), Config()),
    (Config(backend="sol", exact=False), Config()),
    (Config(backend="sol"), Config(backend="sol")),
])
def test_merge_single_owner_and_clone_isolation(patcher, first, second):
    a = install(patcher, first)
    b = install(a, second)
    c = install(a, second)
    assert patcher.model_options == {"transformer_options": {}}
    assert a.model_options["transformer_options"][KEY] == first.metadata()
    assert b.model_options["transformer_options"][KEY] == Config(backend="sol").metadata()
    for wrappers in b.wrappers.values():
        assert len(wrappers[KEY]) == 1
    b.model_options["transformer_options"]["branch"] = True
    assert "branch" not in c.model_options["transformer_options"]


def test_conflicting_sparse_policy_is_explicit(patcher):
    a = install(patcher, Config(backend="sol"))
    with pytest.raises(ValueError, match="Conflicting SOL policies"):
        install(a, Config(backend="sol", tau=.5))


def test_existing_block_provider_is_chained(patcher):
    previous = object()
    patcher.set_model_patch_replace(previous, "dit", "double_block", 0)
    b = install(patcher, Config(backend="sol"))
    assert b.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)].previous is previous


def test_sampling_scope_reentrancy_and_exception_cleanup():
    wrapper = SamplingWrapper(Config(exact=False))
    def outer():
        state = _REQUEST.get()
        def inner():
            assert _REQUEST.get() is not state
            raise ValueError("cancelled")
        with pytest.raises(ValueError):
            wrapper(inner)
        assert _REQUEST.get() is state
    wrapper(outer)
    assert _REQUEST.get() is None


def test_zero_sparse_and_forecast_do_not_fail_or_consume_warmup():
    config = Config(backend="sol")
    class Executor:
        class_obj = SimpleNamespace(blocks=[object()])
        def __call__(self, *args, **kwargs):
            return "forecast"
    def sample():
        result = DiffusionWrapper(config)(Executor(), [torch.zeros(1)], 0, None, {KEY: config.metadata()})
        assert result == "forecast"
        assert _REQUEST.get().evaluations == 0
    SamplingWrapper(config)(sample)
    assert _REQUEST.get() is None and _FORWARD.get() is None


def test_metadata_tamper_is_error():
    with pytest.raises(RuntimeError, match="metadata"):
        DiffusionWrapper(Config())(None, None, None, None, {KEY: {}})


def test_block_replacement_may_bypass_or_delegate_multiple_times():
    cfg = Config(exact=False)
    state = Request(cfg)
    token = _FORWARD.set((SimpleNamespace(blocks=[object()]), state, 0, set(), []))
    args = {"img": 0, "transformer_options": {}}
    try:
        def previous(args, extra):
            extra["original_block"](args)
            return extra["original_block"](args)
        out = BlockPatch(0, cfg, previous)(args, {"original_block": lambda a: {"img": 7}})
        assert out == {"img": 7}
        assert state.evaluations == 1
    finally:
        _FORWARD.reset(token)


def test_unavailable_inherited_dense_provider_demotes_once(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=99, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=None))])
    q = torch.zeros(1, 1, 7, 128, dtype=torch.bfloat16)
    broken_calls = []
    original_calls = []

    def broken(original, q, k, v, heads, **kw):
        broken_calls.append(True)
        raise ImportError(
            "/tmp/sageattention/_fused.so: /lib/x86_64-linux-gnu/libstdc++.so.6: "
            "version `GLIBCXX_3.4.32' not found"
        )

    def original(q, k, v, heads, **kw):
        original_calls.append(True)
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)

    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    opts = {"optimized_attention_override": broken}

    def block(args):
        to = args["transformer_options"]
        out = to["optimized_attention_override"](
            original, q, q, q, 1, skip_reshape=True, transformer_options=to
        )
        return {"img": out}

    for evaluation in range(2):
        token = _FORWARD.set((model, state, evaluation, set(), []))
        try:
            BlockPatch(0, cfg)(
                {"img": torch.zeros(7, 128), "layout": SimpleNamespace(
                    seq_len=7, segments=[(0, 3, "text"), (3, 7, "video")]),
                 "transformer_options": opts},
                {"original_block": block},
            )
        finally:
            _FORWARD.reset(token)

    assert len(broken_calls) == 1
    assert len(original_calls) == 2
    assert sum(state.dense_provider_failures.values()) == 1
    assert state.fallbacks["dense_provider_unavailable:binary_abi"] == 1
    assert state.backend_transitions == 1
    assert state.disabled_dense_providers == {id(broken)}
    assert any("original" in name for name in state.dense_attention_backends)


def test_dense_provider_compute_errors_are_not_swallowed(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=99, dense_layers=0)
    state = Request(cfg)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=None))])
    q = torch.zeros(1, 1, 7, 128, dtype=torch.bfloat16)

    def broken(original, q, k, v, heads, **kw):
        raise RuntimeError("CUDA launch failed")

    def original(q, k, v, heads, **kw):
        raise AssertionError("must not fall back for compute errors")

    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    opts = {"optimized_attention_override": broken}

    def block(args):
        to = args["transformer_options"]
        return {"img": to["optimized_attention_override"](
            original, q, q, q, 1, skip_reshape=True, transformer_options=to
        )}

    token = _FORWARD.set((model, state, 0, set(), []))
    try:
        with pytest.raises(RuntimeError, match="CUDA launch failed"):
            BlockPatch(0, cfg)(
                {"img": torch.zeros(7, 128), "layout": SimpleNamespace(
                    seq_len=7, segments=[(0, 3, "text"), (3, 7, "video")]),
                 "transformer_options": opts},
                {"original_block": block},
            )
    finally:
        _FORWARD.reset(token)

    assert not state.disabled_dense_providers
    assert not state.dense_provider_failures
