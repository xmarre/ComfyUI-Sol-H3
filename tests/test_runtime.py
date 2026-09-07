import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from sol_h3.contracts import KEY, Config
from sol_h3.runtime import BlockPatch, DiffusionWrapper, SamplingWrapper, _FORWARD, _REQUEST, install


@pytest.fixture
def patcher(monkeypatch):
    class Native:
        def __init__(self):
            self.blocks = [object(), object(), object()]
            self.use_adaln_curves = False
            self.training = False

    class Patcher:
        def __init__(self):
            self.inner = Native()
            self.model_options = {"transformer_options": {}}
            self.wrappers = {}

        def get_model_object(self, name):
            return self.inner

        def clone(self):
            other = copy.copy(self)
            other.model_options = copy.deepcopy(self.model_options)
            other.wrappers = copy.deepcopy(self.wrappers)
            return other

        def add_wrapper_with_key(self, kind, key, wrapper):
            self.wrappers.setdefault(kind, {})[key] = wrapper

        def set_model_patch_replace(self, patch, name, block, i):
            self.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault(name, {})[(block, i)] = patch

    native = ModuleType("comfy.ldm.minimax.model")
    native.MiniMaxH3Model = Native
    ext = ModuleType("comfy.patcher_extension")
    ext.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", DIFFUSION_MODEL="diffusion_model")
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.model", native)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", ext)
    return Patcher()


def test_install_clone_ownership(patcher):
    installed = install(patcher, Config(exact=False))
    assert patcher.model_options == {"transformer_options": {}}
    assert not patcher.wrappers
    assert installed.inner is patcher.inner
    assert len(installed.model_options["transformer_options"]["patches_replace"]["dit"]) == 3
    with pytest.raises(RuntimeError, match="already"):
        install(installed, Config())


def test_conflicting_block_fails_before_clone(patcher):
    patcher.set_model_patch_replace(object(), "dit", "double_block", 0)
    with pytest.raises(RuntimeError, match="owns H3 blocks"):
        install(patcher, Config())


def test_sampling_scope_reentrancy_and_exception_cleanup():
    cfg = Config(exact=False)
    wrapper = SamplingWrapper(cfg)
    def outer():
        outer_state = _REQUEST.get()
        def inner():
            assert _REQUEST.get() is not outer_state
            raise ValueError("cancelled")
        with pytest.raises(ValueError):
            wrapper(inner)
        assert _REQUEST.get() is outer_state
    wrapper(outer)
    assert _REQUEST.get() is None


def test_zero_sparse_is_an_error():
    with pytest.raises(RuntimeError, match="zero sparse"):
        SamplingWrapper(Config(exact=False, backend="sol"))(lambda: None)
    assert _REQUEST.get() is None


def test_native_block_delegation():
    cfg = Config(exact=False)
    state = SimpleNamespace(config=cfg)
    token = _FORWARD.set((object(), state, 0))
    args = {"img": object(), "transformer_options": {"custom": 7}}
    try:
        def original(forwarded):
            assert forwarded["transformer_options"] == {"custom": 7}
            assert forwarded is not args
            return {"img": forwarded["img"]}
        assert BlockPatch(0, cfg)(args, {"original_block": original})["img"] is args["img"]
        assert args["transformer_options"] == {"custom": 7}
    finally:
        _FORWARD.reset(token)


def test_metadata_tamper_is_error():
    with pytest.raises(RuntimeError, match="metadata"):
        DiffusionWrapper(Config())(None, None, None, None, {KEY: {}})
