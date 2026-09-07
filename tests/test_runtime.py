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


def test_exact_wraps_existing_block_replacement(patcher):
    previous = object()
    patcher.set_model_patch_replace(previous, "dit", "double_block", 0)
    installed = install(patcher, Config())
    replacement = installed.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    assert isinstance(replacement, BlockPatch)
    assert replacement.previous is previous


def test_sparse_conflicting_block_fails_before_clone(patcher):
    patcher.set_model_patch_replace(object(), "dit", "double_block", 0)
    with pytest.raises(RuntimeError, match="sole sparse/block owner"):
        install(patcher, Config(exact=False, backend="sol"))


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
    state = SimpleNamespace(config=cfg, sparse_calls=0)
    token = _FORWARD.set((object(), state, 0, set()))
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


def test_previous_block_patch_delegates_to_exact_once(monkeypatch):
    from sol_h3 import exact

    cfg = Config()
    block = object()
    model = SimpleNamespace(blocks=[block])
    state = SimpleNamespace(config=cfg, sparse_calls=0, exact_blocks=0, exact_verified=set())
    calls = []

    def fake_execute(current_block, args, verified):
        assert current_block is block
        assert args["img"] == "changed"
        calls.append("exact")
        return {"img": "done"}

    def previous(args, extra):
        calls.append("previous")
        return extra["original_block"]({**args, "img": "changed"})

    monkeypatch.setattr(exact, "execute_block", fake_execute)
    token = _FORWARD.set((model, state, 0, set()))
    try:
        result = BlockPatch(0, cfg, previous)(
            {"img": "start", "transformer_options": {}},
            {"original_block": lambda args: {"img": "native"}},
        )
    finally:
        _FORWARD.reset(token)
    assert result == {"img": "done"}
    assert calls == ["previous", "exact"]
    assert state.exact_blocks == 1


def test_previous_block_patch_must_delegate_exactly_once(monkeypatch):
    from sol_h3 import exact

    cfg = Config()
    model = SimpleNamespace(blocks=[object()])
    state = SimpleNamespace(config=cfg, sparse_calls=0, exact_blocks=0, exact_verified=set())
    monkeypatch.setattr(exact, "execute_block", lambda block, args, verified: {"img": args["img"]})
    token = _FORWARD.set((model, state, 0, set()))
    try:
        with pytest.raises(RuntimeError, match="delegate exactly once"):
            BlockPatch(0, cfg, lambda args, extra: {"img": args["img"]})(
                {"img": "start", "transformer_options": {}},
                {"original_block": lambda args: {"img": "native"}},
            )
    finally:
        _FORWARD.reset(token)


def test_metadata_tamper_is_error():
    with pytest.raises(RuntimeError, match="metadata"):
        DiffusionWrapper(Config())(None, None, None, None, {KEY: {}})


def test_arbitrary_timestep_order_and_dense_warmup(monkeypatch):
    import torch
    from sol_h3 import sparse
    cfg = Config(exact=False, backend="sol", dense_evaluations=1, dense_layers=2)
    def fake_attention(q, k, v, prefix, config, state, dense_attention=None):
        assert callable(dense_attention)
        state.sparse_calls += 1
        return q.transpose(1, 2).reshape(1, q.shape[2], -1)
    monkeypatch.setattr(sparse, "attention", fake_attention)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda: None)) for _ in range(3)])
    class Executor:
        class_obj = model
        def __call__(self, x, t, context, options, **kwargs):
            for i in range(3):
                args = {"img": torch.zeros(7, 128), "transformer_options": options,
                        "layout": SimpleNamespace(seq_len=7, segments=[(0, 2, "text"), (2, 3, "audio"), (3, 7, "video")])}
                def original(forwarded):
                    override = forwarded["transformer_options"].get("optimized_attention_override")
                    if override:
                        q = torch.zeros(1, 1, 7, 128)
                        override(lambda q, k, v, heads, **kw: q, q, q, q, 1, skip_reshape=True)
                    return {"img": forwarded["img"]}
                BlockPatch(i, cfg)(args, {"original_block": original})
            return x
    stats = []
    def sample():
        with torch.no_grad():
            for t in (1000, 700, 900):
                DiffusionWrapper(cfg)(Executor(), [torch.zeros(1)], t, None, {KEY: cfg.metadata()})
        state = _REQUEST.get()
        stats.append((state.evaluations, state.sparse_calls, state.dense_calls))
    SamplingWrapper(cfg)(sample)
    SamplingWrapper(cfg)(sample)
    assert stats == [(3, 2, 7), (3, 2, 7)]
    assert _FORWARD.get() is None and _REQUEST.get() is None


def test_late_provider_cannot_skip_all_blocks():
    import torch
    cfg = Config(exact=False)
    class Executor:
        class_obj = SimpleNamespace(blocks=[object()])
        def __call__(self, *args, **kwargs):
            return None
    def sample():
        with torch.no_grad():
            return DiffusionWrapper(cfg)(Executor(), [torch.zeros(1)], 0, None, {KEY: cfg.metadata()})
    with pytest.raises(RuntimeError, match="apply Sol-H3 after wrappers"):
        SamplingWrapper(cfg)(sample)
    assert _FORWARD.get() is None and _REQUEST.get() is None
