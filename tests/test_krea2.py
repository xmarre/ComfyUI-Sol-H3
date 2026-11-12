"""CPU-level contract tests for the optional Krea 2 sparse-attention adapter."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sol_h3 import krea2
from sol_h3.sparse import KernelUnavailable


class FakeCuda(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")


class FakeKrea:
    def __init__(self, n=28):
        self.blocks = [object() for _ in range(n)]


class FakePatcher:
    def __init__(self, inner, previous=None):
        self.inner = inner
        self.model_options = {"transformer_options": {"optimized_attention_override": previous} if previous else {}}
        self.wrappers = {}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.inner

    def clone(self):
        clone = FakePatcher(self.inner)
        clone.model_options = {k: dict(v) for k, v in self.model_options.items()}
        clone.wrappers = dict(self.wrappers)
        return clone

    def add_wrapper_with_key(self, kind, key, func):
        self.wrappers[(kind, key)] = func


@pytest.fixture(autouse=True)
def mock_core(monkeypatch):
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    ldm = ModuleType("comfy.ldm")
    ldm.__path__ = []
    krea = ModuleType("comfy.ldm.krea2")
    krea.__path__ = []
    model = ModuleType("comfy.ldm.krea2.model")
    model.SingleStreamDiT = FakeKrea
    pe = ModuleType("comfy.patcher_extension")
    pe.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer", DIFFUSION_MODEL="diff")
    for module in [comfy, ldm, krea, model, pe]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)


def qkv(n=4200, h=48, dtype=torch.bfloat16, gpu=True):
    x = torch.zeros(1, h, n, 128, dtype=dtype)
    return x.as_subclass(FakeCuda) if gpu else x


def options(n=4200, prefix=256, i=2, blocks=28):
    return {"transformer_options": {"img_slice": [prefix, n], "block_index": i,
            "block_type": "single", "total_blocks": blocks}, "_inside_attn_wrapper": True}


def run(patcher, func, *, opts=None, q=None, evaluate=2):
    outer = patcher.wrappers[("outer", krea2._KEY)]
    diff = patcher.wrappers[("diff", krea2._KEY)]
    patched = patcher.model_options["transformer_options"]["optimized_attention_override"]
    if q is None:
        q = qkv()
    if opts is None:
        opts = options(q.shape[2])

    def sample():
        result = None
        for _ in range(evaluate):
            def native():
                # The real native DiT reaches block 0 before later blocks.
                if opts["transformer_options"].get("block_index") != 0:
                    gate_options = options(q.shape[2], i=0)
                    patched(func, q, q, q, q.shape[1], mask=None, attn_precision=None,
                            skip_reshape=True, skip_output_reshape=False, **gate_options)
                return patched(func, q, q, q, q.shape[1], mask=None, attn_precision=None,
                               skip_reshape=True, skip_output_reshape=False, **opts)
            result = diff(native)
        return result

    with torch.no_grad():
        return outer(sample)


@pytest.mark.parametrize("kwargs", [
    {"tau": float("nan")}, {"tau": 3.1}, {"min_tokens": -1}, {"dense_layers": True},
    {"extra_tokens": 1}, {"extra_tokens": 257}, {"extra_tokens": True},
])
def test_settings_guards(kwargs):
    with pytest.raises(ValueError):
        krea2.Settings(**kwargs)


def test_wrong_model_type_rejected():
    with pytest.raises(TypeError, match="native ComfyUI Krea 2"):
        krea2.install(FakePatcher(object()), krea2.Settings())


def test_clone_and_upstream_owner_preserved():
    def previous(*args, **kw):
        return "prior"
    original = FakePatcher(FakeKrea(), previous)
    patched = krea2.install(original, krea2.Settings(), verbose=True)
    assert patched is not original
    assert original.model_options["transformer_options"]["optimized_attention_override"] is previous
    assert patched.model_options["transformer_options"][krea2._KEY]["api"] == 1
    assert run(patched, lambda *a, **kw: "native", opts=options(n=4200, i=0)) == "prior"
    with pytest.raises(ValueError, match="already installed"):
        krea2.install(patched, krea2.Settings())


def test_run_scope_resets_after_exception():
    patcher = krea2.install(FakePatcher(FakeKrea()), krea2.Settings())
    scope = patcher.wrappers[("outer", krea2._KEY)]
    with pytest.raises(RuntimeError, match="boom"):
        scope(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert krea2._ACTIVE.get() is None


def test_dense_evaluation_and_leading_layers_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(krea2, "sol_attention",
                        lambda q, k, v, prefix, settings, state: calls.append(prefix) or "sparse")
    patcher = krea2.install(FakePatcher(FakeKrea()), krea2.Settings())
    assert run(patcher, lambda *a, **kw: "native", evaluate=1) == "native"
    assert run(patcher, lambda *a, **kw: "native", opts=options(i=1), evaluate=2) == "native"
    assert run(patcher, lambda *a, **kw: "native", opts=options(i=2), evaluate=2) == "sparse"
    assert calls == [256]


def test_text_prefix_and_reference_tokens_preserved(monkeypatch):
    calls = []
    monkeypatch.setattr(krea2, "sol_attention",
                        lambda q, k, v, prefix, settings, state: calls.append((prefix, q.shape)) or "sparse")
    patcher = krea2.install(FakePatcher(FakeKrea()),
                           krea2.Settings(dense_evaluations=0, dense_layers=0, min_tokens=0))
    assert run(patcher, lambda *a, **kw: "native",
               opts=options(n=2000, prefix=300, i=27), q=qkv(2000), evaluate=1) == "sparse"
    assert calls == [(256, torch.Size([1, 48, 2000, 128])),
                     (300, torch.Size([1, 48, 2000, 128]))]


@pytest.mark.parametrize("overrides", [
    {"mask": torch.ones(4, 4)},
    {"q": lambda: qkv(50)},
    {"opts": lambda: options(n=4200, prefix=4201)},
    {"opts": lambda: {"transformer_options": {}, "_inside_attn_wrapper": True}},
    {"q": lambda: qkv(gpu=False)},
    {"opts": lambda: {**options(), "transformer_options":
                      {**options()["transformer_options"], "reference_image_num_tokens": [100]}}},
])
def test_safe_dense_fallbacks(monkeypatch, overrides):
    monkeypatch.setattr(krea2, "sol_attention", lambda *a: pytest.fail("sparse should not execute"))
    def prior(*args, **kwargs):
        return "prior"
    patcher = krea2.install(FakePatcher(FakeKrea(), prior),
                           krea2.Settings(dense_evaluations=0, dense_layers=0))
    q = overrides.get("q", lambda: qkv())()
    opts = overrides.get("opts", lambda: options(q.shape[2]))()
    if "mask" in overrides:
        opts["mask"] = overrides["mask"]
    outer = patcher.wrappers[("outer", krea2._KEY)]
    diff = patcher.wrappers[("diff", krea2._KEY)]
    patched = patcher.model_options["transformer_options"]["optimized_attention_override"]
    with torch.no_grad():
        assert outer(lambda: diff(
            lambda: patched(lambda *a, **kw: "native", q, q, q, q.shape[1],
                            skip_reshape=True, **opts),
        )) == "prior"


def test_native_fallback_for_unsupported_sol(monkeypatch):
    def unavailable(*args, **kwargs):
        raise KernelUnavailable("no SM120 kernel")
    monkeypatch.setattr(krea2, "sol_attention", unavailable)
    def previous(*args, **kw):
        return "sage"
    patcher = krea2.install(FakePatcher(FakeKrea(), previous),
                           krea2.Settings(dense_evaluations=0, dense_layers=0, min_tokens=0))
    assert run(patcher, lambda *a, **kw: "native", evaluate=1) == "sage"


def test_unexpected_kernel_failure_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("arithmetic gate failed")
    monkeypatch.setattr(krea2, "sol_attention", fail)
    patcher = krea2.install(FakePatcher(FakeKrea()),
                           krea2.Settings(dense_evaluations=0, dense_layers=0, min_tokens=0))
    with pytest.raises(RuntimeError, match="arithmetic gate failed"):
        run(patcher, lambda *a, **kw: "native", evaluate=1)


def test_inherited_preprocessing_is_not_bypassed(monkeypatch):
    monkeypatch.setattr(krea2, "sol_attention", lambda *a: pytest.fail("preprocessing was bypassed"))

    def previous(*args, **kwargs):
        return "prior"

    previous.attention_preprocess_v1 = (lambda *a, **k: None, None)
    patcher = krea2.install(
        FakePatcher(FakeKrea(), previous),
        krea2.Settings(dense_evaluations=0, dense_layers=0),
    )
    assert run(patcher, lambda *a, **kw: "native", evaluate=1) == "prior"


def test_configured_native_attention_is_not_shadowed():
    inner = FakeKrea()
    inner.blocks[0] = SimpleNamespace(
        attn=SimpleNamespace(comfy_attention=SimpleNamespace(function=object()))
    )
    with pytest.raises(RuntimeError, match="configured per-block"):
        krea2.install(FakePatcher(inner), krea2.Settings())



def test_forecasted_diffusion_wrapper_does_not_count_as_actual():
    patcher = krea2.install(FakePatcher(FakeKrea()), krea2.Settings())
    scope = patcher.wrappers[("outer", krea2._KEY)]
    diffusion = patcher.wrappers[("diff", krea2._KEY)]
    override = patcher.model_options["transformer_options"]["optimized_attention_override"]
    seen = []
    def sample():
        seen.append(krea2.spectrum_runtime_status(override)["actual_evaluations"])
        diffusion(lambda: "forecast")
        seen.append(krea2.spectrum_runtime_status(override)["actual_evaluations"])
        def native():
            q = qkv(40)
            return override(lambda *a, **kw: "native", q, q, q, 48,
                            skip_reshape=True, **options(n=40, prefix=4, i=0))
        diffusion(native)
        seen.append(krea2.spectrum_runtime_status(override)["actual_evaluations"])
    with torch.no_grad():
        scope(sample)
    assert seen == [0, 0, 1]
    assert krea2.spectrum_runtime_status(override) is None



def test_native_fp16_route_is_distinct_and_respects_warmup(monkeypatch):
    from sol_h3 import krea2_fp16
    calls = []
    monkeypatch.setattr(krea2_fp16, "attention",
                        lambda q, k, v, prefix, settings, state:
                        calls.append((prefix, settings.extra_tokens)) or "fp16_sparse")
    patcher = krea2.install(
        FakePatcher(FakeKrea()), krea2.Settings(extra_tokens=256),
    )
    assert run(patcher, lambda *a, **kw: "native", q=qkv(dtype=torch.float16),
               evaluate=2) == "fp16_sparse"
    assert calls == [(256, 256)]


def test_fp16_kernel_unavailable_preserves_inherited_dense_backend(monkeypatch):
    from sol_h3 import krea2_fp16
    def unavailable(*args):
        raise KernelUnavailable("no FP16 kitchen backend")
    monkeypatch.setattr(krea2_fp16, "attention", unavailable)
    patcher = krea2.install(
        FakePatcher(FakeKrea(), previous=lambda *a, **kw: "sage"),
        krea2.Settings(dense_evaluations=0),
    )
    assert run(patcher, lambda *a, **kw: "native",
               q=qkv(dtype=torch.float16), evaluate=1) == "sage"



def test_krea_widget_order_preserves_existing_workflows():
    from sol_h3.nodes import SolKrea2Experimental
    schema = SolKrea2Experimental.INPUT_TYPES()
    widget_order = [
        key for key in schema["required"] if key != "model"
    ] + list(schema["optional"])
    assert widget_order == [
        "enabled", "tau", "min_tokens", "dense_evaluations", "dense_layers",
        "verbose", "extra_tokens",
    ]
    assert schema["optional"]["extra_tokens"][1]["default"] == 256


@pytest.mark.parametrize("widgets, expected_verbose, expected_tokens", [
    # Pre-FP16 workflow: only six widgets, verbose must remain position six.
    ((True, 1.0, 4096, 1, 2, True), True, 256),
    # Swapped seven-widget layout in the first FP16 overlay.
    ((True, 1.0, 4096, 1, 2, 256, True), True, 256),
    ((True, 1.0, 4096, 1, 2, 128, False), False, 128),
    ((True, 1.0, 4096, 1, 2, True, False), True, 256),
    # Correct appended layout with an explicit FP16 token budget.
    ((True, 1.0, 4096, 1, 2, False, 192), False, 192),
])
def test_krea_saved_widget_layouts_migrate_without_disabling_sparse(
        monkeypatch, widgets, expected_verbose, expected_tokens):
    from sol_h3.nodes import SolKrea2Experimental
    seen = []
    monkeypatch.setattr(krea2, "install", lambda model, settings, *, verbose:
                        seen.append((settings, verbose)) or "patched")
    # Simulates widget values from serialized LiteGraph nodes, in schema order.
    model = object()
    result = SolKrea2Experimental().apply(model, *widgets)
    assert result == ("patched",)
    settings, verbose = seen[0]
    assert verbose is expected_verbose
    assert settings.extra_tokens == expected_tokens


def test_krea_explicit_invalid_token_budget_stays_rejected(monkeypatch):
    from sol_h3.nodes import SolKrea2Experimental
    monkeypatch.setattr(krea2, "install",
                        lambda *a, **kw: pytest.fail("invalid settings installed"))
    with pytest.raises(ValueError, match="extra_tokens"):
        SolKrea2Experimental().apply(object(), extra_tokens=42)
