"""Request-scoped host attribution for the unchanged packaged SM120 runtime."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import functools
import importlib
import threading
import time


HOOK_ABI = "sol_h3_sm120_compiler_attribution_v1"
_CURRENT = ContextVar("sol_h3_sm120_compiler_attribution", default=None)
_CUDA_DIAGNOSTIC = ContextVar("sol_h3_sm120_cuda_diagnostic", default=None)
_INSTALL_LOCK = threading.Lock()


def _add(telemetry, name, value):
    telemetry[name] = float(telemetry.get(name, 0.0)) + float(value)


class _DispatchProxy:
    __slots__ = ("target",)

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        telemetry = _CURRENT.get()
        diagnostic = _CUDA_DIAGNOSTIC.get()
        if telemetry is None and diagnostic is None:
            return self.target(*args, **kwargs)
        started = time.perf_counter() if telemetry is not None else None
        manager = (
            diagnostic[0].span(diagnostic[1], "compiled_dispatch")
            if diagnostic is not None
            else nullcontext()
        )
        try:
            with manager:
                return self.target(*args, **kwargs)
        finally:
            if telemetry is not None:
                _add(telemetry, "dispatch_host_enqueue_s", time.perf_counter() - started)

    def __getattr__(self, name):
        return getattr(self.target, name)


def _proxy(value):
    return value if isinstance(value, _DispatchProxy) else _DispatchProxy(value)


class _AttributedCache(dict):
    def __init__(self, source=()):
        self.destructive_generation = 0
        super().__init__((key, _proxy(value)) for key, value in dict(source).items())

    def get(self, key, default=None):
        value = super().get(key, default)
        telemetry = _CURRENT.get()
        if telemetry is not None:
            lookup = int(telemetry.get("_compile_cache_lookups", 0))
            telemetry["_compile_cache_lookups"] = lookup + 1
            telemetry["compiler_key"] = repr(key)
            hit = value is not default and value is not None
            if lookup == 0:
                telemetry["compile_cache_initial_hit"] = hit
                if hit:
                    telemetry["compile_hit"] = True
            elif hit and not telemetry.get("compile_hit"):
                telemetry["compile_race_hit"] = True
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.destructive_generation += 1
        super().__setitem__(key, _proxy(value))

    def __delitem__(self, key):
        if key in self:
            self.destructive_generation += 1
        super().__delitem__(key)

    def clear(self):
        if self:
            self.destructive_generation += 1
        super().clear()

    def pop(self, key, *args):
        if key in self:
            self.destructive_generation += 1
        return super().pop(key, *args)

    def popitem(self):
        if self:
            self.destructive_generation += 1
        return super().popitem()

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = value


class _TimedLock:
    __slots__ = ("target",)

    def __init__(self, target):
        self.target = target

    def acquire(self, *args, **kwargs):
        telemetry = _CURRENT.get()
        if telemetry is None:
            return self.target.acquire(*args, **kwargs)
        started = time.perf_counter()
        try:
            return self.target.acquire(*args, **kwargs)
        finally:
            _add(telemetry, "compile_lock_wait_s", time.perf_counter() - started)

    def release(self):
        return self.target.release()

    def locked(self):
        return self.target.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def __getattr__(self, name):
        return getattr(self.target, name)


@contextmanager
def attribution_scope(telemetry, *, diagnostic_state=None, diagnostic_sample=None):
    token = _CURRENT.set(telemetry)
    cuda_token = _CUDA_DIAGNOSTIC.set(
        None
        if diagnostic_state is None or diagnostic_sample is None
        else (diagnostic_state, diagnostic_sample)
    )
    try:
        yield
    finally:
        _CUDA_DIAGNOSTIC.reset(cuda_token)
        _CURRENT.reset(token)


def _refresh_runtime_objects(interface):
    if not isinstance(interface._compiled, _AttributedCache):
        interface._compiled = _AttributedCache(interface._compiled)
    if not isinstance(interface._compile_lock, _TimedLock):
        interface._compile_lock = _TimedLock(interface._compile_lock)


def install_hooks(interface):
    """Wrap process-global compiler objects once without changing reviewed source bytes."""
    if getattr(interface, "_sol_h3_attribution_abi", None) == HOOK_ABI:
        with _INSTALL_LOCK:
            _refresh_runtime_objects(interface)
            return compiler_namespace(interface)
    with _INSTALL_LOCK:
        if getattr(interface, "_sol_h3_attribution_abi", None) == HOOK_ABI:
            _refresh_runtime_objects(interface)
            return compiler_namespace(interface)

        original_compile = interface._compile_sm120
        preprocess = importlib.import_module(interface.__package__ + ".preprocess")
        original_prepare = preprocess.prepare
        _refresh_runtime_objects(interface)

        @functools.wraps(original_compile)
        def compile_sm120(*args, **kwargs):
            telemetry = _CURRENT.get()
            started = time.perf_counter() if telemetry is not None else None
            try:
                compiled, converted = original_compile(*args, **kwargs)
            finally:
                if telemetry is not None:
                    _add(telemetry, "compile_body_s", time.perf_counter() - started)
            key = args[0] if args else kwargs["key"]
            compiled = dict.get(interface._compiled, key, _proxy(compiled))
            if telemetry is not None:
                telemetry["compile_miss"] = True
                telemetry["compiler_key"] = repr(key)
            return compiled, converted

        @functools.wraps(original_prepare)
        def prepare(*args, **kwargs):
            telemetry = _CURRENT.get()
            diagnostic = _CUDA_DIAGNOSTIC.get()
            if telemetry is None and diagnostic is None:
                return original_prepare(*args, **kwargs)
            started = time.perf_counter() if telemetry is not None else None
            manager = (
                diagnostic[0].span(diagnostic[1], "prepare")
                if diagnostic is not None
                else nullcontext()
            )
            try:
                with manager:
                    return original_prepare(*args, **kwargs)
            finally:
                if telemetry is not None:
                    _add(telemetry, "prepare_jit_host_wall_s", time.perf_counter() - started)

        interface._compile_sm120 = compile_sm120
        preprocess.prepare = prepare
        interface._sol_h3_original_compile_sm120 = original_compile
        interface._sol_h3_original_prepare = original_prepare
        interface._sol_h3_attribution_abi = HOOK_ABI
        return compiler_namespace(interface)


def compiler_namespace(interface):
    preprocess = importlib.import_module(interface.__package__ + ".preprocess")
    original_compile = getattr(
        interface,
        "_sol_h3_original_compile_sm120",
        interface._compile_sm120,
    )
    original_prepare = getattr(
        interface,
        "_sol_h3_original_prepare",
        preprocess.prepare,
    )
    cache_generation = int(getattr(interface._compiled, "destructive_generation", 0))
    return (
        HOOK_ABI,
        id(interface._compiled),
        cache_generation,
        id(interface._compile_sm120),
        id(original_compile),
        id(preprocess.prepare),
        id(original_prepare),
        id(interface._compile_lock),
    )


__all__ = ["HOOK_ABI", "attribution_scope", "compiler_namespace", "install_hooks"]
