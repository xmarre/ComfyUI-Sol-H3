from __future__ import annotations

import builtins
from dataclasses import dataclass, replace
import dis
import hashlib
import importlib
import marshal
import threading
import types
import uuid
import weakref


ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_CAPABILITIES_KEY = "attention_measure_capabilities_v1"
PROVIDER_IDENTITY = "comfy.sol_h3.sm120"
SPARSE_IMPLEMENTATION_PROFILE = "weighted_exact_blocks_v1"
DENSE_IMPLEMENTATION_PROFILE = "dense_exact_v1"
SPARSE_NUMERICAL_ROUTE = "sol_h3_sm120_weighted_exact_blocks"
DENSE_NUMERICAL_ROUTE = "sol_h3_weighted_dense_exact"
# Backwards-compatible names used by the first provider tests and callers.
IMPLEMENTATION_PROFILE = SPARSE_IMPLEMENTATION_PROFILE
NUMERICAL_ROUTE = SPARSE_NUMERICAL_ROUTE
_ROUTE_PROFILES = {
    SPARSE_NUMERICAL_ROUTE: SPARSE_IMPLEMENTATION_PROFILE,
    DENSE_NUMERICAL_ROUTE: DENSE_IMPLEMENTATION_PROFILE,
}
_OWNER_GENERATIONS = {}
_OWNER_GENERATIONS_LOCK = threading.Lock()
_GLOBAL_IDENTITY_MAX_ITEMS = 64
_GLOBAL_IDENTITY_MAX_DEPTH = 4
_MISSING = object()


def _core(required=True):
    try:
        return importlib.import_module("comfy.attention_measure")
    except ModuleNotFoundError as exc:
        if exc.name not in {"comfy", "comfy.attention_measure"}:
            raise
        if required:
            raise RuntimeError(
                "attention_measure_v1 requires a ComfyUI build with the generic attention-measure contract"
            ) from exc
        return None


def _provider_name(provider):
    if provider is None:
        return "comfy.default"
    return f"{getattr(provider, '__module__', '<unknown>')}.{getattr(provider, '__qualname__', type(provider).__name__)}"


def _code_digest(code):
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _loaded_global_names(code):
    return tuple(
        sorted(
            {
                instruction.argval
                for instruction in dis.get_instructions(code)
                if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"} and isinstance(instruction.argval, str)
            }
        )
    )


def _semantic_value_identity(value, *, depth, seen):
    """Return bounded semantic identity for code-reachable global state.

    Unsupported state deliberately returns ``None``. Callers then use a volatile
    identity rather than reusing a weighted plan whose preprocessing semantics
    cannot be proven stable.
    """
    if value is None or type(value) in {bool, int, float, complex, str, bytes}:
        return (type(value).__name__, value)
    if isinstance(value, types.ModuleType):
        return ("module_v1", getattr(value, "__name__", "<unknown>"))
    if isinstance(value, type):
        return ("type_v1", getattr(value, "__module__", "<unknown>"), getattr(value, "__qualname__", repr(value)))
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return (
            "builtin_v1",
            getattr(value, "__module__", "builtins"),
            getattr(value, "__qualname__", getattr(value, "__name__", repr(value))),
        )

    if isinstance(value, types.FunctionType):
        code = getattr(value, "__code__", None)
        if code is None:
            return None
        base = ("function_v2", _provider_name(value), _code_digest(code))
        if depth <= 0:
            return base
        owner_id = id(value)
        if owner_id in seen:
            return base + (("cycle",),)
        nested_seen = set(seen)
        nested_seen.add(owner_id)

        defaults = getattr(value, "__defaults__", None)
        defaults_identity = _semantic_value_identity(defaults, depth=depth - 1, seen=nested_seen)
        if defaults is not None and defaults_identity is None:
            return None
        kwdefaults = getattr(value, "__kwdefaults__", None)
        kwdefaults_identity = _semantic_value_identity(kwdefaults, depth=depth - 1, seen=nested_seen)
        if kwdefaults is not None and kwdefaults_identity is None:
            return None
        closure = getattr(value, "__closure__", None)
        if closure is None:
            closure_identity = None
        else:
            closure_values = []
            for cell in closure:
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    return None
                cell_identity = _semantic_value_identity(cell_value, depth=depth - 1, seen=nested_seen)
                if cell_identity is None:
                    return None
                closure_values.append(cell_identity)
            closure_identity = tuple(closure_values)

        globals_dict = getattr(value, "__globals__", None)
        if not isinstance(globals_dict, dict):
            return None
        global_identity = []
        for name in _loaded_global_names(code):
            if name in globals_dict:
                source = "global"
                dependency = globals_dict[name]
            else:
                dependency = getattr(builtins, name, _MISSING)
                if dependency is _MISSING:
                    return None
                source = "builtin"
            dependency_identity = _semantic_value_identity(
                dependency,
                depth=depth - 1,
                seen=nested_seen,
            )
            if dependency_identity is None:
                return None
            global_identity.append((source, name, dependency_identity))
        return base + (defaults_identity, kwdefaults_identity, closure_identity, tuple(global_identity))

    if isinstance(value, (tuple, list, frozenset, set)):
        if len(value) > _GLOBAL_IDENTITY_MAX_ITEMS:
            return None
        owner_id = id(value)
        if owner_id in seen:
            return ("container_cycle_v1", type(value).__name__)
        nested_seen = set(seen)
        nested_seen.add(owner_id)
        items = []
        for item in value:
            identity = _semantic_value_identity(item, depth=depth - 1, seen=nested_seen)
            if identity is None:
                return None
            items.append(identity)
        if isinstance(value, (set, frozenset)):
            items.sort(key=repr)
        return (type(value).__name__, tuple(items))

    if isinstance(value, dict):
        if len(value) > _GLOBAL_IDENTITY_MAX_ITEMS:
            return None
        owner_id = id(value)
        if owner_id in seen:
            return ("dict_cycle_v1",)
        nested_seen = set(seen)
        nested_seen.add(owner_id)
        items = []
        for key, item in value.items():
            key_identity = _semantic_value_identity(key, depth=depth - 1, seen=nested_seen)
            item_identity = _semantic_value_identity(item, depth=depth - 1, seen=nested_seen)
            if key_identity is None or item_identity is None:
                return None
            items.append((key_identity, item_identity))
        items.sort(key=repr)
        return ("dict", tuple(items))

    return None


def _stateless_function_identity(provider):
    """Return stable code+global identity for a provably stateless function.

    Fresh factory-created preprocess functions should share one request-cache
    identity when their numerical semantics are equal. Code bytes alone are not
    sufficient because ``LOAD_GLOBAL`` can observe mutable module state. We
    therefore include a bounded semantic snapshot of code-reachable globals and
    helper functions. If any reachable value cannot be represented safely, the
    identity becomes deliberately volatile so stale weighted plans cannot be
    reused.
    """
    code = getattr(provider, "__code__", None)
    if code is None:
        return None
    if getattr(provider, "__closure__", None) is not None:
        return None
    if getattr(provider, "__defaults__", None):
        return None
    if getattr(provider, "__kwdefaults__", None):
        return None
    code_digest = _code_digest(code)
    semantic = _semantic_value_identity(
        provider,
        depth=_GLOBAL_IDENTITY_MAX_DEPTH,
        seen=set(),
    )
    if semantic is None:
        return ("volatile_global_function_v1", _provider_name(provider), code_digest, uuid.uuid4().hex)
    semantic_digest = hashlib.sha256(repr(semantic).encode("utf-8")).hexdigest()
    return ("stateless_function_v2", _provider_name(provider), code_digest, semantic_digest)


def _owner_identity(provider):
    if provider is None:
        return ("singleton_v1", "comfy.default")
    stable = _stateless_function_identity(provider)
    if stable is not None:
        return stable

    # Stateful callable identity must not be represented by raw id(obj): the
    # object may die while a digest remains in a request cache and CPython may
    # later reuse that address for a different owner. A weakref-bound generation
    # token changes on replacement without globally retaining the provider.
    try:
        weakref.ref(provider)
    except TypeError as exc:
        raise RuntimeError(
            "stateful attention preprocessing/provider owners must support weak references"
        ) from exc

    key = id(provider)
    with _OWNER_GENERATIONS_LOCK:
        current = _OWNER_GENERATIONS.get(key)
        if current is not None and current[0]() is provider:
            token = current[1]
        else:
            token = uuid.uuid4().hex

            def cleanup(ref, *, owner_id=key, generation=token):
                with _OWNER_GENERATIONS_LOCK:
                    retained = _OWNER_GENERATIONS.get(owner_id)
                    if retained is not None and retained[0] is ref and retained[1] == generation:
                        _OWNER_GENERATIONS.pop(owner_id, None)

            _OWNER_GENERATIONS[key] = (weakref.ref(provider, cleanup), token)
    return ("owner_generation_v1", _provider_name(provider), token)


def preprocess_digest(provider):
    """Hash numerical preprocessing semantics plus concrete stateful ownership.

    Closure-free/default-free function preprocessors use code plus bounded
    code-reachable global semantics, allowing equivalent factory-created wrappers
    to share one identity without ignoring ``LOAD_GLOBAL`` state. Unsupported
    reachable global state is deliberately volatile. Stateful callables retain
    concrete weakref-bound generations so replacement cannot inherit a cached
    plan through CPython address reuse.
    """
    chain = []
    seen = set()
    current = provider
    while hasattr(current, "attention_preprocess_v1"):
        if id(current) in seen:
            raise RuntimeError("cyclic attention preprocessing contract")
        seen.add(id(current))
        transform, current = current.attention_preprocess_v1
        chain.append(_owner_identity(transform))
    chain.append(_owner_identity(current))
    return hashlib.sha256(repr(tuple(chain)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderOwner:
    generation: str


class Capability:
    api = 1
    operator = "key_log_measure"
    provider_identity = PROVIDER_IDENTITY

    def __init__(self, owner):
        self.owner = owner

    def prepare(self, request, execution_context):
        core = _core()
        if not isinstance(execution_context, core.MeasureExecutionContext):
            raise TypeError("Sol-H3 attention measure requires MeasureExecutionContext")
        if execution_context.owner is not self.owner:
            raise RuntimeError("Sol-H3 attention-measure capability is bound to another owner")
        if execution_context.provider_identity != PROVIDER_IDENTITY:
            raise RuntimeError("Sol-H3 attention-measure provider identity changed during binding")
        implementation_profile = _ROUTE_PROFILES.get(execution_context.numerical_route)
        if implementation_profile is None:
            raise RuntimeError("Sol-H3 attention-measure numerical route is not capability-bound")
        return core.bind(
            request,
            context=execution_context,
            block_size=64,
            implementation_profile=implementation_profile,
        )


def register(transformer_options, *, refresh_owned=False):
    """Register a model-local capability when the companion core API exists.

    ``MODEL.clone()`` can carry a shallow copy of the previous capability
    registry. A new Sol-H3 installation owns a new replacement chain, so it must
    receive a fresh owner generation rather than silently reusing the source
    model's capability. Only a capability created by this module may be
    refreshed; another owner under the same provider identity remains an error.
    """
    core = _core(required=False)
    if core is None:
        return None
    current = transformer_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    existing = current.get(PROVIDER_IDENTITY) if isinstance(current, dict) else None
    if existing is not None:
        if not isinstance(existing, Capability):
            raise RuntimeError("attention-measure provider identity is already owned by another capability")
        if not refresh_owned:
            return existing
        registry = dict(current)
        del registry[PROVIDER_IDENTITY]
        transformer_options[ATTENTION_MEASURE_CAPABILITIES_KEY] = registry
    capability = Capability(ProviderOwner(f"sol-h3-{uuid.uuid4().hex}"))
    core.register_capability(transformer_options, PROVIDER_IDENTITY, capability)
    return capability


def prepare(
    state,
    transformer_options,
    request,
    *,
    block_index,
    layout,
    q_rows,
    kv_rows,
    dtype,
    device,
    head_dim,
    existing_sink,
    external_sequence,
    preprocess_identity,
    implementation_profile=SPARSE_IMPLEMENTATION_PROFILE,
    numerical_route=SPARSE_NUMERICAL_ROUTE,
):
    """Bind one actual all-row weighted route and cache its O(T) bias per request."""
    core = _core()
    expected_profile = _ROUTE_PROFILES.get(numerical_route)
    if expected_profile is None or expected_profile != implementation_profile:
        raise ValueError("Sol-H3 attention-measure route/profile pair is invalid")
    registry = transformer_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    capability = registry.get(PROVIDER_IDENTITY) if isinstance(registry, dict) else None
    if not isinstance(capability, Capability):
        raise RuntimeError(
            "attention_measure_v1 selected Sol-H3 but its owner-bound capability is not registered"
        )
    digest = core.semantic_digest(request)
    context = core.MeasureExecutionContext(
        provider_identity=PROVIDER_IDENTITY,
        block_index=int(block_index),
        owner=capability.owner,
        owner_generation=capability.owner.generation,
        layout=layout,
        q_rows=int(q_rows),
        kv_rows=int(kv_rows),
        dtype=dtype,
        device=device,
        head_dim=int(head_dim),
        mask_class="none",
        preprocess_digest=str(preprocess_identity),
        numerical_route=numerical_route,
        existing_sink=tuple(existing_sink),
        external_sequence=external_sequence,
    )
    key = (
        digest,
        context.block_index,
        context.q_rows,
        context.kv_rows,
        str(context.device),
        str(context.dtype),
        context.head_dim,
        context.existing_sink,
        context.preprocess_digest,
        context.owner_generation,
        implementation_profile,
        numerical_route,
    )
    plan = state.measure_plans.get(key)
    if plan is None:
        plan = core.prepare_capability(transformer_options, request, context)
        if plan.implementation_profile != implementation_profile or plan.numerical_route != numerical_route:
            raise RuntimeError("Sol-H3 capability selected an unexpected attention-measure route/profile")
        bias_key = (digest, str(context.device))
        shared = state.measure_biases.get(bias_key)
        if shared is None:
            state.measure_biases[bias_key] = plan.key_log_measure
        elif shared is not plan.key_log_measure:
            plan = replace(plan, key_log_measure=shared)
        state.measure_plans[key] = plan
    else:
        # The plan cache intentionally excludes transient layout/external objects
        # so identical geometry can reuse the O(T) measure buffer. Revalidate the
        # current objects on every cache hit; otherwise a stale/foreign VDN
        # external-sequence contract or changed H3 layout could bypass core's
        # fail-closed geometry checks merely because the semantic request digest
        # and numerical route stayed the same.
        core.validate_h3(
            request,
            layout=context.layout,
            q_rows=context.q_rows,
            kv_rows=context.kv_rows,
            external_sequence=context.external_sequence,
        )
        exact_k_block_range = core.merge_exact_k_blocks(
            request,
            64,
            context.existing_sink,
        )
        if exact_k_block_range != plan.exact_k_block_range:
            raise RuntimeError("cached Sol-H3 attention-measure exact K range is stale")
    return plan


def dense(q, k, v, heads, plan, *, scale=None, output_heads=False):
    """Exact all-row weighted fallback over already-preprocessed BHND tensors."""
    core = _core()
    return core.weighted_dense(
        q,
        k,
        v,
        heads,
        key_bias=plan.key_log_measure,
        skip_reshape=True,
        skip_output_reshape=bool(output_heads),
        scale=scale,
    )


def receipt_fields(plan, *, call_token):
    return (
        ATTENTION_MEASURE_KEY,
        ("sol_h3_evaluation", int(call_token)),
        plan.owner_generation,
        plan.semantic_digest,
        plan.implementation_profile,
        plan.numerical_route,
        plan.q_rows,
        plan.kv_rows,
        plan.exact_range_digest,
        plan.preprocess_digest,
        True,
    )
