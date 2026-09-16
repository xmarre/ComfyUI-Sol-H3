from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import marshal
import threading
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


def _stateless_function_identity(provider):
    """Return a stable implementation identity for a genuinely stateless function.

    Attention preprocess factories may create a fresh function object for every
    model invocation. Object identity is not numerical identity in that case and
    would grow the request-scoped weighted-plan cache once per evaluation. Only
    functions with no closure/default state qualify for code identity; anything
    stateful falls back to concrete owner-generation tracking below.
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
    digest = hashlib.sha256(marshal.dumps(code)).hexdigest()
    return ("stateless_function_v1", _provider_name(provider), digest)


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

    Closure-free/default-free function preprocessors are identified by immutable
    code so equivalent factory-created wrappers share one numerical identity.
    Stateful callables retain concrete weakref-bound generations so replacement
    cannot inherit a cached plan through CPython address reuse.
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
