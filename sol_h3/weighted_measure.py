from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import uuid


ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_CAPABILITIES_KEY = "attention_measure_capabilities_v1"
PROVIDER_IDENTITY = "comfy.sol_h3.sm120"
IMPLEMENTATION_PROFILE = "weighted_exact_blocks_v1"
NUMERICAL_ROUTE = "sol_h3_sm120_weighted_exact_blocks"


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


def preprocess_digest(provider):
    """Hash the concrete preprocessing/leaf ownership chain for this process.

    Object identity is intentional: a forecast or cached plan must not survive a
    runtime provider replacement merely because the replacement has the same name.
    """
    chain = []
    seen = set()
    current = provider
    while hasattr(current, "attention_preprocess_v1"):
        if id(current) in seen:
            raise RuntimeError("cyclic attention preprocessing contract")
        seen.add(id(current))
        transform, current = current.attention_preprocess_v1
        chain.append((_provider_name(transform), id(transform)))
    chain.append((_provider_name(current), id(current)))
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
        return core.bind(
            request,
            context=execution_context,
            block_size=64,
            implementation_profile=IMPLEMENTATION_PROFILE,
        )


def register(transformer_options):
    """Register when the companion core API exists; preserve legacy installations otherwise."""
    core = _core(required=False)
    if core is None:
        return None
    current = transformer_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    existing = current.get(PROVIDER_IDENTITY) if isinstance(current, dict) else None
    if existing is not None:
        if not isinstance(existing, Capability):
            raise RuntimeError("attention-measure provider identity is already owned by another capability")
        return existing
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
):
    """Bind one actual all-row weighted route and cache its O(T) bias per request."""
    core = _core()
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
        numerical_route=NUMERICAL_ROUTE,
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
    )
    plan = state.measure_plans.get(key)
    if plan is None:
        plan = core.prepare_capability(transformer_options, request, context)
        if plan.implementation_profile != IMPLEMENTATION_PROFILE:
            raise RuntimeError("Sol-H3 capability selected an unexpected attention-measure profile")
        bias_key = (digest, str(context.device))
        shared = state.measure_biases.get(bias_key)
        if shared is None:
            state.measure_biases[bias_key] = plan.key_log_measure
        elif shared is not plan.key_log_measure:
            plan = replace(plan, key_log_measure=shared)
        state.measure_plans[key] = plan
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
