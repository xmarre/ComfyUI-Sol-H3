"""Generic Mixed-Grid key-measure binding for the Sol-H3 SM120 provider.

The schema and bound-plan implementation live in ComfyUI core.  This module
only owns the Sol-H3 provider capability and the provider-specific exact-bias
interval derived from a core-validated plan.  Legacy representative K/V
selection remains in :mod:`sol_h3.mixed_measure` and never enters this path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import uuid


ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_CAPABILITIES_KEY = "attention_measure_capabilities_v1"
PROVIDER_IDENTITY = "xmarre.comfyui_sol_h3.sm120"
NUMERICAL_ROUTE = "sol_h3_sm120_weighted_exact_blocks_v1"
IMPLEMENTATION_PROFILE = "weighted_exact_blocks_v1"
PREPROCESS_POLICY = "sol_h3_full_domain_preprocess_v1"


def _core():
    try:
        from comfy import attention_measure as core
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "attention_measure_v1 requires a ComfyUI build with the generic attention-measure API"
        ) from exc
    return core


def preprocess_digest(identity) -> str:
    """Hash only the already-resolved preprocessing numerical identity."""
    return hashlib.sha256(
        (PREPROCESS_POLICY + "|" + repr(identity)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedMeasure:
    plan: object
    bias_start: int
    bias_stop: int
    exact_rows: int

    @property
    def semantic_digest(self) -> str:
        return self.plan.semantic_digest

    @property
    def key_log_measure(self):
        return self.plan.key_log_measure

    def receipt_fields(self):
        return (
            ("owner_generation", self.plan.owner_generation),
            ("semantic_digest", self.plan.semantic_digest),
            ("implementation_profile", self.plan.implementation_profile),
            ("numerical_route", self.plan.numerical_route),
            ("q_rows", self.plan.q_rows),
            ("kv_rows", self.plan.kv_rows),
            ("exact_range_digest", self.plan.exact_range_digest),
            ("preprocess_digest", self.plan.preprocess_digest),
            ("completed", True),
        )


class SolMeasureCapability:
    api = 1
    operator = "key_log_measure"
    provider_identity = PROVIDER_IDENTITY

    def __init__(self, owners):
        self._owners = dict(owners)
        self._generation = f"sol-h3-measure-{uuid.uuid4().hex}"

    def generation_for(self, owner, block_index: int) -> str:
        if type(block_index) is not int or self._owners.get(block_index) is not owner:
            raise RuntimeError("Sol-H3 attention-measure capability is bound to another block owner")
        return f"{self._generation}:{block_index}"

    def prepare(self, request, execution_context):
        core = _core()
        if not isinstance(execution_context, core.MeasureExecutionContext):
            raise TypeError("Sol-H3 attention measure requires MeasureExecutionContext")
        if execution_context.provider_identity != PROVIDER_IDENTITY:
            raise RuntimeError("Sol-H3 attention-measure provider identity mismatch")
        expected_generation = self.generation_for(
            execution_context.owner, execution_context.block_index
        )
        if execution_context.owner_generation != expected_generation:
            raise RuntimeError("Sol-H3 attention-measure owner generation is stale")
        if execution_context.numerical_route != NUMERICAL_ROUTE:
            raise RuntimeError("Sol-H3 attention measure selected an unexpected numerical route")
        if execution_context.mask_class != "none":
            raise RuntimeError("Sol-H3 weighted sparse attention does not accept an attention mask")
        return core.bind(
            request,
            context=execution_context,
            block_size=64,
            implementation_profile=IMPLEMENTATION_PROFILE,
        )


def register(options, owners):
    """Publish one owner-bound Sol capability without disturbing other providers.

    Current released ComfyUI builds do not yet expose the generic API.  Normal
    unweighted Sol-H3 installation therefore remains valid; the capability is
    registered only when that API exists.  A later attention_measure_v1 request
    still fails explicitly in :func:`prepare` if the companion core is absent.
    """
    try:
        core = _core()
    except RuntimeError:
        return None
    capability = SolMeasureCapability(owners)
    current = options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    registry = dict(current) if isinstance(current, dict) else {}
    existing = registry.get(PROVIDER_IDENTITY)
    if existing is not None and not isinstance(existing, SolMeasureCapability):
        raise RuntimeError("Sol-H3 attention-measure provider identity already has another owner")
    registry.pop(PROVIDER_IDENTITY, None)
    options[ATTENTION_MEASURE_CAPABILITIES_KEY] = registry
    core.register_capability(options, PROVIDER_IDENTITY, capability)
    return capability


def _weighted_interval(core, request):
    normalized = core.normalize(request)
    weighted = [
        segment
        for segment in normalized["segments"]
        if (segment["mass_num"], segment["mass_den"]) != (1, 1)
    ]
    if len(weighted) != 1:
        raise RuntimeError("Sol-H3 Mixed-Grid measure requires one contiguous non-unit key interval")
    segment = weighted[0]
    expected_start = normalized["video_start"]
    expected_stop = expected_start + normalized["prefix_t"] * (
        normalized["prefix_grid"][0] * normalized["prefix_grid"][1]
    )
    if (segment["start"], segment["stop"]) != (expected_start, expected_stop):
        raise RuntimeError("Sol-H3 Mixed-Grid key-measure interval does not match the protected prefix")
    return normalized, int(segment["start"]), int(segment["stop"])


def prepare(
    options,
    request,
    *,
    owner,
    block_index: int,
    layout,
    q_rows: int,
    kv_rows: int,
    dtype,
    device,
    head_dim: int,
    existing_sink,
    external_sequence,
    preprocess_identity,
) -> PreparedMeasure:
    core = _core()
    registry = options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    capability = registry.get(PROVIDER_IDENTITY) if isinstance(registry, dict) else None
    if not isinstance(capability, SolMeasureCapability):
        raise RuntimeError("attention_measure_v1 selected Sol-H3 without its owner-bound capability")
    generation = capability.generation_for(owner, block_index)
    digest = preprocess_digest(preprocess_identity)
    context = core.MeasureExecutionContext(
        provider_identity=PROVIDER_IDENTITY,
        block_index=int(block_index),
        owner=owner,
        owner_generation=generation,
        layout=layout,
        q_rows=int(q_rows),
        kv_rows=int(kv_rows),
        dtype=dtype,
        device=device,
        head_dim=int(head_dim),
        mask_class="none",
        preprocess_digest=digest,
        numerical_route=NUMERICAL_ROUTE,
        existing_sink=tuple(existing_sink),
        external_sequence=external_sequence,
    )
    plan = core.prepare_capability(options, request, context)
    if plan.implementation_profile != IMPLEMENTATION_PROFILE:
        raise RuntimeError("Sol-H3 attention-measure capability selected an unexpected profile")
    normalized, bias_start, bias_stop = _weighted_interval(core, request)
    if plan.exact_k_block_range[0] != 0:
        raise RuntimeError("Sol-H3 Mixed-Grid measure requires one exact prefix K-block range")
    exact_rows = min(
        int(normalized["kv_rows"]),
        int(plan.exact_k_block_range[1]) * 64,
    )
    if exact_rows < bias_stop:
        raise RuntimeError("Sol-H3 exact K-block range does not cover the non-unit key measure")
    return PreparedMeasure(plan, bias_start, bias_stop, exact_rows)


def history_identity(options):
    """Return stable measure identity for Spectrum preflight, or None when absent."""
    request = options.get(ATTENTION_MEASURE_KEY)
    if request is None:
        return None
    core = _core()
    registry = options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    capability = registry.get(PROVIDER_IDENTITY) if isinstance(registry, dict) else None
    if not isinstance(capability, SolMeasureCapability):
        return None
    normalized, bias_start, bias_stop = _weighted_interval(core, request)
    return (
        PROVIDER_IDENTITY,
        core.semantic_digest(normalized),
        IMPLEMENTATION_PROFILE,
        NUMERICAL_ROUTE,
        bias_start,
        bias_stop,
        capability._generation,
    )
