"""Small duck-typed contracts shared with optional attention/forecast providers."""
from dataclasses import dataclass
import struct

HISTORY_KEY = "attention_backend_history_v1"
RECEIPTS_KEY = "attention_backend_receipts_v1"
VDN_KEY = "vdn_softmax_provider_v1"
VDN_KEY_V2 = "vdn_softmax_provider_v2"
VDN_KEY_V3 = "vdn_softmax_provider_v3"
VDN_PREPROCESS_KEY = "vdn_attention_preprocess_v1"
SPECTRUM_EXTERNAL_RUNTIME_KEY = "spectrum_h3_external_patch_runtime"
SPECTRUM_RUNTIME_KEY = "spectrum_h3_runtime"
_SPECTRUM_EXTERNAL_STATE_ATTR = "_spectrum_h3_external_patch_compat"


def provider_name(provider):
    if provider is None:
        return "comfy.default"
    return f"{getattr(provider, '__module__', '<unknown>')}.{getattr(provider, '__qualname__', type(provider).__name__)}"


def receipt(options, block, route):
    sink = options.get(RECEIPTS_KEY)
    if sink is not None:
        sink.append(("sol_h3", block, route))


def _freeze_history_value(value):
    """Turn small provider config/state into a stable hashable history identity."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return tuple(sorted(
            (str(key), _freeze_history_value(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_history_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_history_value(item) for item in value), key=repr))
    return repr(value)


def _closure_values(function):
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None)
    if code is None or cells is None or len(code.co_freevars) != len(cells):
        return None
    values = {}
    for name, cell in zip(code.co_freevars, cells):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            return None
    return values


def _vdn_history_identity(forward, options, layout):
    """Describe known VDN routing without requiring VDN to own hybrid.py metadata.

    Newer/older VDN builds may publish ``attention_history_v1`` directly; prefer it.
    The stack-compatible VDN overlay deliberately does not edit ``hybrid.py`` because
    the audio-fidelity overlay owns that file. For the audited closure shape, grouped
    routing is still deterministic from its captured config/backend, so expose that
    identity here. Any unknown closure or Flex routing remains opaque/actual-only.
    """
    describe = getattr(forward, "attention_history_v1", None)
    if callable(describe):
        return describe(options, layout)

    values = _closure_values(forward)
    if values is None:
        return None
    state = values.get("state")
    cfg = values.get("cfg", getattr(state, "cfg", None))
    base_branch = values.get("base_branch")
    backend = getattr(state, "softmax_backend", None)
    if backend != "grouped" or not isinstance(cfg, dict):
        # Flex can fail into grouped at runtime, so its next numerical route is not
        # preflight-provable. Unknown backends likewise execute actuals rather than
        # being rejected or forecast across an opaque transition.
        return None
    if state is None or getattr(state, "cfg", cfg) is not cfg:
        return None

    branch_identity = (
        "none" if base_branch is None else provider_name(type(base_branch)),
        bool(getattr(base_branch, "enable_text_state", False)) if base_branch is not None else False,
    )
    return (
        "vdn_h3_grouped_closure_v1",
        id(forward),
        _freeze_history_value(cfg),
        branch_identity,
    )


def _diffaid_runtime_identity(options):
    """Return stable Diff-Aid instance identities published to transformer options."""
    raw = options.get(SPECTRUM_EXTERNAL_RUNTIME_KEY)
    if raw is None:
        return ()
    entries = raw if isinstance(raw, (tuple, list)) else (raw,)
    identities = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("provider") != "comfyui-diffaid-patches" or entry.get("schema_version") != 1:
            continue
        instance = entry.get("instance_id")
        if not isinstance(instance, str) or not instance:
            continue
        identities.append(("comfyui-diffaid-patches", 1, instance))
    return tuple(sorted(set(identities)))


def _float32_scalar(value):
    """Match Diff-Aid's hard-window descriptor rounding without a tensor sync."""
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


def _diffaid_config_identity(config):
    fields = (
        "strength", "sigma_start", "sigma_end", "sigma_ramp",
        "token_weight_mode", "token_tail", "cond_only",
    )
    if config is None or any(not hasattr(config, name) for name in fields):
        return None
    ramp = float(getattr(config, "sigma_ramp"))
    start = float(getattr(config, "sigma_start"))
    end = float(getattr(config, "sigma_end"))
    if ramp == 0.0:
        start = _float32_scalar(start)
        end = _float32_scalar(end)
    return (
        ("strength", _freeze_history_value(float(getattr(config, "strength")))),
        ("sigma_start", _freeze_history_value(start)),
        ("sigma_end", _freeze_history_value(end)),
        ("sigma_ramp", _freeze_history_value(ramp)),
        ("token_weight_mode", _freeze_history_value(str(getattr(config, "token_weight_mode")))),
        ("token_tail", _freeze_history_value(float(getattr(config, "token_tail")))),
        ("cond_only", _freeze_history_value(bool(getattr(config, "cond_only")))),
    )


def _diffaid_spectrum_contract_identity(options, block_index, config_identity):
    """Resolve the already-validated Spectrum Diff-Aid declaration at preflight time.

    Spectrum configures external-patch descriptors on its runtime before model calls.
    Diff-Aid's per-call runtime entry can be injected later by the legacy model
    wrapper, so requiring only that entry makes backend-history preflight depend on
    wrapper ordering. The configured Spectrum descriptor is the same contract owner
    and is available before the DIFFUSION_MODEL wrapper asks for history identity.
    """
    runtime = options.get(SPECTRUM_RUNTIME_KEY)
    compat = getattr(runtime, _SPECTRUM_EXTERNAL_STATE_ATTR, None)
    parsed = getattr(compat, "parsed", None)
    descriptors = getattr(parsed, "descriptors", ())
    matches = []
    for descriptor in descriptors if isinstance(descriptors, (tuple, list)) else ():
        if (
            getattr(descriptor, "provider", None) != "comfyui-diffaid-patches"
            or getattr(descriptor, "schema_version", None) != 1
        ):
            continue
        instance = getattr(descriptor, "instance_id", None)
        indices = getattr(descriptor, "block_indices_0based", ())
        if not isinstance(instance, str) or not instance or block_index not in indices:
            continue
        if _diffaid_config_identity(descriptor) != config_identity:
            continue
        matches.append(("comfyui-diffaid-patches", 1, instance))
    identities = tuple(sorted(set(matches)))
    return identities if len(identities) == 1 else ()


def _diffaid_replacement_identity(patch, options, block_index):
    """Recognize the audited MiniMax-H3 Diff-Aid activation-only block wrapper.

    The preferred proof is Spectrum's parsed static external-patch declaration,
    which exists before per-call legacy wrappers inject runtime coordinates. A
    dynamic Diff-Aid declaration is still accepted for older compatible stacks.
    If both are present they must agree on the instance identity. This preserves
    fail-closed behavior without making history safety depend on wrapper order.
    """
    if type(patch).__name__ != "MiniMaxH3BlockReplacePatch" or not hasattr(patch, "existing_patch"):
        return None
    config_identity = _diffaid_config_identity(getattr(patch, "config", None))
    if config_identity is None:
        return None
    contract_identity = _diffaid_spectrum_contract_identity(
        options,
        int(block_index),
        config_identity,
    )
    runtime_identity = _diffaid_runtime_identity(options)
    if contract_identity:
        if runtime_identity and not all(item in runtime_identity for item in contract_identity):
            return None
        declaration_identity = contract_identity
    else:
        declaration_identity = runtime_identity
    if not declaration_identity:
        return None
    return (
        ("spectrum_declared_diffaid_h3_v1", declaration_identity, config_identity),
        patch.existing_patch,
    )


def _flow_mixed_grid_replacement_identity(patch, block_index):
    """Recognize Flow v0.3.x's audited exact-prefix mixed-grid block wrapper.

    Flow constructs one wrapper per H3 block immediately outside the existing DiT
    replacement chain. The wrapper deterministically changes only the block-local
    packed geometry/RoPE/modulation metadata, publishes VDN external-sequence API 2,
    and then delegates to the previous replacement. Spectrum preflights before the
    wrapper executes, so treating this known closure as opaque would turn every
    scheduler forecast into an actual transformer NFE. Describe the geometry here
    without executing the wrapper; unknown or malformed closures remain opaque.
    """
    module = str(getattr(patch, "__module__", ""))
    qualname = str(getattr(patch, "__qualname__", ""))
    if not (
        (module == "h3_flow_regenerate.mixed_grid" or module.endswith(".h3_flow_regenerate.mixed_grid"))
        and qualname.endswith("mixed_diffusion_wrapper.<locals>.wrap.<locals>.call")
    ):
        return None

    values = _closure_values(patch)
    required = {"layer", "previous", "plan", "layout", "mixed_layout", "va", "vb", "old_prefix", "inner"}
    if values is None or not required.issubset(values):
        return None
    if type(values["layer"]) is not int or values["layer"] != int(block_index):
        return None

    plan = values["plan"]
    layout = values["layout"]
    mixed_layout = values["mixed_layout"]
    inner = values["inner"]
    try:
        prefix_t = int(plan.prefix_t)
        temporal = int(plan.temporal)
        source_rows = int(plan.source_rows)
        target_rows = int(plan.target_rows)
        prefix_rows = int(plan.prefix_rows)
        mixed_rows = int(plan.mixed_rows)
        target_hw = tuple(int(value) for value in plan.target_hw)
        va = int(values["va"])
        vb = int(values["vb"])
        old_prefix = int(values["old_prefix"])
        native_rows = int(layout.seq_len)
        mixed_sequence_rows = int(mixed_layout.seq_len)
        native_segments = tuple(layout.segments)
        mixed_segments = tuple(mixed_layout.segments)
        inner_blocks = len(inner.blocks)
    except (AttributeError, TypeError, ValueError):
        return None

    if (
        not 0 < prefix_t < temporal
        or source_rows <= 0
        or target_rows <= source_rows
        or len(target_hw) != 2
        or any(value <= 0 or value % 2 for value in target_hw)
        or prefix_rows != prefix_t * target_rows
        or old_prefix != prefix_t * source_rows
        or mixed_rows != prefix_rows + (temporal - prefix_t) * source_rows
        or va < 0
        or vb != va + temporal * source_rows
        or native_rows != vb
        or mixed_sequence_rows != va + mixed_rows
        or not native_segments
        or native_segments[-1] != (va, vb, "video")
        or not mixed_segments
        or mixed_segments[-1] != (va, mixed_sequence_rows, "video")
        or not isinstance(getattr(mixed_layout, "signature", None), tuple)
        or not mixed_layout.signature
        or mixed_layout.signature[0] != "h3_flow_mixed_grid_v1"
        or block_index < 0
        or block_index >= inner_blocks
    ):
        return None

    identity = (
        "h3_flow_mixed_grid_v1",
        native_rows,
        mixed_sequence_rows,
        va,
        temporal,
        prefix_t,
        source_rows,
        target_rows,
        target_hw,
        repr(getattr(layout, "signature", None)),
        repr(mixed_layout.signature),
    )
    return identity, values["previous"]


def _replacement_history_identity(patch, block_index, config, options):
    """Describe one proven-transparent replacement chain containing Sol-H3 once.

    Unknown replacement wrappers remain opaque. This is intentionally stricter
    than merely observing a previous successful receipt: a forecast skips the
    transformer and therefore cannot discover a later wrapper-owned route change.
    """
    from .runtime import BlockPatch

    current = patch
    seen = set()
    chain = []
    sol_count = 0
    while current is not None:
        if id(current) in seen:
            return None
        seen.add(id(current))
        if isinstance(current, BlockPatch):
            if current.index != block_index or current.config != config:
                return None
            sol_count += 1
            if sol_count != 1:
                return None
            chain.append(("sol_h3_block_patch", config.metadata()["fingerprint"]))
            current = current.previous
            continue
        declared = _flow_mixed_grid_replacement_identity(current, block_index)
        if declared is None:
            declared = _diffaid_replacement_identity(current, options, block_index)
        if declared is None:
            return None
        identity, current = declared
        chain.append(identity)
    if sol_count != 1:
        return None
    return tuple(chain)


@dataclass(frozen=True)
class HistoryPolicy:
    config: object

    def __call__(self, *, layout, options, model):
        # Called BEFORE Spectrum decides whether to use an anchor. Never advances
        # actual counters. A policy that cannot prove its next routing returns None;
        # Spectrum executes that call and does not forecast across opaque routing.
        from .runtime import _REQUEST
        state = _REQUEST.get()
        if state is None:
            return None
        replacements = options.get("patches_replace", {}).get("dit", {})
        replacement_identity = []
        for index in range(len(model.blocks)):
            patch = replacements.get(("double_block", index))
            identity = _replacement_history_identity(patch, index, self.config, options)
            if identity is None:
                return None
            replacement_identity.append(identity)
        vdn = []
        for block in model.blocks:
            forward = block.attn.forward
            if getattr(forward, "_vdn_forward", False):
                identity = _vdn_history_identity(forward, options, layout)
                if identity is None:
                    return None
                vdn.append(identity)
        signature = getattr(layout, "signature", None)
        phase = "dense" if state.evaluations < self.config.dense_evaluations else "sol"
        return (self.config.metadata()["fingerprint"], phase, repr(signature),
                getattr(layout, "seq_len", None), tuple(getattr(layout, "segments", ())),
                str(getattr(model, "dtype", None)), tuple(replacement_identity),
                provider_identity(
                    options.get("optimized_attention_override"),
                    state.disabled_dense_providers,
                ), tuple(vdn))

    def accept_receipts(self, receipts):
        return bool(receipts) and all(
            len(item) == 3 and item[0] == "sol_h3" and item[2] in (
                "sol", "sol_external_mixed", "dense_warmup", "vdn_local_sol", "vdn_dense_warmup",
                "vdn_local_native", "vdn_global_native", "vdn_anchor_native",
                "vdn_flex_masked_native", "external_sequence_native")
            for item in receipts)


def provider_identity(provider, disabled_dense_providers=()):
    transforms = []
    seen = set()
    while hasattr(provider, "attention_preprocess_v1"):
        if id(provider) in seen:
            return ("cyclic_preprocess", id(provider))
        seen.add(id(provider))
        transform, provider = provider.attention_preprocess_v1
        transforms.append(provider_name(transform))
    identity = (tuple(transforms), provider_name(provider), id(provider))
    if id(provider) in disabled_dense_providers:
        return (*identity, "unavailable_fallback")
    return identity
