"""Small duck-typed contracts shared with optional attention/forecast providers."""
from dataclasses import dataclass

HISTORY_KEY = "attention_backend_history_v1"
RECEIPTS_KEY = "attention_backend_receipts_v1"
VDN_KEY = "vdn_softmax_provider_v1"
VDN_KEY_V2 = "vdn_softmax_provider_v2"
VDN_KEY_V3 = "vdn_softmax_provider_v3"
VDN_PREPROCESS_KEY = "vdn_attention_preprocess_v1"
SPECTRUM_EXTERNAL_RUNTIME_KEY = "spectrum_h3_external_patch_runtime"


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
    """Return stable Diff-Aid instance identities published to transformer options.

    Diff-Aid's Spectrum compatibility layer publishes one runtime entry per active
    MiniMax-H3 patch instance before Spectrum preflights attention history. The
    normalized sigma in that payload is intentionally excluded: Spectrum already
    owns patch-regime transitions, while Sol-H3 only needs proof that a known
    activation-only replacement is the wrapper around its block patch.
    """
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


def _diffaid_replacement_identity(patch, options):
    """Recognize the audited MiniMax-H3 Diff-Aid activation-only block wrapper.

    This deliberately requires Diff-Aid's per-call Spectrum runtime declaration.
    A same-named or structurally similar wrapper without that declaration remains
    opaque. The replacement itself only modulates ``args['img']`` and delegates to
    ``existing_patch``/``original_block``; it does not own attention routing.
    """
    if type(patch).__name__ != "MiniMaxH3BlockReplacePatch":
        return None
    runtime_identity = _diffaid_runtime_identity(options)
    if not runtime_identity or not hasattr(patch, "existing_patch"):
        return None
    config = getattr(patch, "config", None)
    fields = (
        "strength", "sigma_start", "sigma_end", "sigma_ramp",
        "token_weight_mode", "token_tail", "cond_only",
    )
    if config is None or any(not hasattr(config, name) for name in fields):
        return None
    config_identity = tuple((name, _freeze_history_value(getattr(config, name))) for name in fields)
    return (
        ("spectrum_declared_diffaid_h3_v1", runtime_identity, config_identity),
        patch.existing_patch,
    )


def _flow_layout_replacement_identity(patch, block_index):
    """Recognize Flow v0.3.x's marked generic H3 layout/context wrapper.

    ``patch_flow_model`` installs this wrapper around block 0 even when Flow's
    optional attention modes are disabled. The wrapper only publishes the already
    known packed layout/layer as temporary transformer context, records metrics,
    delegates to the previous replacement, and restores the prior context. Flow
    explicitly marks the wrapper and its previous link. Validate both those public
    markers and the real closure captures so same-named lookalikes remain opaque.
    """
    module = str(getattr(patch, "__module__", ""))
    qualname = str(getattr(patch, "__qualname__", ""))
    if not (
        (module == "h3_flow_regenerate.attention" or module.endswith(".h3_flow_regenerate.attention"))
        and qualname.endswith("make_layout_block_wrapper.<locals>.wrapper")
        and getattr(patch, "_h3_flow_layout_wrapper", False) is True
    ):
        return None

    scope = getattr(patch, "_h3_flow_layout_scope", None)
    if scope not in {"layout", "attention"}:
        return None
    marker_previous = getattr(patch, "_h3_flow_previous", None)
    marker_metrics = getattr(patch, "_h3_flow_metrics", None)
    values = _closure_values(patch)
    required = {"layer", "metrics", "previous", "record_layout"}
    if values is None or not required.issubset(values):
        return None
    if type(values["layer"]) is not int or values["layer"] != int(block_index):
        return None
    if values["previous"] is not marker_previous or values["metrics"] is not marker_metrics:
        return None
    if type(values["record_layout"]) is not bool:
        return None

    record_layout = values["record_layout"]
    if scope == "layout":
        if block_index != 0 or record_layout is not True:
            return None
    elif record_layout != (block_index == 0):
        return None

    return (
        ("h3_flow_layout_wrapper_v1", scope, block_index, record_layout),
        marker_previous,
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
        declared = _flow_layout_replacement_identity(current, block_index)
        if declared is None:
            declared = _flow_mixed_grid_replacement_identity(current, block_index)
        if declared is None:
            declared = _diffaid_replacement_identity(current, options)
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
