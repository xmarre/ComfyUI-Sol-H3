"""Small duck-typed contracts shared with optional attention/forecast providers."""
from dataclasses import dataclass

HISTORY_KEY = "attention_backend_history_v1"
RECEIPTS_KEY = "attention_backend_receipts_v1"
VDN_KEY = "vdn_softmax_provider_v1"
VDN_KEY_V2 = "vdn_softmax_provider_v2"
VDN_PREPROCESS_KEY = "vdn_attention_preprocess_v1"


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
        from .runtime import BlockPatch
        for patch in replacements.values():
            if not isinstance(patch, BlockPatch) or patch.previous is not None:
                return None
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
                str(getattr(model, "dtype", None)),
                provider_identity(
                    options.get("optimized_attention_override"),
                    state.disabled_dense_providers,
                ), tuple(vdn))

    def accept_receipts(self, receipts):
        return bool(receipts) and all(
            len(item) == 3 and item[0] == "sol_h3" and item[2] in (
                "sol", "dense_warmup", "vdn_local_sol", "vdn_dense_warmup",
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
