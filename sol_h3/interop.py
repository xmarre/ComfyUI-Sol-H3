"""Small duck-typed contracts shared with optional attention/forecast providers."""
from dataclasses import dataclass

HISTORY_KEY = "attention_backend_history_v1"
RECEIPTS_KEY = "attention_backend_receipts_v1"
VDN_KEY = "vdn_softmax_provider_v1"


def provider_name(provider):
    if provider is None:
        return "comfy.default"
    return f"{getattr(provider, '__module__', '<unknown>')}.{getattr(provider, '__qualname__', type(provider).__name__)}"


def receipt(options, block, route):
    sink = options.get(RECEIPTS_KEY)
    if sink is not None:
        sink.append(("sol_h3", block, route))


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
            if getattr(block.attn.forward, "_vdn_forward", False):
                describe = getattr(block.attn.forward, "attention_history_v1", None)
                identity = describe(options, layout) if callable(describe) else None
                if identity is None:
                    return None
                vdn.append(identity)
        signature = getattr(layout, "signature", None)
        phase = "dense" if state.evaluations < self.config.dense_evaluations else "sol"
        return (self.config.metadata()["fingerprint"], phase, repr(signature),
                getattr(layout, "seq_len", None), tuple(getattr(layout, "segments", ())),
                str(getattr(model, "dtype", None)),
                provider_identity(options.get("optimized_attention_override")), tuple(vdn))

    def accept_receipts(self, receipts):
        return bool(receipts) and all(
            len(item) == 3 and item[0] == "sol_h3" and item[2] in ("sol", "dense_warmup", "vdn_local_sol", "vdn_dense_warmup",
                       "vdn_local_native", "vdn_global_native", "vdn_anchor_native", "vdn_flex_masked_native",
                       "external_sequence_native")
            for item in receipts)


def provider_identity(provider):
    transforms = []
    seen = set()
    while hasattr(provider, "attention_preprocess_v1"):
        if id(provider) in seen:
            return ("cyclic_preprocess", id(provider))
        seen.add(id(provider))
        transform, provider = provider.attention_preprocess_v1
        transforms.append(provider_name(transform))
    return (tuple(transforms), provider_name(provider), id(provider))
