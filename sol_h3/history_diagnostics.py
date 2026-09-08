"""Non-invasive diagnostics for Spectrum/SOL numerical-backend history.

This module does not change routing. It wraps the existing HistoryPolicy methods
only to explain why a preflight is opaque, which semantic identity component
changes between calls, or which actual receipt set is rejected.
"""
from __future__ import annotations

import logging
from typing import Any

from . import interop

log = logging.getLogger("comfy.sol_h3")
_INSTALLED = False
_ORIGINAL_CALL = None
_ORIGINAL_ACCEPT = None
_COMPONENTS = (
    "config",
    "phase",
    "layout_signature",
    "seq_len",
    "segments",
    "dtype",
    "replacements",
    "dense_provider",
    "vdn",
)


def _request_state():
    try:
        from .runtime import _REQUEST
    except (ImportError, AttributeError):
        return None
    return _REQUEST.get()


def _seen(state) -> set[Any]:
    value = getattr(state, "_backend_history_diag_seen", None)
    if value is None:
        value = set()
        setattr(state, "_backend_history_diag_seen", value)
    return value


def _next_call(state) -> int:
    value = int(getattr(state, "_backend_history_diag_calls", 0)) + 1
    setattr(state, "_backend_history_diag_calls", value)
    return value


def _label(value: Any) -> str:
    if value is None:
        return "none"
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module or qualname:
        return f"{module or '<unknown>'}.{qualname or type(value).__name__}"
    cls = type(value)
    return f"{getattr(cls, '__module__', '<unknown>')}.{getattr(cls, '__qualname__', cls.__name__)}"


def _opaque_reason(policy, *, layout, options, model) -> str:
    replacements = options.get("patches_replace", {}).get("dit", {})
    for index in range(len(model.blocks)):
        patch = replacements.get(("double_block", index))
        if interop._replacement_history_identity(patch, index, policy.config, options) is None:
            return f"replacement_chain:block={index}:outer={_label(patch)}"

    for index, block in enumerate(model.blocks):
        forward = block.attn.forward
        if not getattr(forward, "_vdn_forward", False):
            continue
        if interop._vdn_history_identity(forward, options, layout) is not None:
            continue
        values = interop._closure_values(forward)
        closure_keys = "-"
        backend = None
        if isinstance(values, dict):
            closure_keys = ",".join(sorted(str(key) for key in values)) or "-"
            state = values.get("state")
            backend = getattr(state, "softmax_backend", None)
        return (
            f"vdn_history:block={index}:forward={_label(forward)}:"
            f"backend={backend}:closure={closure_keys}"
        )
    return "provider_or_unclassified"


def _record_identity(state, call_index: int, identity) -> None:
    previous = getattr(state, "_backend_history_diag_identity", None)
    if previous is None:
        setattr(state, "_backend_history_diag_identity", identity)
        key = ("established", identity[1], identity[3], identity[7], len(identity[8]))
        if key not in _seen(state):
            _seen(state).add(key)
            log.info(
                "Sol-H3 backend-history diagnostic established call=%d phase=%s seq_len=%s "
                "dense_provider=%s vdn_count=%d",
                call_index,
                identity[1],
                identity[3],
                identity[7],
                len(identity[8]),
            )
        return

    if previous == identity:
        return

    changed = tuple(
        name
        for name, old, new in zip(_COMPONENTS, previous, identity, strict=True)
        if old != new
    )
    setattr(state, "_backend_history_diag_identity", identity)
    log.info(
        "Sol-H3 backend-history diagnostic transition call=%d components=%s phase=%s->%s seq_len=%s->%s",
        call_index,
        ",".join(changed) or "unknown",
        previous[1],
        identity[1],
        previous[3],
        identity[3],
    )


def _diagnostic_call(self, *, layout, options, model):
    assert _ORIGINAL_CALL is not None
    result = _ORIGINAL_CALL(self, layout=layout, options=options, model=model)
    state = _request_state()
    if state is None:
        return result

    call_index = _next_call(state)
    if result is None:
        reason = _opaque_reason(self, layout=layout, options=options, model=model)
        key = ("opaque", reason)
        if key not in _seen(state):
            _seen(state).add(key)
            log.warning(
                "Sol-H3 backend-history diagnostic opaque call=%d reason=%s",
                call_index,
                reason,
            )
        return result

    _record_identity(state, call_index, result)
    return result


def _receipt_rejection_reason(receipts) -> str:
    if not receipts:
        return "empty"
    allowed = {
        "sol",
        "sol_external_mixed",
        "dense_warmup",
        "vdn_local_sol",
        "vdn_dense_warmup",
        "vdn_local_native",
        "vdn_global_native",
        "vdn_anchor_native",
        "vdn_flex_masked_native",
        "external_sequence_native",
    }
    for item in receipts:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            return f"malformed:{item!r}"
        if item[0] != "sol_h3":
            return f"foreign_provider:{item[0]!r}"
        if item[2] not in allowed:
            return f"unsupported_route:{item[2]!r}"
    return "unclassified"


def _diagnostic_accept(self, receipts):
    assert _ORIGINAL_ACCEPT is not None
    accepted = _ORIGINAL_ACCEPT(self, receipts)
    state = _request_state()
    if state is None:
        return accepted

    if accepted:
        routes = tuple(sorted({item[2] for item in receipts if isinstance(item, (tuple, list)) and len(item) == 3}))
        key = ("accepted_receipts", routes)
        if key not in _seen(state):
            _seen(state).add(key)
            log.info(
                "Sol-H3 backend-history diagnostic receipts accepted routes=%s count=%d",
                ",".join(routes) or "-",
                len(receipts),
            )
        return accepted

    reason = _receipt_rejection_reason(receipts)
    key = ("rejected_receipts", reason)
    if key not in _seen(state):
        _seen(state).add(key)
        log.warning(
            "Sol-H3 backend-history diagnostic receipts rejected reason=%s count=%d",
            reason,
            len(receipts) if hasattr(receipts, "__len__") else -1,
        )
    return accepted


def install() -> None:
    global _INSTALLED, _ORIGINAL_CALL, _ORIGINAL_ACCEPT
    if _INSTALLED:
        return
    _ORIGINAL_CALL = interop.HistoryPolicy.__call__
    _ORIGINAL_ACCEPT = interop.HistoryPolicy.accept_receipts
    interop.HistoryPolicy.__call__ = _diagnostic_call
    interop.HistoryPolicy.accept_receipts = _diagnostic_accept
    _INSTALLED = True


__all__ = ["install"]
