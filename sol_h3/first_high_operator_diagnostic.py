"""Sol-H3 history/receipt contract for first-high operator comparison W.

Ordinary Sol-H3 routing is unchanged when the namespaced Flow request is absent.
For W, VDN owns the support/complement intervention while Sol owns the local
provider route and backend-history identity.  The diagnostic binds to the live
BlockPatch v3 provider closure and records a distinct native-local route only
after VDN's native SDPA succeeds; the sparse Sol-Attn body is never executed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REQUEST_KEY = "h3_first_high_operator_diagnostic_v1"
VDN_PROVIDER_V3_KEY = "vdn_softmax_provider_v3"
_ALLOWED_MODES = frozenset({"native_window", "native_full_support"})
_REQUIRED_FIELDS = (
    "api",
    "capture_id",
    "mode",
    "stage",
    "logical_call_limit",
    "sigma",
    "target_shapes_digest",
    "source_contract_digest",
)
_DIAGNOSTIC_ROUTES = {
    "vdn_local_native_window_w": "native_window",
    "vdn_local_native_full_w": "native_full_support",
}
_HEX = frozenset("0123456789abcdef")


def parse_request(options: Mapping[str, Any] | None) -> dict[str, Any] | None:
    value = (options or {}).get(REQUEST_KEY)
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != len(_REQUIRED_FIELDS):
        raise RuntimeError("first-high operator diagnostic request must be an immutable exact-field tuple")
    result: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
            raise RuntimeError("first-high operator diagnostic entries must be (name, value) tuples")
        key, field_value = item
        if key in result:
            raise RuntimeError(f"duplicate first-high operator diagnostic field: {key}")
        result[key] = field_value
    if tuple(result) != _REQUIRED_FIELDS:
        raise RuntimeError("first-high operator diagnostic fields/order do not match API 1")
    if result["api"] != 1:
        raise RuntimeError("unsupported first-high operator diagnostic API")
    if not isinstance(result["capture_id"], str) or not result["capture_id"]:
        raise RuntimeError("first-high operator diagnostic capture_id is invalid")
    if result["mode"] not in _ALLOWED_MODES:
        raise RuntimeError(f"unsupported first-high operator diagnostic mode: {result['mode']!r}")
    if result["stage"] != "high" or result["logical_call_limit"] != 1:
        raise RuntimeError("first-high operator diagnostic is restricted to one high-stage logical call")
    sigma = result["sigma"]
    if type(sigma) is not float or not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
        raise RuntimeError("first-high operator diagnostic sigma must be a finite float in (0, 1]")
    for name in ("target_shapes_digest", "source_contract_digest"):
        digest = result[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in _HEX for ch in digest):
            raise RuntimeError(f"first-high operator diagnostic {name} must be lowercase SHA-256")
    options = options or {}
    if options.get("vdn_h3_external_sequence_v1") is not None:
        raise RuntimeError("first-high operator diagnostic forbids external/reduced VDN sequence")
    if options.get("attention_measure_v1") is not None or options.get("h3_flow_mixed_grid_attention_measure_v1") is not None:
        raise RuntimeError("first-high operator diagnostic forbids weighted/Mixed-Grid attention")
    stage = options.get("h3_flow_stage")
    if stage not in {None, "high"}:
        raise RuntimeError("first-high operator diagnostic reached Sol outside the high stage")
    return result


def history_identity(options: Mapping[str, Any] | None):
    request = parse_request(options)
    if request is None:
        return None
    return (
        "h3_first_high_operator_diagnostic_v1",
        request["mode"],
        request["capture_id"],
        request["logical_call_limit"],
        request["sigma"],
        request["target_shapes_digest"],
        request["source_contract_digest"],
    )


def _closure_values(function: Any) -> dict[str, Any]:
    target = getattr(function, "__func__", function)
    code = getattr(target, "__code__", None)
    cells = getattr(target, "__closure__", None)
    if code is None or cells is None or len(code.co_freevars) != len(cells):
        return {}
    result: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, cells, strict=True):
        try:
            result[str(name)] = cell.cell_contents
        except ValueError:
            continue
    return result


def _provider_record(provider: Any):
    """Resolve the reviewed BlockPatch-local receipt owner fail-closed."""
    seen: set[int] = set()
    pending = [provider]
    while pending:
        current = pending.pop()
        if not callable(current) or id(current) in seen:
            continue
        seen.add(id(current))
        values = _closure_values(current)
        record = values.get("record")
        if callable(record):
            return record
        for value in values.values():
            if callable(value):
                pending.append(value)
    raise RuntimeError("first-high operator diagnostic could not resolve the Sol BlockPatch receipt owner")


def _provider_block_index(provider: Any) -> int:
    record = _provider_record(provider)
    owner = _closure_values(record).get("self")
    index = getattr(owner, "index", None)
    if type(index) is not int or index < 0:
        raise RuntimeError("first-high operator diagnostic could not resolve the Sol block owner")
    return index


def record_native_local(
    options: Mapping[str, Any],
    block_index: int,
    route: str,
    request: Mapping[str, Any],
) -> None:
    """Record a completed VDN-native local call through the live Sol owner.

    VDN calls this only after native SDPA returned successfully.  Resolving the
    active v3 provider closure ties the diagnostic receipt to the actual
    BlockPatch request/counter lifetime instead of fabricating a parallel sink.
    """
    active = parse_request(options)
    if active is None or dict(active) != dict(request):
        raise RuntimeError("first-high operator diagnostic Sol/VDN request identity diverged")
    expected_mode = _DIAGNOSTIC_ROUTES.get(route)
    if expected_mode is None or expected_mode != active["mode"]:
        raise RuntimeError("first-high operator diagnostic local route does not match the selected mode")
    provider = options.get(VDN_PROVIDER_V3_KEY)
    if not callable(provider):
        raise RuntimeError("first-high operator diagnostic requires the live Sol VDN v3 provider")
    if _provider_block_index(provider) != int(block_index):
        raise RuntimeError("first-high operator diagnostic Sol/VDN block ownership diverged")
    _provider_record(provider)(route)


def _diagnostic_receipt(item: Any) -> tuple | None:
    if not isinstance(item, tuple) or len(item) != 3 or item[0] != "sol_h3":
        return None
    if item[2] not in _DIAGNOSTIC_ROUTES:
        return None
    return (item[0], item[1], "vdn_local_native")


def _install_history_policy_patch() -> None:
    from . import interop

    cls = interop.HistoryPolicy
    if getattr(cls, "_first_high_operator_diagnostic_v1", False):
        return
    original_call = cls.__call__
    original_accept = cls.accept_receipts

    def call(self, *, layout, options, model):
        identity = original_call(self, layout=layout, options=options, model=model)
        diagnostic = history_identity(options)
        if diagnostic is None:
            return identity
        if identity is None:
            return None
        return (*identity, diagnostic)

    def accept_receipts(self, receipts):
        if not receipts:
            return False
        normalized = []
        for item in receipts:
            diagnostic = _diagnostic_receipt(item)
            normalized.append(diagnostic if diagnostic is not None else item)
        return original_accept(self, normalized)

    cls.__call__ = call
    cls.accept_receipts = accept_receipts
    cls._first_high_operator_diagnostic_v1 = True


_install_history_policy_patch()

__all__ = ["REQUEST_KEY", "history_identity", "parse_request", "record_native_local"]
