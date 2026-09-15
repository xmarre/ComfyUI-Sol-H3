"""Sol-H3 history/receipt contract for first-high operator comparison W.

Ordinary Sol-H3 routing is unchanged when the namespaced Flow request is absent.
For W, VDN owns the native-SDPA intervention while Sol owns the backend-history
identity and acceptance of the two explicit local-native receipt kinds.  Sparse
Sol-Attn is therefore never executed for W local calls.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REQUEST_KEY = "h3_first_high_operator_diagnostic_v1"
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


def _diagnostic_receipt(item: Any) -> tuple | None:
    if not isinstance(item, tuple) or len(item) != 4 or item[0] != "sol_h3":
        return None
    route = item[2]
    expected_mode = _DIAGNOSTIC_ROUTES.get(route)
    if expected_mode is None:
        return None
    fields = item[3]
    if not isinstance(fields, tuple) or len(fields) != 4:
        raise RuntimeError("first-high operator diagnostic Sol receipt fields are malformed")
    values = dict(fields)
    if tuple(key for key, _value in fields) != ("capture_id", "mode", "source_contract_digest", "completed"):
        raise RuntimeError("first-high operator diagnostic Sol receipt schema is malformed")
    if not isinstance(values.get("capture_id"), str) or not values["capture_id"]:
        raise RuntimeError("first-high operator diagnostic Sol receipt capture_id is invalid")
    if values.get("mode") != expected_mode:
        raise RuntimeError("first-high operator diagnostic Sol receipt mode/route disagree")
    digest = values.get("source_contract_digest")
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in _HEX for ch in digest):
        raise RuntimeError("first-high operator diagnostic Sol receipt source digest is invalid")
    if values.get("completed") is not True:
        raise RuntimeError("first-high operator diagnostic Sol receipt is incomplete")
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

__all__ = ["REQUEST_KEY", "history_identity", "parse_request"]
