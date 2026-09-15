"""Diagnostic-only first-high operator contract for the Flow W experiment.

This module does not change ordinary Sol-H3 routing.  When Flow publishes the
strict ``h3_first_high_operator_diagnostic_v1`` request, VDN may ask this module
to consume the already-installed Sol v3 local-provider closure as a native-SDPA
receipt owner.  The helper records a Sol receipt without executing the sparse
kernel.  HistoryPolicy is extended at import time so Spectrum sees the selected
operator mode before preflight and accepts the two explicit diagnostic receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

FIRST_HIGH_OPERATOR_KEY = "h3_first_high_operator_diagnostic_v1"
VDN_PROVIDER_V3_KEY = "vdn_softmax_provider_v3"
_NATIVE_ROUTES = {
    "native_window": "vdn_local_native_window",
    "native_full_support": "vdn_local_native_full_support",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = frozenset(
    {
        "api",
        "capture_id",
        "mode",
        "stage",
        "logical_call_limit",
        "sigma",
        "target_shapes_digest",
        "source_contract_digest",
    }
)


@dataclass(frozen=True, slots=True)
class FirstHighOperatorRequest:
    api: int
    capture_id: str
    mode: str
    stage: str
    logical_call_limit: int
    sigma: float
    target_shapes_digest: str
    source_contract_digest: str


def parse_request(options: dict[str, Any] | None) -> FirstHighOperatorRequest | None:
    options = options or {}
    raw = options.get(FIRST_HIGH_OPERATOR_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != _REQUIRED_FIELDS:
        raise RuntimeError("first-high operator diagnostic request has an invalid schema")
    if type(raw.get("api")) is not int or raw["api"] != 1:
        raise RuntimeError("first-high operator diagnostic API is unsupported")
    capture_id = raw.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id or len(capture_id) > 128:
        raise RuntimeError("first-high operator diagnostic capture_id is invalid")
    mode = raw.get("mode")
    if mode not in _NATIVE_ROUTES:
        raise RuntimeError("first-high operator diagnostic mode is unsupported")
    if raw.get("stage") != "high" or options.get("h3_flow_stage") not in {None, "high"}:
        raise RuntimeError("first-high operator diagnostic is restricted to the high stage")
    if type(raw.get("logical_call_limit")) is not int or raw["logical_call_limit"] != 1:
        raise RuntimeError("first-high operator diagnostic requires one logical call")
    sigma = raw.get("sigma")
    if type(sigma) not in {int, float} or not math.isfinite(float(sigma)) or not 0.0 < float(sigma) <= 1.0:
        raise RuntimeError("first-high operator diagnostic sigma is invalid")
    target_digest = raw.get("target_shapes_digest")
    source_digest = raw.get("source_contract_digest")
    if not isinstance(target_digest, str) or _HEX64.fullmatch(target_digest) is None:
        raise RuntimeError("first-high operator diagnostic target-shapes digest is invalid")
    if not isinstance(source_digest, str) or _HEX64.fullmatch(source_digest) is None:
        raise RuntimeError("first-high operator diagnostic source-contract digest is invalid")
    for forbidden in ("attention_measure_v1", "h3_flow_mixed_grid_attention_measure_v1"):
        if options.get(forbidden) is not None:
            raise RuntimeError("first-high operator diagnostic cannot run with weighted/Mixed-Grid attention")
    if options.get("vdn_h3_external_sequence_v1") is not None:
        raise RuntimeError("first-high operator diagnostic cannot run on an external mixed/reduced sequence")
    return FirstHighOperatorRequest(
        api=1,
        capture_id=capture_id,
        mode=str(mode),
        stage="high",
        logical_call_limit=1,
        sigma=float(sigma),
        target_shapes_digest=target_digest,
        source_contract_digest=source_digest,
    )


def history_identity(options: dict[str, Any] | None):
    request = parse_request(options)
    if request is None:
        return None
    return (
        "h3_first_high_operator_diagnostic_v1",
        request.mode,
        request.capture_id,
        request.logical_call_limit,
        request.sigma,
        request.target_shapes_digest,
        request.source_contract_digest,
    )


def _closure_values(function: Any) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None)
    if code is None or cells is None or len(code.co_freevars) != len(cells):
        return {}
    values: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, cells, strict=True):
        try:
            values[str(name)] = cell.cell_contents
        except ValueError:
            continue
    return values


def _provider_record(provider: Any):
    """Resolve the current BlockPatch-local ``record`` closure fail-closed.

    The diagnostic deliberately depends on the reviewed Sol v3 provider shape.
    Unknown providers are not treated as equivalent native routes.
    """
    seen: set[int] = set()
    pending = [provider]
    while pending:
        current = pending.pop()
        ident = id(current)
        if ident in seen or not callable(current):
            continue
        seen.add(ident)
        values = _closure_values(current)
        record = values.get("record")
        if callable(record):
            return record
        for value in values.values():
            if callable(value):
                pending.append(value)
    raise RuntimeError("first-high operator diagnostic could not resolve the Sol v3 receipt owner")


def provider_block_index(provider: Any) -> int:
    record = _provider_record(provider)
    values = _closure_values(record)
    owner = values.get("self")
    index = getattr(owner, "index", None)
    if type(index) is not int or index < 0:
        raise RuntimeError("first-high operator diagnostic could not resolve the Sol block owner")
    return index


def run_native_local(
    options: dict[str, Any] | None,
    native,
    q,
    k,
    v,
    *,
    kind: str,
    scale: float,
    square_aligned: bool = False,
    sink_rows: int = 0,
):
    """Execute VDN's native local closure and emit the explicit Sol receipt.

    VDN already owns Q/K/V selection.  This function owns only the diagnostic
    Sol-routing decision and its receipt; it does not preprocess, gather, gate,
    project, or evaluate Sol-Attn.
    """
    _ = q, k, v, scale, square_aligned, sink_rows
    request = parse_request(options)
    if request is None:
        raise RuntimeError("native first-high local dispatch requires an active diagnostic request")
    if kind != "local":
        raise RuntimeError("native first-high dispatch may only replace VDN local attention")
    provider = (options or {}).get(VDN_PROVIDER_V3_KEY)
    if not callable(provider):
        raise RuntimeError("first-high operator diagnostic requires the active Sol VDN v3 provider")
    record = _provider_record(provider)
    route = _NATIVE_ROUTES[request.mode]
    result = native()
    record(route)
    return result, provider_block_index(provider), route


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
            if (
                isinstance(item, tuple)
                and len(item) == 3
                and item[0] == "sol_h3"
                and item[2] in set(_NATIVE_ROUTES.values())
            ):
                normalized.append((item[0], item[1], "vdn_local_native"))
            else:
                normalized.append(item)
        return original_accept(self, normalized)

    cls.__call__ = call
    cls.accept_receipts = accept_receipts
    cls._first_high_operator_diagnostic_v1 = True


_install_history_policy_patch()

__all__ = [
    "FIRST_HIGH_OPERATOR_KEY",
    "FirstHighOperatorRequest",
    "history_identity",
    "parse_request",
    "provider_block_index",
    "run_native_local",
]
