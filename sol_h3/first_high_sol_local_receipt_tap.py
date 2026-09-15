"""Diagnostic receipt tap for bounded first-high Sol-local E.

Backend receipts remain owned by the ordinary Sol interop sink.  This tap copies
only the completed route identity into E's clone-stable evidence owner so the E
report can prove actual 528 all-selected local, 22 dense-layer local, 50 global,
and 100 anchor routes independently of VDN's expected-route receipts.
"""
from __future__ import annotations

from . import first_high_sol_local_diagnostic as diagnostic
from . import runtime

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    current = runtime.receipt
    if not getattr(current, "_first_high_sol_local_e_v1", False):
        raise RuntimeError("Sol-local E receipt tap requires the E route owner first")

    def receipt(options, block_index, route, *args, **kwargs):
        request = diagnostic.parse_request(options) if isinstance(options, dict) else None
        effective = route
        group = diagnostic._LOCAL_GROUP.get()
        if group is not None and route == "vdn_local_sol":
            effective = "vdn_local_sol_all_selected_e"
        result = current(options, block_index, route, *args, **kwargs)
        if request is not None:
            sink = options.get(diagnostic.EVIDENCE_KEY)
            append = getattr(sink, "append", None)
            items = getattr(sink, "items", None)
            if not callable(append) or not isinstance(items, list):
                raise RuntimeError("first-high Sol-local E backend receipt evidence sink is missing")
            if sum(1 for item in items if isinstance(item, dict) and item.get("kind") == "sol_backend_receipt") >= 700:
                raise RuntimeError("first-high Sol-local E backend receipt bound exceeded")
            append(
                {
                    "kind": "sol_backend_receipt",
                    "block_index": int(block_index),
                    "route": str(effective),
                    "capture_id": request["capture_id"],
                }
            )
        return result

    receipt._first_high_sol_local_receipt_tap_v1 = True
    receipt._first_high_sol_local_inner = current
    runtime.receipt = receipt
    _INSTALLED = True


install()

__all__ = ["install"]
