"""Native MiniMax-H3 optimizations; optional kernels load only on execution."""

from .history_diagnostics import install as _install_history_diagnostics
from . import first_high_operator_diagnostic as _first_high_operator_diagnostic  # noqa: F401
from . import first_high_sol_local_diagnostic as _first_high_sol_local_diagnostic  # noqa: F401
from . import first_high_sol_local_witness_bridge as _first_high_sol_local_witness_bridge  # noqa: F401
from . import first_high_sol_local_receipt_tap as _first_high_sol_local_receipt_tap  # noqa: F401
from . import first_high_mapped_neighbor_diagnostic as _first_high_mapped_neighbor_diagnostic  # noqa: F401
from . import interop as _interop

# M is stacked over E, but installing the M overlay must not make the preserved E
# request invalid.  Restore E's mode constant and multiplex only the separately
# versioned M request through the exact E parser after rewriting its mode field in
# a private validation copy.  All other E validation remains unchanged.
_E_MODE = "all_selected_e"
_M_MODE = _first_high_mapped_neighbor_diagnostic.MODE
_ORIGINAL_E_PARSE_REQUEST = _first_high_sol_local_diagnostic.parse_request
_first_high_sol_local_diagnostic.MODE = _E_MODE


def _parse_e_or_m_request(options):
    raw = (options or {}).get(_first_high_sol_local_diagnostic.REQUEST_KEY)
    if not (
        isinstance(raw, tuple)
        and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in raw)
    ):
        return _ORIGINAL_E_PARSE_REQUEST(options)
    modes = [item[1] for item in raw if item[0] == "mode"]
    if modes != [_M_MODE]:
        return _ORIGINAL_E_PARSE_REQUEST(options)
    rewritten = tuple((name, _E_MODE if name == "mode" else value) for name, value in raw)
    proxy = dict(options or {})
    proxy[_first_high_sol_local_diagnostic.REQUEST_KEY] = rewritten
    parsed = _ORIGINAL_E_PARSE_REQUEST(proxy)
    if parsed is None:
        raise RuntimeError("mapped-neighbor M request unexpectedly parsed as inactive")
    result = dict(parsed)
    result["mode"] = _M_MODE
    return result


_first_high_sol_local_diagnostic.parse_request = _parse_e_or_m_request

_install_history_diagnostics()

# M changes the diagnostic route label only. Backend-history acceptance must see
# the same numerical ownership as the ordinary VDN local Sol route so the M label
# cannot create a synthetic history transition or force a forecast to actual.
_ORIGINAL_M_HISTORY_ACCEPT = _interop.HistoryPolicy.accept_receipts


def _accept_e_or_m_history_receipts(self, receipts):
    if not receipts:
        return _ORIGINAL_M_HISTORY_ACCEPT(self, receipts)
    normalized = []
    for item in receipts:
        if isinstance(item, tuple) and len(item) >= 3 and item[0] == "sol_h3" and item[2] == _first_high_mapped_neighbor_diagnostic.ROUTE:
            normalized.append((item[0], item[1], "vdn_local_sol", *item[3:]))
        else:
            normalized.append(item)
    return _ORIGINAL_M_HISTORY_ACCEPT(self, normalized)


_accept_e_or_m_history_receipts._first_high_mapped_neighbor_m_v1 = True
_accept_e_or_m_history_receipts._first_high_mapped_neighbor_inner = _ORIGINAL_M_HISTORY_ACCEPT
_interop.HistoryPolicy.accept_receipts = _accept_e_or_m_history_receipts
