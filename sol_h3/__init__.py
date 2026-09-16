"""Native MiniMax-H3 optimizations; optional kernels load only on execution."""

from .history_diagnostics import install as _install_history_diagnostics
from . import first_high_operator_diagnostic as _first_high_operator_diagnostic  # noqa: F401
from . import first_high_sol_local_diagnostic as _first_high_sol_local_diagnostic  # noqa: F401
from . import first_high_sol_local_witness_bridge as _first_high_sol_local_witness_bridge  # noqa: F401
from . import first_high_sol_local_receipt_tap as _first_high_sol_local_receipt_tap  # noqa: F401
from . import first_high_mapped_neighbor_diagnostic as _first_high_mapped_neighbor_diagnostic  # noqa: F401

_install_history_diagnostics()
