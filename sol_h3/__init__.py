"""Native MiniMax-H3 optimizations; optional kernels load only on execution."""

from .history_diagnostics import install as _install_history_diagnostics
from . import first_high_operator_diagnostic as _first_high_operator_diagnostic  # noqa: F401

_install_history_diagnostics()
