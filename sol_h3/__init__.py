"""Native MiniMax-H3 optimizations; optional kernels load only on execution."""

from .history_diagnostics import install as _install_history_diagnostics
from .keyless_compat import install as _install_keyless_compat
from .keyless_domain_guard import install as _install_keyless_domain_guard

_install_history_diagnostics()
_install_keyless_compat()
_install_keyless_domain_guard()
