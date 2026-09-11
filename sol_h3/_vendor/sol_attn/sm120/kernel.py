"""SM120 kernel recipe."""

from .mainloop import SolAttnForwardSm120


def make_kernel(
    *,
    debug_route_trace: bool = False,
    prefetch_first_exact_k: bool = True,
    prefetch_next_route_k: bool = True,
    key_bias_enabled: bool = False,
):
    return SolAttnForwardSm120(
        debug_route_trace=debug_route_trace,
        prefetch_first_exact_k=prefetch_first_exact_k,
        prefetch_next_route_k=prefetch_next_route_k,
        key_bias_enabled=key_bias_enabled,
    )


__all__ = ["make_kernel"]
