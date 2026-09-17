"""Spectrum history identity for Flow partitioned exact-prefix replacements.

The released Sol history parser recognizes the retired Mixed-Grid replacement by
its concrete closure contract.  The new partitioned path must have a distinct
identity rather than masquerading as that historical topology.  This opt-in
bridge extends only the replacement classifier; ordinary history behavior is
unchanged until Flow explicitly installs it.
"""
from __future__ import annotations

PARTITIONED_FLOW_IDENTITY = "h3_flow_partitioned_exact_prefix_v1"
_BRIDGE_MARKER = "_sol_h3_partitioned_history_bridge_v1"


def _partitioned_flow_replacement_identity(interop, patch, block_index):
    module = str(getattr(patch, "__module__", ""))
    qualname = str(getattr(patch, "__qualname__", ""))
    if not (
        (
            module == "h3_flow_regenerate.partitioned_mixed"
            or module.endswith(".h3_flow_regenerate.partitioned_mixed")
        )
        and qualname.endswith(
            "partitioned_diffusion_wrapper.<locals>.wrap.<locals>.call"
        )
    ):
        return None

    values = interop._closure_values(patch)
    required = {
        "layer",
        "previous",
        "plan",
        "layout",
        "mixed_layout",
        "va",
        "vb",
        "old_prefix",
        "inner",
        "partition_contract",
    }
    if values is None or not required.issubset(values):
        return None
    if type(values["layer"]) is not int or values["layer"] != int(block_index):
        return None

    plan = values["plan"]
    layout = values["layout"]
    mixed_layout = values["mixed_layout"]
    inner = values["inner"]
    partition_contract = values["partition_contract"]
    try:
        prefix_t = int(plan.prefix_t)
        temporal = int(plan.temporal)
        source_rows = int(plan.source_rows)
        target_rows = int(plan.target_rows)
        prefix_rows = int(plan.prefix_rows)
        mixed_rows = int(plan.mixed_rows)
        target_hw = tuple(int(value) for value in plan.target_hw)
        va = int(values["va"])
        vb = int(values["vb"])
        old_prefix = int(values["old_prefix"])
        native_rows = int(layout.seq_len)
        mixed_sequence_rows = int(mixed_layout.seq_len)
        native_segments = tuple(layout.segments)
        mixed_segments = tuple(mixed_layout.segments)
        inner_blocks = len(inner.blocks)
        semantic_digest = str(partition_contract["semantic_digest"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None

    if (
        not 0 < prefix_t < temporal
        or source_rows <= 0
        or target_rows <= source_rows
        or len(target_hw) != 2
        or any(value <= 0 or value % 2 for value in target_hw)
        or prefix_rows != prefix_t * target_rows
        or old_prefix != prefix_t * source_rows
        or mixed_rows != prefix_rows + (temporal - prefix_t) * source_rows
        or va <= 0
        or vb != va + temporal * source_rows
        or native_rows != vb
        or mixed_sequence_rows != va + mixed_rows
        or not native_segments
        or native_segments[-1] != (va, vb, "video")
        or not mixed_segments
        or mixed_segments[-1] != (va, mixed_sequence_rows, "video")
        or not isinstance(getattr(mixed_layout, "signature", None), tuple)
        or not mixed_layout.signature
        or mixed_layout.signature[0] != PARTITIONED_FLOW_IDENTITY
        or not semantic_digest
        or block_index < 0
        or block_index >= inner_blocks
    ):
        return None

    identity = (
        PARTITIONED_FLOW_IDENTITY,
        semantic_digest,
        native_rows,
        mixed_sequence_rows,
        va,
        temporal,
        prefix_t,
        source_rows,
        target_rows,
        target_hw,
        repr(getattr(layout, "signature", None)),
        repr(mixed_layout.signature),
    )
    return identity, values["previous"]


def install_partitioned_history_bridge() -> None:
    """Extend Sol history-v1 with the explicit Flow partition replacement."""
    from . import interop

    current = interop._flow_mixed_grid_replacement_identity
    if getattr(current, _BRIDGE_MARKER, False):
        return
    released = current

    def partitioned_aware(patch, block_index):
        identity = released(patch, block_index)
        if identity is not None:
            return identity
        return _partitioned_flow_replacement_identity(
            interop,
            patch,
            block_index,
        )

    setattr(partitioned_aware, _BRIDGE_MARKER, True)
    partitioned_aware._sol_h3_released_mixed_grid_classifier = released
    interop._flow_mixed_grid_replacement_identity = partitioned_aware


__all__ = ["PARTITIONED_FLOW_IDENTITY", "install_partitioned_history_bridge"]
