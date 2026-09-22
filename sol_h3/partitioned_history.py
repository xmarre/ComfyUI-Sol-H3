"""Spectrum history identity and receipts for partitioned exact-prefix Flow.

The released Sol history parser recognizes the retired Mixed-Grid replacement by
its concrete closure contract. The new partitioned path has a distinct identity,
layout contract and explicit completion receipts. This opt-in bridge extends only
those classifiers; ordinary history behavior delegates to the released
implementation.
"""
from __future__ import annotations

import math

PARTITIONED_FLOW_IDENTITY = "h3_flow_partitioned_exact_prefix_v1"
VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_PARTITIONED_SEQUENCE_API = 4
VDN_PARTITIONED_SEQUENCE_MODE = "partitioned_attention_variable_grid_linear"
_BRIDGE_MARKER = "_sol_h3_partitioned_history_bridge_v1"
_RECEIPT_BRIDGE_MARKER = "_sol_h3_partitioned_receipt_bridge_v1"
_VDN_HISTORY_BRIDGE_MARKER = "_sol_h3_partitioned_vdn_history_bridge_v1"


def _digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _partitioned_history_layout_valid(options, layout) -> bool:
    """Validate Flow's current partitioned layout before any forecast identity."""
    contract = (options or {}).get(PARTITIONED_FLOW_IDENTITY)
    if contract is None:
        return True
    if not isinstance(contract, dict):
        return False
    names = (
        "sequence_rows",
        "video_start",
        "temporal",
        "prefix_t",
        "source_rows_per_frame",
        "target_rows_per_frame",
    )
    if any(type(contract.get(name)) is not int for name in names):
        return False
    sequence_rows = contract["sequence_rows"]
    video_start = contract["video_start"]
    temporal = contract["temporal"]
    prefix_t = contract["prefix_t"]
    source_rows = contract["source_rows_per_frame"]
    target_rows = contract["target_rows_per_frame"]
    if (
        sequence_rows <= 0
        or video_start <= 0
        or temporal <= 1
        or not 0 < prefix_t < temporal
        or source_rows <= 0
        or target_rows <= source_rows
        or sequence_rows
        != video_start + prefix_t * target_rows + (temporal - prefix_t) * source_rows
        or not _digest(contract.get("semantic_digest"))
    ):
        return False

    signature = getattr(layout, "signature", None)
    segments = getattr(layout, "segments", None)
    if (
        int(getattr(layout, "seq_len", -1)) != sequence_rows
        or not isinstance(signature, tuple)
        or not signature
        or signature[0] != PARTITIONED_FLOW_IDENTITY
        or not isinstance(segments, (tuple, list))
        or not segments
        or tuple(segments[-1]) != (video_start, sequence_rows, "video")
    ):
        return False

    external = (options or {}).get(VDN_EXTERNAL_SEQUENCE_KEY)
    if not isinstance(external, dict):
        return False
    return bool(
        external.get("api") == VDN_PARTITIONED_SEQUENCE_API
        and external.get("mode") == VDN_PARTITIONED_SEQUENCE_MODE
        and external.get("topology") == "target_prefix_source_suffix"
        and external.get("sequence_rows") == sequence_rows
        and external.get("video_start") == video_start
        and external.get("temporal") == temporal
        and external.get("prefix_t") == prefix_t
        and external.get("source_rows_per_frame") == source_rows
        and external.get("target_rows_per_frame") == target_rows
        and external.get("flow_semantic_digest") == contract["semantic_digest"]
    )


def _partitioned_flow_replacement_identity(interop, patch, block_index):
    """Recognize the dedicated Flow partitioned transformer replacement exactly."""
    module = str(getattr(patch, "__module__", ""))
    qualname = str(getattr(patch, "__qualname__", ""))
    if not (
        (
            module == "h3_flow_regenerate.partitioned_transformer"
            or module.endswith(".h3_flow_regenerate.partitioned_transformer")
        )
        and qualname.endswith("partitioned_diffusion_wrapper.<locals>.wrap.<locals>.call")
    ):
        return None

    values = interop._closure_values(patch)
    required = {
        "layer",
        "previous",
        "plan",
        "layout",
        "partitioned_layout",
        "video_start",
        "video_end",
        "carrier_prefix_rows",
        "inner",
        "partition_contract",
    }
    if values is None or not required.issubset(values):
        return None
    if type(values["layer"]) is not int or values["layer"] != int(block_index):
        return None

    plan = values["plan"]
    layout = values["layout"]
    partitioned_layout = values["partitioned_layout"]
    inner = values["inner"]
    partition_contract = values["partition_contract"]
    try:
        prefix_t = int(plan.prefix_t)
        temporal = int(plan.temporal)
        source_rows = int(plan.source_rows)
        target_rows = int(plan.target_rows)
        prefix_rows = int(plan.prefix_rows)
        partitioned_rows = int(plan.partitioned_rows)
        target_hw = tuple(int(value) for value in plan.target_hw)
        video_start = int(values["video_start"])
        video_end = int(values["video_end"])
        carrier_prefix_rows = int(values["carrier_prefix_rows"])
        native_rows = int(layout.seq_len)
        partitioned_sequence_rows = int(partitioned_layout.seq_len)
        native_segments = tuple(layout.segments)
        partitioned_segments = tuple(partitioned_layout.segments)
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
        or carrier_prefix_rows != prefix_t * source_rows
        or partitioned_rows != prefix_rows + (temporal - prefix_t) * source_rows
        or video_start <= 0
        or video_end != video_start + temporal * source_rows
        or native_rows != video_end
        or partitioned_sequence_rows != video_start + partitioned_rows
        or not native_segments
        or native_segments[-1] != (video_start, video_end, "video")
        or not partitioned_segments
        or partitioned_segments[-1] != (video_start, partitioned_sequence_rows, "video")
        or not isinstance(getattr(partitioned_layout, "signature", None), tuple)
        or not partitioned_layout.signature
        or partitioned_layout.signature[0] != PARTITIONED_FLOW_IDENTITY
        or not _digest(semantic_digest)
        or block_index < 0
        or block_index >= inner_blocks
    ):
        return None

    identity = (
        PARTITIONED_FLOW_IDENTITY,
        semantic_digest,
        native_rows,
        partitioned_sequence_rows,
        video_start,
        temporal,
        prefix_t,
        source_rows,
        target_rows,
        target_hw,
        repr(getattr(layout, "signature", None)),
        repr(partitioned_layout.signature),
    )
    return identity, values["previous"]


def _accept_partitioned_receipt(item) -> bool:
    from .mapped_neighbors import POLICY as MAPPED_POLICY
    from .partitioned_request import (
        PARTITIONED_DENSE_ROUTE,
        PARTITIONED_MAPPED_ROUTE,
        PARTITIONED_RECEIPT_TAG,
        PARTITIONED_REQUEST_ABI,
        PARTITIONED_SOL_ROUTE,
    )
    from .runtime import _REQUEST

    if not isinstance(item, tuple) or len(item) != 4 or item[0] != "sol_h3":
        return False
    block = item[1]
    route = item[2]
    fields = item[3]
    if type(block) is not int or block < 0:
        return False
    if route not in {PARTITIONED_DENSE_ROUTE, PARTITIONED_SOL_ROUTE, PARTITIONED_MAPPED_ROUTE}:
        return False
    if not isinstance(fields, tuple) or len(fields) != 16:
        return False
    (
        tag,
        call_token,
        abi,
        semantic_digest,
        kind,
        execution_mode,
        q_rows,
        kv_rows,
        sink_rows,
        prefix_k_range,
        prefix_log_key_measure,
        map_digest,
        descriptor_digest,
        mapped_policy,
        kernel_contract,
        completed,
    ) = fields
    if (
        tag != PARTITIONED_RECEIPT_TAG
        or not isinstance(call_token, tuple)
        or len(call_token) != 2
        or call_token[0] != "sol_h3_evaluation"
        or type(call_token[1]) is not int
        or call_token[1] < 0
        or abi != PARTITIONED_REQUEST_ABI
        or not _digest(semantic_digest)
        or kind not in {"global", "local", "anchor", "full"}
        or type(q_rows) is not int
        or q_rows <= 0
        or type(kv_rows) is not int
        or kv_rows <= 0
        or type(sink_rows) is not int
        or not 0 <= sink_rows <= kv_rows
        or type(prefix_log_key_measure) is not float
        or not math.isfinite(prefix_log_key_measure)
        or prefix_log_key_measure > 0.0
        or completed is not True
    ):
        return False

    if prefix_k_range is None:
        if prefix_log_key_measure != 0.0:
            return False
    else:
        if (
            not isinstance(prefix_k_range, tuple)
            or len(prefix_k_range) != 2
            or any(type(value) is not int for value in prefix_k_range)
        ):
            return False
        start, end = prefix_k_range
        if not sink_rows <= start < end <= kv_rows:
            return False
        if route != PARTITIONED_DENSE_ROUTE and start != sink_rows:
            return False

    if map_digest is None:
        if descriptor_digest is not None or mapped_policy is not None:
            return False
    else:
        if not _digest(map_digest) or mapped_policy != MAPPED_POLICY:
            return False
        if descriptor_digest is not None and not _digest(descriptor_digest):
            return False

    if route == PARTITIONED_DENSE_ROUTE:
        if execution_mode not in {"dense_forced", "dense_warmup"} or kernel_contract is not None:
            return False
    elif route == PARTITIONED_MAPPED_ROUTE:
        if (
            execution_mode != "sm120_mapped"
            or not _digest(map_digest)
            or not _digest(descriptor_digest)
            or not isinstance(kernel_contract, str)
            or not kernel_contract
        ):
            return False
    else:
        if (
            execution_mode != "sm120_union"
            or descriptor_digest is not None
            or not isinstance(kernel_contract, str)
            or not kernel_contract
        ):
            return False

    state = _REQUEST.get()
    owned = getattr(state, "partitioned_validated_receipts", set()) if state is not None else set()
    return (block, fields) in owned


def install_partitioned_history_bridge() -> None:
    """Extend Sol history-v1 with explicit Flow partition identity and receipts."""
    from . import interop
    from .partitioned_request import (
        PARTITIONED_DENSE_ROUTE,
        PARTITIONED_MAPPED_ROUTE,
        PARTITIONED_SOL_ROUTE,
    )

    current = interop._flow_mixed_grid_replacement_identity
    if not getattr(current, _BRIDGE_MARKER, False):
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

    current_vdn_history = interop._mapped_vdn_history_identity
    if not getattr(current_vdn_history, _VDN_HISTORY_BRIDGE_MARKER, False):
        released_vdn_history = current_vdn_history

        def partitioned_vdn_history(forward, options, layout):
            if (options or {}).get(PARTITIONED_FLOW_IDENTITY) is not None:
                if not _partitioned_history_layout_valid(options, layout):
                    return True, None
            return released_vdn_history(forward, options, layout)

        setattr(partitioned_vdn_history, _VDN_HISTORY_BRIDGE_MARKER, True)
        partitioned_vdn_history._sol_h3_released_mapped_vdn_history = released_vdn_history
        interop._mapped_vdn_history_identity = partitioned_vdn_history

    current_accept = interop.HistoryPolicy.accept_receipts
    if not getattr(current_accept, _RECEIPT_BRIDGE_MARKER, False):
        released_accept = current_accept
        partitioned_routes = {
            PARTITIONED_DENSE_ROUTE,
            PARTITIONED_SOL_ROUTE,
            PARTITIONED_MAPPED_ROUTE,
        }

        def partitioned_accept(self, receipts):
            if not receipts:
                return False
            ordinary = []
            saw_partitioned = False
            for item in receipts:
                route = item[2] if isinstance(item, tuple) and len(item) >= 3 else None
                if route in partitioned_routes:
                    saw_partitioned = True
                    if not _accept_partitioned_receipt(item):
                        return False
                else:
                    ordinary.append(item)
            if ordinary and not released_accept(self, ordinary):
                return False
            return bool(saw_partitioned or ordinary)

        setattr(partitioned_accept, _RECEIPT_BRIDGE_MARKER, True)
        partitioned_accept._sol_h3_released_accept_receipts = released_accept
        interop.HistoryPolicy.accept_receipts = partitioned_accept


__all__ = [
    "PARTITIONED_FLOW_IDENTITY",
    "install_partitioned_history_bridge",
]
