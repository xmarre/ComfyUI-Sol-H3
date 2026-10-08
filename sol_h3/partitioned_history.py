"""Spectrum history identity and receipts for partitioned exact-prefix Flow.

The released Sol history parser recognizes the retired Mixed-Grid replacement by
its concrete closure contract. The new partitioned path has a distinct identity,
layout contract and explicit completion receipts. This opt-in bridge extends only
those classifiers; ordinary history behavior delegates to the released
implementation.
"""
from __future__ import annotations

import hashlib
import json
import math

PARTITIONED_FLOW_IDENTITY = "h3_flow_partitioned_exact_prefix_v1"
VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_PARTITIONED_SEQUENCE_API = 4
VDN_PARTITIONED_SEQUENCE_MODE = "partitioned_attention_variable_grid_linear"
_BRIDGE_MARKER = "_sol_h3_partitioned_history_bridge_v1"
_RECEIPT_BRIDGE_MARKER = "_sol_h3_partitioned_receipt_bridge_v1"
_VDN_HISTORY_BRIDGE_MARKER = "_sol_h3_partitioned_vdn_history_bridge_v1"
# Native (pre-partition) video carriers whose Flow replacement closures this
# release recognizes. "source" is the historical uniform reduced-grid carrier.
# "target" is Flow's target-band continuation, whose native sequence is the
# uniform target grid while the partition stays [target head | source tail].
PARTITIONED_NATIVE_CARRIER_GRIDS = ("source", "target")
# Flow target-band domain-uniform execution runs two uniform-grid hidden streams
# inside one block replacement. Each stream carries a canonical equal-grid
# contract; this API version recognizes that replacement as one history identity.
PARTITIONED_DOMAIN_STREAM_API = 1
PARTITIONED_DOMAIN_STREAM_KEY = "h3_flow_partitioned_domain_stream_v1"
PARTITIONED_DOMAIN_UNIFORM_IDENTITY = "h3_flow_partitioned_domain_uniform_v1"
PARTITIONED_DOMAIN_UNIFORM_POLICY = "domain_uniform_v1"
_DOMAIN_STREAM_NAMES = ("target", "source")


def _native_carrier_rows_per_frame(contract, source_rows, target_rows):
    """Return the native carrier rows per frame declared by Flow, or None."""
    if not isinstance(contract, dict):
        return None
    if "native_carrier_grid" not in contract:
        return source_rows if "native_carrier_rows_per_frame" not in contract else None
    if (
        contract.get("native_carrier_grid") != "target"
        or type(contract.get("native_carrier_rows_per_frame")) is not int
        or contract["native_carrier_rows_per_frame"] != target_rows
        or target_rows <= source_rows
    ):
        return None
    return target_rows


def _digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _same_grid_control_contract_valid(contract) -> bool:
    """Recognize only Flow's canonical v1 equal-grid control, including its digest."""
    if not isinstance(contract, dict) or type(contract.get("api")) is not int:
        return False
    names = (
        "video_start", "temporal", "prefix_t", "source_grid_h", "source_grid_w",
        "target_grid_h", "target_grid_w",
    )
    if any(type(contract.get(name)) is not int or contract[name] <= 0 for name in names):
        return False
    start, temporal, prefix_t, source_h, source_w, target_h, target_w = (
        contract[name] for name in names
    )
    if prefix_t >= temporal or (source_h, source_w) != (target_h, target_w):
        return False
    if (
        contract.get("exact_prefix_queries_preserved") is not True
        or contract.get("generated_suffix_queries_preserved") is not True
        or contract.get("heterogeneous_spatial_domains") is not False
    ):
        return False
    rows = source_h * source_w
    prefix_end = start + prefix_t * rows
    sequence_rows = start + temporal * rows
    canonical = {
        "api": 1,
        "topology": "target_prefix_source_suffix",
        "sequence_rows": sequence_rows,
        **{name: contract[name] for name in names},
        "source_rows_per_frame": rows,
        "target_rows_per_frame": rows,
        "prefix_range": [start, prefix_end],
        "suffix_range": [prefix_end, sequence_rows],
        "prefix_log_key_measure": 0.0,
        "nonvideo_log_key_measure": 0.0,
        "suffix_log_key_measure": 0.0,
        "exact_prefix_queries_preserved": True,
        "generated_suffix_queries_preserved": True,
        "heterogeneous_spatial_domains": False,
    }
    if any(contract.get(name) != value for name, value in canonical.items()):
        return False
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return contract.get("semantic_digest") == hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
        or target_rows < source_rows
        or (target_rows == source_rows and not _same_grid_control_contract_valid(contract))
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

    native_rows_per_frame = _native_carrier_rows_per_frame(partition_contract, source_rows, target_rows)
    if (
        native_rows_per_frame is None
        or not 0 < prefix_t < temporal
        or source_rows <= 0
        or target_rows < source_rows
        or len(target_hw) != 2
        or any(value <= 0 or value % 2 for value in target_hw)
        or prefix_rows != prefix_t * target_rows
        or carrier_prefix_rows != prefix_t * native_rows_per_frame
        or partitioned_rows != prefix_rows + (temporal - prefix_t) * source_rows
        or video_start <= 0
        or video_end != video_start + temporal * native_rows_per_frame
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

    if target_rows == source_rows:
        # Equal row products alone do not establish equal physical geometry.
        # Bind the canonical control to the actual latent-grid closure as well.
        if not _same_grid_control_contract_valid(partition_contract):
            return None
        source_h = getattr(plan, "source_h", None)
        source_w = getattr(plan, "source_w", None)
        if (
            type(source_h) is not int
            or type(source_w) is not int
            or (source_h, source_w) != target_hw
            or target_hw != (
                2 * partition_contract["target_grid_h"],
                2 * partition_contract["target_grid_w"],
            )
            or partition_contract["video_start"] != video_start
            or partition_contract["temporal"] != temporal
            or partition_contract["prefix_t"] != prefix_t
            or partition_contract["source_rows_per_frame"] != source_rows
            or partition_contract["sequence_rows"] != partitioned_sequence_rows
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


def _partitioned_flow_domain_replacement_identity(interop, patch, block_index):
    """Recognize Flow's two-stream domain-uniform block replacement exactly."""
    module = str(getattr(patch, "__module__", ""))
    qualname = str(getattr(patch, "__qualname__", ""))
    if not (
        (
            module == "h3_flow_regenerate.partitioned_transformer"
            or module.endswith(".h3_flow_regenerate.partitioned_transformer")
        )
        and qualname.endswith("_domain_uniform_forward.<locals>.wrap.<locals>.call")
    ):
        return None

    values = interop._closure_values(patch)
    required = {"layer", "previous", "streams", "layout", "video_start", "inner", "domain_policy"}
    if values is None or not required.issubset(values):
        return None
    if type(values["layer"]) is not int or values["layer"] != int(block_index):
        return None
    policy = values["domain_policy"]
    streams = values["streams"]
    layout = values["layout"]
    try:
        native_rows = int(layout.seq_len)
        native_segments = tuple(layout.segments)
        video_start = int(values["video_start"])
        inner_blocks = len(values["inner"].blocks)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        policy != PARTITIONED_DOMAIN_UNIFORM_POLICY
        or not isinstance(streams, tuple)
        or len(streams) != len(_DOMAIN_STREAM_NAMES)
        or not native_segments
        or native_segments[-1] != (video_start, native_rows, "video")
        or block_index < 0
        or block_index >= inner_blocks
    ):
        return None

    stream_identity = []
    for expected_name, stream in zip(_DOMAIN_STREAM_NAMES, streams):
        try:
            name = str(stream.name)
            contract = stream.contract
            stream_layout = stream.layout
            leaf = stream.leaf
            head = int(stream.attention_head_t)
            rows = int(stream_layout.seq_len)
            signature = stream_layout.signature
            segments = tuple(stream_layout.segments)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            name != expected_name
            or not _same_grid_control_contract_valid(contract)
            or rows != contract["sequence_rows"]
            or not segments
            or segments[-1] != (contract["video_start"], rows, "video")
            or not isinstance(signature, tuple)
            or len(signature) < 2
            or signature[0] != PARTITIONED_FLOW_IDENTITY
            or signature[-1] != (PARTITIONED_DOMAIN_STREAM_KEY, policy, name)
            or not isinstance(leaf, dict)
            or leaf
            != {
                "api": PARTITIONED_DOMAIN_STREAM_API,
                "policy": policy,
                "stream": name,
                "flow_semantic_digest": contract["semantic_digest"],
                "native_sequence_rows": native_rows,
                "native_video_start": video_start,
            }
            or not contract["prefix_t"] <= head < contract["temporal"]
        ):
            return None
        stream_identity.append(
            (name, contract["semantic_digest"], rows, contract["temporal"], head, repr(signature))
        )

    identity = (
        PARTITIONED_DOMAIN_UNIFORM_IDENTITY,
        policy,
        native_rows,
        video_start,
        repr(getattr(layout, "signature", None)),
        tuple(stream_identity),
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
        PARTITIONED_WEIGHTED_DENSE_CONTRACT,
    )
    from .runtime import _FORWARD, _REQUEST

    if not isinstance(item, tuple) or len(item) != 4 or item[0] != "sol_h3":
        return False
    block = item[1]
    route = item[2]
    fields = item[3]
    if type(block) is not int or block < 0:
        return False
    if route not in {PARTITIONED_DENSE_ROUTE, PARTITIONED_SOL_ROUTE, PARTITIONED_MAPPED_ROUTE}:
        return False
    if not isinstance(fields, tuple) or len(fields) != 15:
        return False
    (
        tag,
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
        # Dense requests may also weigh the whole global sink: (0, end) with end
        # beyond the sink. Sparse requests keep the measure at the sink boundary.
        dense_sink_measure = route == PARTITIONED_DENSE_ROUTE and start == 0 and sink_rows < end <= kv_rows
        if not (sink_rows <= start < end <= kv_rows or dense_sink_measure):
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
        native_dense = execution_mode in {"dense_forced", "dense_warmup"} and kernel_contract is None
        weighted_dense = (
            execution_mode in {"dense_sm120_forced", "dense_sm120_warmup"}
            and kernel_contract == PARTITIONED_WEIGHTED_DENSE_CONTRACT
            and prefix_k_range is not None
            and prefix_log_key_measure < 0.0
        )
        if not (native_dense or weighted_dense):
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
    active = _FORWARD.get()
    if (
        state is None
        or not isinstance(active, tuple)
        or len(active) != 5
        or active[1] is not state
        or type(active[2]) is not int
        or active[2] < 0
        or getattr(state, "partitioned_receipt_evaluation", None) != active[2]
    ):
        return False
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
            identity = _partitioned_flow_replacement_identity(
                interop,
                patch,
                block_index,
            )
            if identity is not None:
                return identity
            return _partitioned_flow_domain_replacement_identity(
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
    "PARTITIONED_DOMAIN_STREAM_API",
    "PARTITIONED_DOMAIN_STREAM_KEY",
    "PARTITIONED_DOMAIN_UNIFORM_IDENTITY",
    "PARTITIONED_FLOW_IDENTITY",
    "PARTITIONED_NATIVE_CARRIER_GRIDS",
    "install_partitioned_history_bridge",
]
