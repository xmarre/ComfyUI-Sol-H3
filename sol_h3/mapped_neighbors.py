"""Validate VDN provider-v4 maps and compile bounded additive K64 metadata.

All semantic validation and interval derivation are CPU-only.  The only CUDA
allocation is the tiny int32 [ceil(Tq/64), 2] interval tensor consumed by the
SM120 kernel; descriptor values never enter a JIT specialization key.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

WIRE_TAG = "vdn_query_positions"
WIRE_SCHEMA = 1
POLICY = "k64-union-radius1-additive-interval4-v1"
RECEIPT_TAG = "vdn_mapped_neighbor_v1"
MAX_INTERVAL_WIDTH = 4
BLOCK_SIZE = 64
MAX_CACHE_ENTRIES = 64
MAX_DEVICE_CACHE_BYTES = 4 * 1024 * 1024
_INT32_MAX = 2**31 - 1


class MappingUnavailable(ValueError):
    """Validated local call cannot use the bounded mapped-neighbor route."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ValidatedMap:
    owner_generation: str
    plan_digest: str
    group_index: int
    q_rows: int
    kv_rows: int
    sink_rows: int
    runs: tuple[tuple[int, int, int], ...]
    map_digest: str
    identity_aligned: bool


@dataclass(frozen=True)
class DescriptorPlan:
    validated: ValidatedMap
    intervals: tuple[tuple[int, int], ...]
    descriptor_digest: str


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > _INT32_MAX:
        raise MappingUnavailable("schema")
    return value


def _digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and value == value.lower() and all(
        char in "0123456789abcdef" for char in value
    )


def validate_wire_map(value: Any, *, q_rows: int, kv_rows: int, sink_rows: int) -> ValidatedMap:
    """Validate the immutable v4 wire tuple without importing VDN."""
    if not isinstance(value, tuple) or len(value) != 9:
        raise MappingUnavailable("missing" if value is None else "schema")
    tag, schema, owner, plan_digest, group_index, wire_q, wire_kv, wire_sink, raw_runs = value
    if tag != WIRE_TAG or schema != WIRE_SCHEMA:
        raise MappingUnavailable("schema")
    if not isinstance(owner, str) or not owner or not _digest(plan_digest):
        raise MappingUnavailable("owner")
    group_index = _strict_int(group_index, "group_index")
    wire_q = _strict_int(wire_q, "q_rows", minimum=1)
    wire_kv = _strict_int(wire_kv, "kv_rows", minimum=1)
    wire_sink = _strict_int(wire_sink, "sink_rows")
    if type(q_rows) is not int or type(kv_rows) is not int or type(sink_rows) is not int:
        raise MappingUnavailable("domain")
    if (wire_q, wire_kv, wire_sink) != (q_rows, kv_rows, sink_rows) or not 0 <= wire_sink <= wire_kv:
        raise MappingUnavailable("domain")
    if not isinstance(raw_runs, tuple) or not raw_runs:
        raise MappingUnavailable("schema")

    runs: list[tuple[int, int, int]] = []
    expected_q = 0
    previous_position = -1
    identity_aligned = q_rows == kv_rows
    for raw in raw_runs:
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise MappingUnavailable("schema")
        q_begin = _strict_int(raw[0], "q_begin")
        q_end = _strict_int(raw[1], "q_end", minimum=1)
        kv_begin = _strict_int(raw[2], "kv_begin")
        if q_begin != expected_q or q_end <= q_begin or q_end > q_rows:
            raise MappingUnavailable("schema")
        span = q_end - q_begin
        if kv_begin + span > kv_rows:
            raise MappingUnavailable("domain")
        if kv_begin <= previous_position:
            raise MappingUnavailable("domain")
        # Every represented row is affine with slope one. This implies strict
        # increase inside the run; the boundary check above proves it across runs.
        previous_position = kv_begin + span - 1
        identity_aligned = identity_aligned and kv_begin == q_begin
        runs.append((q_begin, q_end, kv_begin))
        expected_q = q_end
    if expected_q != q_rows:
        raise MappingUnavailable("schema")

    map_digest = _sha256_json(
        {
            "tag": WIRE_TAG,
            "schema": WIRE_SCHEMA,
            "owner_generation": owner,
            "plan_digest": plan_digest,
            "group_index": group_index,
            "q_rows": wire_q,
            "kv_rows": wire_kv,
            "sink_rows": wire_sink,
            "runs": runs,
        }
    )
    return ValidatedMap(
        owner_generation=owner,
        plan_digest=plan_digest,
        group_index=group_index,
        q_rows=wire_q,
        kv_rows=wire_kv,
        sink_rows=wire_sink,
        runs=tuple(runs),
        map_digest=map_digest,
        identity_aligned=identity_aligned,
    )


def _position_for_row(runs: tuple[tuple[int, int, int], ...], row: int, run_index: int):
    while run_index < len(runs) and row >= runs[run_index][1]:
        run_index += 1
    if run_index >= len(runs):
        raise MappingUnavailable("schema")
    q_begin, q_end, kv_begin = runs[run_index]
    if not q_begin <= row < q_end:
        raise MappingUnavailable("schema")
    return kv_begin + row - q_begin, run_index


def compile_descriptor(validated: ValidatedMap) -> DescriptorPlan | None:
    """Compile exact +/-1 physical K64 neighbors for each requested Q64 tile.

    We never take the hull of disjoint sets.  A Q tile is accepted only when its
    exact union is one non-empty contiguous interval of at most four K64 blocks.
    Identity-aligned square maps return ``None`` so the established ordinal route
    remains byte-for-byte the ordinary path.
    """
    if validated.identity_aligned:
        return None
    k_blocks = (validated.kv_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    q_tiles = (validated.q_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    intervals: list[tuple[int, int]] = []
    run_index = 0
    for tile in range(q_tiles):
        q_begin = tile * BLOCK_SIZE
        q_end = min(validated.q_rows, q_begin + BLOCK_SIZE)
        represented: set[int] = set()
        for row in range(q_begin, q_end):
            position, run_index = _position_for_row(validated.runs, row, run_index)
            represented.add(position // BLOCK_SIZE)
        selected: set[int] = set()
        for block in represented:
            for candidate in (block - 1, block, block + 1):
                if 0 <= candidate < k_blocks:
                    selected.add(candidate)
        if not selected:
            raise MappingUnavailable("fragmented")
        ordered = sorted(selected)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise MappingUnavailable("fragmented")
        start, end = ordered[0], ordered[-1] + 1
        if end - start > MAX_INTERVAL_WIDTH:
            raise MappingUnavailable("interval_width")
        intervals.append((start, end))

    descriptor_digest = _sha256_json(
        {
            "policy": POLICY,
            "map_digest": validated.map_digest,
            "q_rows": validated.q_rows,
            "kv_rows": validated.kv_rows,
            "intervals": intervals,
        }
    )
    return DescriptorPlan(validated, tuple(intervals), descriptor_digest)


def validate_preflight_summary(summary: Any, wire: tuple[Any, ...], plan: DescriptorPlan | None) -> bool:
    """Require the current VDN forward's pure preflight summary to own this map."""
    if summary is None:
        return False
    if getattr(summary, "tag", None) != "vdn_query_position_plan_v1" or getattr(summary, "schema", None) != 1:
        return False
    if getattr(summary, "owner_generation", None) != wire[2] or getattr(summary, "plan_digest", None) != wire[3]:
        return False
    groups = getattr(summary, "groups", None)
    if not isinstance(groups, tuple):
        return False
    group_index = wire[4]
    if type(group_index) is not int or group_index < 0 or group_index >= len(groups):
        return False
    if groups[group_index] != wire:
        return False
    if plan is not None and plan.validated.group_index != group_index:
        return False
    return True


def descriptor_cache_key(plan: DescriptorPlan, device: Any) -> tuple[Any, ...]:
    v = plan.validated
    return (
        WIRE_SCHEMA,
        v.owner_generation,
        v.plan_digest,
        v.group_index,
        v.q_rows,
        v.kv_rows,
        v.sink_rows,
        v.runs,
        POLICY,
        str(device),
    )


def device_descriptor(state: Any, plan: DescriptorPlan, device: Any):
    """Return a stream-safe cached int32 interval tensor for one sampling request."""
    import torch

    cache = state.mapped_descriptor_cache
    key = descriptor_cache_key(plan, device)
    hit = cache.get(key)
    current = torch.cuda.current_stream(device)
    if hit is not None:
        cache.move_to_end(key)
        tensor, event, size = hit
        current.wait_event(event)
        tensor.record_stream(current)
        return tensor
    if torch.cuda.is_current_stream_capturing():
        raise MappingUnavailable("backend")

    tensor = torch.tensor(plan.intervals, dtype=torch.int32, device=device).contiguous()
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise RuntimeError("mapped-neighbor descriptor tensor has an invalid shape")
    event = torch.cuda.Event()
    event.record(current)
    tensor.record_stream(current)
    size = int(tensor.numel()) * int(tensor.element_size())
    if size > MAX_DEVICE_CACHE_BYTES:
        raise MappingUnavailable("metadata_budget")
    cache[key] = (tensor, event, size)
    state.mapped_descriptor_bytes += size
    while len(cache) > MAX_CACHE_ENTRIES or state.mapped_descriptor_bytes > MAX_DEVICE_CACHE_BYTES:
        _, (_old_tensor, _old_event, old_size) = cache.popitem(last=False)
        state.mapped_descriptor_bytes -= old_size
    return tensor


def receipt_fields(plan: DescriptorPlan, *, kernel_contract: str) -> tuple[Any, ...]:
    v = plan.validated
    return (
        RECEIPT_TAG,
        v.owner_generation,
        v.plan_digest,
        v.group_index,
        v.q_rows,
        v.kv_rows,
        v.sink_rows,
        v.map_digest,
        plan.descriptor_digest,
        POLICY,
        kernel_contract,
        True,
    )


__all__ = [
    "BLOCK_SIZE",
    "DescriptorPlan",
    "MAX_CACHE_ENTRIES",
    "MAX_DEVICE_CACHE_BYTES",
    "MappingUnavailable",
    "POLICY",
    "RECEIPT_TAG",
    "ValidatedMap",
    "compile_descriptor",
    "descriptor_cache_key",
    "device_descriptor",
    "receipt_fields",
    "validate_preflight_summary",
    "validate_wire_map",
]
