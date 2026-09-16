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
    if tag != WIRE_TAG or type(schema) is not int or schema != WIRE_SCHEMA:
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


def _merge_block_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent half-open integer block intervals exactly."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def compile_descriptor(validated: ValidatedMap) -> DescriptorPlan | None:
    """Compile exact +/-1 physical K64 neighbors for each requested Q64 tile.

    Affine wire runs are intersected with each Q64 tile and converted directly to
    K64 block intervals.  No per-row position vector is expanded.  The exact
    neighbor union is accepted only when it is one nonempty interval no wider
    than four blocks; disjoint sets are never replaced by their hull.
    """
    if validated.identity_aligned:
        return None
    k_blocks = (validated.kv_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    q_tiles = (validated.q_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    intervals: list[tuple[int, int]] = []
    run_index = 0
    for tile in range(q_tiles):
        tile_q_begin = tile * BLOCK_SIZE
        tile_q_end = min(validated.q_rows, tile_q_begin + BLOCK_SIZE)
        while run_index < len(validated.runs) and validated.runs[run_index][1] <= tile_q_begin:
            run_index += 1
        scan = run_index
        selected_ranges: list[tuple[int, int]] = []
        while scan < len(validated.runs):
            run_q_begin, run_q_end, run_kv_begin = validated.runs[scan]
            if run_q_begin >= tile_q_end:
                break
            overlap_begin = max(tile_q_begin, run_q_begin)
            overlap_end = min(tile_q_end, run_q_end)
            if overlap_begin < overlap_end:
                first_position = run_kv_begin + overlap_begin - run_q_begin
                last_position = run_kv_begin + overlap_end - run_q_begin - 1
                first_block = first_position // BLOCK_SIZE
                last_block = last_position // BLOCK_SIZE
                selected_ranges.append(
                    (max(0, first_block - 1), min(k_blocks, last_block + 2))
                )
            scan += 1

        merged = _merge_block_intervals(selected_ranges)
        if not merged or len(merged) != 1:
            raise MappingUnavailable("fragmented")
        start, end = merged[0]
        if end <= start:
            raise MappingUnavailable("fragmented")
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
    if (
        getattr(summary, "tag", None) != "vdn_query_position_plan_v1"
        or type(getattr(summary, "schema", None)) is not int
        or getattr(summary, "schema", None) != 1
        or getattr(summary, "mode", None) != "grouped"
    ):
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
    """Return a request-local cached int32 interval tensor.

    CUDA cache hits are synchronized to the current stream with the creation
    event. CPU descriptors use the same request-local cache without touching any
    CUDA API, which keeps CPU oracle/integration paths valid.
    """
    import torch

    device = torch.device(device)
    cache = state.mapped_descriptor_cache
    key = descriptor_cache_key(plan, device)
    hit = cache.get(key)
    if device.type == "cuda":
        current = torch.cuda.current_stream(device)
        if hit is not None:
            cache.move_to_end(key)
            tensor, event, _size = hit
            if event is None:
                raise RuntimeError("CUDA mapped-neighbor descriptor cache entry is missing its stream event")
            current.wait_event(event)
            tensor.record_stream(current)
            return tensor
        if torch.cuda.is_current_stream_capturing():
            raise MappingUnavailable("backend")
    else:
        current = None
        if hit is not None:
            cache.move_to_end(key)
            tensor, event, _size = hit
            if event is not None:
                raise RuntimeError("non-CUDA mapped-neighbor descriptor cache entry has a CUDA stream event")
            return tensor

    tensor = torch.tensor(plan.intervals, dtype=torch.int32, device=device).contiguous()
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise RuntimeError("mapped-neighbor descriptor tensor has an invalid shape")
    event = None
    if current is not None:
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
