from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Mapping

import torch

ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_CAPABILITIES_KEY = "attention_measure_capabilities_v1"
PROVIDER_IDENTITY = "sol_h3.sm120"


@dataclass(frozen=True, slots=True)
class BoundMeasurePlan:
    semantic_digest: str
    implementation_profile: str
    owner_generation: str
    q_rows: int
    kv_rows: int
    exact_k_blocks: tuple[int, int]
    exact_range_digest: str
    key_log_measure: torch.Tensor


def _int(name, value, minimum=0):
    if type(value) is not int:
        raise TypeError(f"attention measure {name} must be an integer")
    if value < minimum:
        raise ValueError(f"attention measure {name} must be >= {minimum}")
    return value


def _grid(name, value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"attention measure {name} must be a two-element grid")
    return _int(f"{name}[0]", value[0], 1), _int(f"{name}[1]", value[1], 1)


def normalize(request: Mapping[str, Any]):
    if not isinstance(request, Mapping):
        raise TypeError("attention_measure_v1 must be a mapping")
    expected = {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise ValueError(f"attention measure {key} must be {value!r}")
    q_rows = _int("q_rows", request.get("q_rows"), 1)
    kv_rows = _int("kv_rows", request.get("kv_rows"), 1)
    video_start = _int("video_start", request.get("video_start"))
    temporal = _int("temporal", request.get("temporal"), 2)
    prefix_t = _int("prefix_t", request.get("prefix_t"), 1)
    source_grid = _grid("source_grid", request.get("source_grid"))
    prefix_grid = _grid("prefix_grid", request.get("prefix_grid"))
    source_rows, prefix_rows = math.prod(source_grid), math.prod(prefix_grid)
    if q_rows != kv_rows:
        raise ValueError("Sol-H3 weighted measure retains the full square Q/K/V domain")
    if video_start >= q_rows or prefix_t >= temporal or source_rows >= prefix_rows:
        raise ValueError("attention measure mixed-grid geometry is invalid")
    if q_rows != video_start + prefix_t * prefix_rows + (temporal - prefix_t) * source_rows:
        raise ValueError("attention measure row count is inconsistent")
    raw_segments = request.get("segments")
    if not isinstance(raw_segments, (list, tuple)) or not raw_segments:
        raise TypeError("attention measure segments must be a nonempty list")
    segments, cursor = [], 0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise TypeError(f"attention measure segments[{index}] must be a mapping")
        start = _int(f"segments[{index}].start", raw.get("start"))
        stop = _int(f"segments[{index}].stop", raw.get("stop"))
        num = _int(f"segments[{index}].mass_num", raw.get("mass_num"), 1)
        den = _int(f"segments[{index}].mass_den", raw.get("mass_den"), 1)
        if start != cursor or stop <= start or stop > kv_rows:
            raise ValueError("attention measure segments must be contiguous, sorted, nonempty and in range")
        ratio = Fraction(num, den)
        segments.append({"start": start, "stop": stop, "mass_num": ratio.numerator, "mass_den": ratio.denominator})
        cursor = stop
    if cursor != kv_rows:
        raise ValueError("attention measure segments do not cover the K domain")
    weighted_start = video_start
    weighted_stop = video_start + prefix_t * prefix_rows
    weighted_ratio = Fraction(source_rows, prefix_rows)
    for segment in segments:
        start, stop = segment["start"], segment["stop"]
        if start < weighted_start < stop or start < weighted_stop < stop:
            raise ValueError("attention measure segment crosses a protected-prefix boundary")
        overlap = max(start, weighted_start) < min(stop, weighted_stop)
        ratio = Fraction(segment["mass_num"], segment["mass_den"])
        if overlap and ratio != weighted_ratio:
            raise ValueError("protected-prefix measure must be source_rows/prefix_rows")
        if not overlap and ratio != 1:
            raise ValueError("conditioning and generated-suffix measure must remain one")
        if not math.isfinite(math.log(float(ratio))):
            raise ValueError("attention measure bias must be finite")
    return {**expected, "q_rows": q_rows, "kv_rows": kv_rows, "video_start": video_start,
            "temporal": temporal, "prefix_t": prefix_t, "source_grid": list(source_grid),
            "prefix_grid": list(prefix_grid), "segments": segments}


def digest(request):
    data = json.dumps(normalize(request), sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def bind(request, *, q_rows, kv_rows, device, block_size=64, existing_sink=(0, 0), owner_generation="sol_h3"):
    normalized = normalize(request)
    if normalized["q_rows"] != q_rows or normalized["kv_rows"] != kv_rows:
        raise ValueError("attention measure row counts are stale for current Q/K/V")
    weighted = [s for s in normalized["segments"] if (s["mass_num"], s["mass_den"]) != (1, 1)]
    first = min(s["start"] for s in weighted) // block_size
    last = (max(s["stop"] for s in weighted) + block_size - 1) // block_size
    if existing_sink[0] != existing_sink[1]:
        first, last = min(first, existing_sink[0]), max(last, existing_sink[1])
    bias = torch.empty(kv_rows, dtype=torch.float32, device=device)
    for segment in normalized["segments"]:
        bias[segment["start"]:segment["stop"]] = math.log(segment["mass_num"] / segment["mass_den"])
    if not torch.isfinite(bias).all():
        raise ValueError("attention measure materialization produced a non-finite bias")
    range_digest = hashlib.sha256(f"{first}:{last}".encode("ascii")).hexdigest()
    return BoundMeasurePlan(digest(normalized), "weighted_exact_blocks_v1", str(owner_generation), q_rows, kv_rows,
                            (first, last), range_digest, bias.contiguous())


class Capability:
    api = 1
    operator = "key_log_measure"
    implementation_profile = "weighted_exact_blocks_v1"
    provider_identity = PROVIDER_IDENTITY

    def __init__(self, prepare):
        self._prepare = prepare

    def prepare(self, request, execution_context):
        return self._prepare(request, execution_context)
