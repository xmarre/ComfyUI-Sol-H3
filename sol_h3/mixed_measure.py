"""Flow Mixed-Grid attention-measure contract for rectangular Sol-Attn.

Flow's mixed transformer stream keeps an exact target-grid protected prefix but
uses a lower-density source-grid generated suffix. Treating every packed video
row as equal softmax measure overweights each protected prefix frame in direct
attention. This module validates Flow's independent measure contract and builds
a deterministic K/V-only representative set at the source-grid spatial density.

Queries are never reduced. Non-video K/V rows and all source-grid suffix K/V
rows are preserved byte-for-byte and in order. Only the denser target-prefix
K/V rows are stratified by nearest native MiniMax-H3 area-normalized spatial
coordinate.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch

FLOW_MIXED_MEASURE_KEY = "h3_flow_mixed_grid_attention_measure_v1"
FLOW_MIXED_MEASURE_API = 1
FLOW_MIXED_MEASURE_MODE = "prefix_kv_stratified_subsample"


def _axis_coordinates(grid_h: int, grid_w: int, *, axis: str) -> tuple[float, ...]:
    if axis not in {"h", "w"}:
        raise ValueError("axis must be h or w")
    latent_h = grid_h * 2
    latent_w = grid_w * 2
    latent_dim = latent_h if axis == "h" else latent_w
    count = grid_h if axis == "h" else grid_w
    area = math.sqrt(latent_h * latent_w)
    ratio = latent_dim / area
    return tuple((i * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0 for i in range(count))


@lru_cache(maxsize=64)
def representative_spatial_indices(
    source_grid_h: int,
    source_grid_w: int,
    prefix_grid_h: int,
    prefix_grid_w: int,
) -> tuple[int, ...]:
    """Map each source-grid patch coordinate to its nearest target-prefix row."""
    values = (source_grid_h, source_grid_w, prefix_grid_h, prefix_grid_w)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("mixed-grid spatial grid sizes must be positive integers")
    if source_grid_h > prefix_grid_h or source_grid_w > prefix_grid_w:
        raise ValueError("source grid cannot exceed target-prefix grid")
    if source_grid_h * source_grid_w >= prefix_grid_h * prefix_grid_w:
        raise ValueError("attention-measure repair requires a strictly denser target prefix")

    source_h = _axis_coordinates(source_grid_h, source_grid_w, axis="h")
    source_w = _axis_coordinates(source_grid_h, source_grid_w, axis="w")
    prefix_h = _axis_coordinates(prefix_grid_h, prefix_grid_w, axis="h")
    prefix_w = _axis_coordinates(prefix_grid_h, prefix_grid_w, axis="w")

    h_index = tuple(min(range(prefix_grid_h), key=lambda j: (abs(prefix_h[j] - value), j)) for value in source_h)
    w_index = tuple(min(range(prefix_grid_w), key=lambda j: (abs(prefix_w[j] - value), j)) for value in source_w)
    if len(set(h_index)) != source_grid_h or len(set(w_index)) != source_grid_w:
        raise RuntimeError("nearest mixed-grid representative mapping is not one-to-one")
    rows = tuple(y * prefix_grid_w + x for y in h_index for x in w_index)
    if len(rows) != source_grid_h * source_grid_w or len(set(rows)) != len(rows):
        raise RuntimeError("mixed-grid representative row mapping is not unique")
    return rows


def validate_measure_contract(measure, external, *, q_rows: int, kv_rows: int) -> dict:
    """Validate the Flow measure contract against the already-validated API-2 stream."""
    if not isinstance(measure, dict):
        raise RuntimeError("Flow mixed-grid attention-measure contract must be a dictionary")
    if measure.get("api") != FLOW_MIXED_MEASURE_API or measure.get("mode") != FLOW_MIXED_MEASURE_MODE:
        raise RuntimeError("Flow mixed-grid attention-measure API/mode is unsupported")
    if not isinstance(external, dict) or external.get("api") != 2:
        raise RuntimeError("Flow mixed-grid attention measure requires external-sequence API 2")

    integer_fields = (
        "video_start",
        "sequence_rows",
        "temporal",
        "prefix_t",
        "source_grid_h",
        "source_grid_w",
        "prefix_grid_h",
        "prefix_grid_w",
        "source_rows_per_frame",
        "prefix_rows_per_frame",
        "expected_kv_rows",
    )
    if any(type(measure.get(name)) is not int for name in integer_fields):
        raise RuntimeError("Flow mixed-grid attention-measure counts must be integers")
    if measure.get("exact_prefix_queries_preserved") is not True or measure.get("suffix_kv_unchanged") is not True:
        raise RuntimeError("Flow mixed-grid attention-measure preservation contract is incomplete")

    for name in (
        "video_start",
        "sequence_rows",
        "temporal",
        "prefix_t",
        "source_rows_per_frame",
        "prefix_rows_per_frame",
    ):
        if measure[name] != external.get(name):
            raise RuntimeError(f"Flow mixed-grid attention-measure {name} disagrees with external sequence")

    start = measure["video_start"]
    temporal = measure["temporal"]
    prefix_t = measure["prefix_t"]
    source_rows = measure["source_rows_per_frame"]
    prefix_rows = measure["prefix_rows_per_frame"]
    if q_rows != measure["sequence_rows"] or kv_rows != measure["sequence_rows"]:
        raise RuntimeError("Flow mixed-grid attention measure expects the unreduced mixed QKV domain")
    if not 0 < start < q_rows or not 0 < prefix_t < temporal:
        raise RuntimeError("Flow mixed-grid attention-measure temporal geometry is invalid")
    if source_rows != measure["source_grid_h"] * measure["source_grid_w"]:
        raise RuntimeError("Flow mixed-grid source grid does not match rows per frame")
    if prefix_rows != measure["prefix_grid_h"] * measure["prefix_grid_w"]:
        raise RuntimeError("Flow mixed-grid prefix grid does not match rows per frame")
    if not 0 < source_rows < prefix_rows:
        raise RuntimeError("Flow mixed-grid attention measure requires a denser protected prefix")
    expected_mixed = start + prefix_t * prefix_rows + (temporal - prefix_t) * source_rows
    expected_kv = start + temporal * source_rows
    if expected_mixed != q_rows or measure["expected_kv_rows"] != expected_kv:
        raise RuntimeError("Flow mixed-grid attention-measure row accounting is inconsistent")
    if external.get("native_sequence_rows") != expected_kv:
        raise RuntimeError("Flow mixed-grid attention-measure K/V rows must equal the native carrier sequence")

    representatives = representative_spatial_indices(
        measure["source_grid_h"],
        measure["source_grid_w"],
        measure["prefix_grid_h"],
        measure["prefix_grid_w"],
    )
    return {
        "video_start": start,
        "temporal": temporal,
        "prefix_t": prefix_t,
        "source_rows_per_frame": source_rows,
        "prefix_rows_per_frame": prefix_rows,
        "expected_kv_rows": expected_kv,
        "representative_spatial_indices": representatives,
    }


def kv_gather_indices(validated: dict, *, device) -> torch.Tensor:
    """Return exact mixed-stream K/V row indices in native-carrier density/order."""
    start = validated["video_start"]
    prefix_t = validated["prefix_t"]
    prefix_rows = validated["prefix_rows_per_frame"]
    source_rows = validated["source_rows_per_frame"]
    temporal = validated["temporal"]
    reps = validated["representative_spatial_indices"]

    indices = list(range(start))
    for frame in range(prefix_t):
        base = start + frame * prefix_rows
        indices.extend(base + index for index in reps)
    suffix_start = start + prefix_t * prefix_rows
    suffix_rows = (temporal - prefix_t) * source_rows
    indices.extend(range(suffix_start, suffix_start + suffix_rows))
    if len(indices) != validated["expected_kv_rows"] or len(set(indices)) != len(indices):
        raise RuntimeError("Flow mixed-grid attention-measure K/V gather is inconsistent")
    return torch.tensor(indices, dtype=torch.long, device=device)


def reduce_kv(k: torch.Tensor, v: torch.Tensor, validated: dict):
    """Reduce BHTD K/V only; return tensors plus row-accounting telemetry."""
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise RuntimeError("Flow mixed-grid attention measure requires matching BHTD K/V")
    indices = kv_gather_indices(validated, device=k.device)
    reduced_k = k.index_select(2, indices)
    reduced_v = v.index_select(2, indices)
    expected = validated["expected_kv_rows"]
    if reduced_k.shape[2] != expected or reduced_v.shape[2] != expected:
        raise RuntimeError("Flow mixed-grid attention-measure reduction returned wrong K/V row count")
    return (
        reduced_k,
        reduced_v,
        {
            "kv_rows_before": int(k.shape[2]),
            "kv_rows_after": int(expected),
            "kv_rows_removed": int(k.shape[2] - expected),
        },
    )
