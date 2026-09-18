from __future__ import annotations

import pytest
import torch

from sol_h3.keyless_selector import (
    CONTRACT,
    route_trace_metrics,
    route_trace_shape,
    run_materialized_exact_selector_isolation,
    sink_block_range,
    threshold_from_route_centroids,
)


def test_k3_contract_is_calibration_only_identity():
    assert CONTRACT == "sol-h3-keyless-selector-k3-calibration-v1"


@pytest.mark.parametrize(
    ("q_rows", "kv_rows", "expected"),
    [
        (63, 63, (1, 1, 56, 1, 2)),
        (64, 64, (1, 1, 56, 1, 2)),
        (65, 65, (1, 2, 56, 1, 2)),
        (65, 4096, (1, 2, 56, 1, 2)),
        (65, 4097, (1, 2, 56, 2, 2)),
    ],
)
def test_route_trace_shape_tracks_q_tiles_and_kv_route_groups(
    q_rows,
    kv_rows,
    expected,
):
    assert route_trace_shape(
        batch=1,
        q_rows=q_rows,
        kv_rows=kv_rows,
        heads=56,
    ) == expected


def test_route_trace_shape_rejects_nonpositive_geometry():
    with pytest.raises(ValueError, match="positive"):
        route_trace_shape(batch=1, q_rows=0, kv_rows=64, heads=56)


@pytest.mark.parametrize(
    ("kv_rows", "sink_start", "sink_tokens", "expected"),
    [
        (257, 0, 0, (5, 5)),
        (257, 0, 128, (0, 2)),
        (257, 64, 1, (1, 2)),
        (257, 128, 129, (2, 5)),
        (4097, 1024, 192, (16, 19)),
    ],
)
def test_sink_block_range_matches_physical_v64_blocks(
    kv_rows,
    sink_start,
    sink_tokens,
    expected,
):
    assert sink_block_range(
        kv_rows=kv_rows,
        sink_start=sink_start,
        sink_tokens=sink_tokens,
    ) == expected


def test_sink_block_range_rejects_out_of_domain_rows():
    with pytest.raises(ValueError, match="exceeds"):
        sink_block_range(kv_rows=257, sink_start=256, sink_tokens=2)


def test_route_trace_metrics_handles_signed_ballots_and_ignores_invalid_tail_bits():
    shape = route_trace_shape(batch=1, q_rows=1, kv_rows=4097, heads=1)
    got = torch.zeros(shape, dtype=torch.int32)
    want = torch.zeros_like(got)

    # Block 31 is the sign bit of the first int32 word.
    got[0, 0, 0, 0, 0] = -2147483648
    want[0, 0, 0, 0, 0] = -2147483648

    # Block 64 is the only valid block in route group 1.
    got[0, 0, 0, 1, 0] = 1
    want[0, 0, 0, 1, 0] = 1

    # Bit 1 in route group 1 would describe block 65, which is outside 4097 rows.
    # The comparator must not treat diagnostic scratch beyond the physical domain
    # as a routing mismatch.
    got[0, 0, 0, 1, 0] = 3

    metrics = route_trace_metrics(got, want, kv_rows=4097)
    assert metrics["equal"] is True
    assert metrics["differing_bits"] == 0
    assert metrics["total_routing_bits"] == 65
    assert metrics["candidate_exact_bits"] == 2
    assert metrics["reference_exact_bits"] == 2


def test_route_trace_metrics_reports_one_valid_bit_difference():
    got = torch.zeros((1, 1, 1, 1, 2), dtype=torch.int32)
    want = torch.zeros_like(got)
    got[0, 0, 0, 0, 1] = 1  # physical block 32

    metrics = route_trace_metrics(got, want, kv_rows=64)
    assert metrics["equal"] is False
    assert metrics["differing_bits"] == 1
    assert metrics["total_routing_bits"] == 64
    assert metrics["differing_bit_fraction"] == pytest.approx(1 / 64)
    assert metrics["candidate_exact_bits"] == 1
    assert metrics["reference_exact_bits"] == 0


def test_threshold_binding_rejects_wrong_rc_block_count_before_cuda_import():
    q = torch.zeros((1, 65, 56, 128), dtype=torch.bfloat16)
    rc = torch.zeros((1, 1, 56, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="block count"):
        threshold_from_route_centroids(
            q,
            rc,
            kv_rows=65,
            tau=1.0,
            scale=128 ** -0.5,
        )


def test_threshold_binding_fails_closed_on_cpu_before_vendor_execution():
    q = torch.zeros((1, 65, 56, 128), dtype=torch.bfloat16)
    rc = torch.zeros((1, 2, 56, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one CUDA device"):
        threshold_from_route_centroids(
            q,
            rc,
            kv_rows=65,
            tau=1.0,
            scale=128 ** -0.5,
        )


def test_selector_executor_fails_closed_on_cpu():
    q = torch.zeros((1, 65, 56, 128), dtype=torch.bfloat16)
    route = torch.zeros((1, 129, 56, 128), dtype=torch.bfloat16)
    raw_v = torch.zeros_like(route)
    rc = torch.zeros((1, 3, 56, 128), dtype=torch.bfloat16)
    vc = torch.zeros_like(rc)
    threshold = torch.zeros((1, 2, 56), dtype=torch.float32)

    with pytest.raises(ValueError, match="requires CUDA"):
        run_materialized_exact_selector_isolation(
            q,
            route,
            raw_v,
            rc,
            vc,
            threshold,
            scale=128 ** -0.5,
        )
