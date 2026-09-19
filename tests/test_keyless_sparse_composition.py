from __future__ import annotations

import pytest
import torch

from sol_h3.keyless_sparse_composition import (
    CONTRACT,
    exact_heads_from_route_trace,
)


def test_k4_composition_contract_is_diagnostic_only():
    assert CONTRACT == "sol-h3-keyless-k4-selected-route-composition-v1"


def test_route_trace_decoder_reads_both_words_per_group():
    trace = torch.zeros((1, 1, 3, 1, 2), dtype=torch.int32)
    trace[0, 0, 0, 0, 0] = 1 << 0
    trace[0, 0, 1, 0, 0] = -(1 << 31)
    trace[0, 0, 2, 0, 1] = 1 << 7

    got = exact_heads_from_route_trace(
        trace,
        q_tile=0,
        block_start=0,
        block_count=40,
    )
    assert got.shape == (3, 40)
    assert got[0, 0]
    assert got[1, 31]
    assert got[2, 39]
    assert int(got.sum().item()) == 3


def test_route_trace_decoder_rejects_non_group_boundary():
    trace = torch.zeros((1, 1, 1, 1, 2), dtype=torch.int32)
    with pytest.raises(ValueError, match="route-group boundary"):
        exact_heads_from_route_trace(
            trace,
            q_tile=0,
            block_start=1,
            block_count=1,
        )
