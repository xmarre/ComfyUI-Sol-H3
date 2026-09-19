from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sol_h3.keyless_route_summary import (
    BOUNDED_SPILL_CONTRACT,
    FUSED_V5_CONTROL_CONTRACT,
    NORM_EPS,
    SOL_REDUCTION_CHUNK_BLOCKS,
    SOL_REDUCTION_CONTRACT,
    route_centroid_bounded_spill_diagnostic,
    route_centroid_sol_reduction_config_diagnostic,
    route_centroid_split_rms_diagnostic,
    route_summary_bounded_spill_diagnostic,
    route_summary_sol_reduction,
    route_summary_sol_reduction_v5_diagnostic,
)


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "sol_h3"
        / "keyless_route_summary.py"
    ).read_text(encoding="utf-8")


def test_bounded_spill_contract_is_versioned():
    assert BOUNDED_SPILL_CONTRACT == "sol-h3-keyless-k1-bounded-spill-v1"


def test_sol_reduction_k1_contract_is_versioned():
    assert (
        SOL_REDUCTION_CONTRACT
        == "sol-h3-keyless-route-summary-sol-reduction-v6-bounded-b8"
    )


def test_sol_reduction_k1_fixed_chunk_contract_is_versioned():
    assert SOL_REDUCTION_CHUNK_BLOCKS == 8
    assert (
        FUSED_V5_CONTROL_CONTRACT
        == "sol-h3-keyless-route-summary-sol-reduction-v5-rejected-control-v1"
    )


def test_sol_reduction_k1_v6_reuses_bounded_exact_reducer_for_rc_and_vc():
    source = _source()
    bounded = source[
        source.index("def route_summary_bounded_spill_diagnostic"):
        source.index("def route_centroid_sol_reduction_config_diagnostic")
    ]
    production = source[
        source.index("def route_summary_sol_reduction("):
        source.index("def route_summary(")
    ]
    assert "from ._vendor.sol_attn.preprocess import _reduce_kv_kernel" in bounded
    assert "_reduce_kv_kernel[grid](" in bounded
    assert "route_desc," in bounded
    assert "raw_desc," in bounded
    assert "chunk_blocks=SOL_REDUCTION_CHUNK_BLOCKS" in production
    assert "SOL_REDUCTION_CHUNK_BLOCKS = 8" in source
    assert "torch.empty_like(v)" not in production


def test_rejected_v5_control_keeps_historical_fused_reduction_for_evidence():
    source = _source()
    rc_kernel = source[
        source.index("def _reduce_route_rc_sol_kernel"):
        source.index("def _reduce_route_vc_kernel")
    ]
    assert "mean_square = tl.sum(raw * raw, axis=1) / D" not in rc_kernel
    assert "route = tl.where(" in rc_kernel
    assert "summary = tl.sum(route, axis=0) / block_len" in rc_kernel
    assert "tl.sum(route.to(tl.float32), axis=0)" not in rc_kernel
    assert ").to(tl.bfloat16)\n\n        summary = tl.sum(route, axis=0)" in rc_kernel
    assert "lane_offsets = tl.arange(0, 32)" in rc_kernel
    assert "lane_sum = tl.fma(lane3, lane3, lane_sum)" in rc_kernel
    assert "mean_square = tl.sum(lane_sum, axis=1) / D" in rc_kernel
    assert "torch.empty_like(v)" not in source[
        source.index("def route_summary_sol_reduction("):
        source.index("def route_summary(")
    ]


def test_sol_reduction_v5_control_is_separate_from_v6_candidate():
    source = _source()
    assert "def materialized_sol_reduction_v4_route_diagnostic(" in source
    assert "def route_summary_sol_reduction_v5_diagnostic(" in source
    v5_control = source[
        source.index("def route_summary_sol_reduction_v5_diagnostic"):
        source.index("def route_summary_sol_reduction(")
    ]
    production = source[
        source.index("def route_summary_sol_reduction("):
        source.index("def route_summary(")
    ]
    assert "_reduce_route_rc_sol_kernel[grid](" in v5_control
    assert "scratch_kc5 = torch.empty_like(rc5)" in v5_control
    assert "route_summary_bounded_spill_diagnostic(" in production
    assert "materialized_sol_reduction_v4_route_diagnostic(" not in production


def test_split_rms_diagnostic_uses_only_row_scalar_scratch():
    source = _source()
    split = source[
        source.index("def route_centroid_split_rms_diagnostic"):
        source.index("def route_centroid_sol_reduction_config_diagnostic")
    ]
    assert 'inv_rms = torch.empty((rows, heads)' in split
    assert "dtype=torch.float32" in split
    assert "_compute_route_inv_rms_kernel[grid](" in split
    assert "_reduce_route_rc_from_inv_rms_kernel[grid](" in split
    assert "torch.empty_like(v)" not in split


def test_split_rms_diagnostic_fails_closed_on_cpu():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.bfloat16)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        route_centroid_split_rms_diagnostic(v, weight, NORM_EPS, rope)


def test_bounded_spill_diagnostic_uses_one_route_block_and_exact_reducer():
    source = _source()
    bounded = source[
        source.index("def route_centroid_bounded_spill_diagnostic"):
        source.index("def route_centroid_sol_reduction_config_diagnostic")
    ]
    assert "for block in range(blocks):" in bounded
    assert "end = min(rows, start + BLOCK_SIZE)" in bounded
    assert "materialized_sol_reduction_v4_route_diagnostic(" in bounded
    assert "_reduce_kv_kernel[grid](" in bounded
    assert '"max_route_tile_bytes"' in bounded
    assert '"max_working_scratch_bytes"' in bounded
    assert "torch.empty_like(v)" not in bounded


def test_bounded_spill_diagnostic_fails_closed_on_cpu():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.bfloat16)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        route_centroid_bounded_spill_diagnostic(v, weight, NORM_EPS, rope)


def test_bounded_spill_chunked_summary_uses_route_for_rc_and_raw_v_for_vc():
    source = _source()
    chunked = source[
        source.index("def route_summary_bounded_spill_diagnostic"):
        source.index("def route_centroid_sol_reduction_config_diagnostic")
    ]
    assert "for first_block in range(0, blocks, chunk_blocks):" in chunked
    assert "materialized_sol_reduction_v4_route_diagnostic(" in chunked
    assert "route_desc," in chunked
    assert "raw_desc," in chunked
    assert "_reduce_kv_kernel[grid](" in chunked
    assert '"max_route_chunk_bytes"' in chunked
    assert '"reducer_configs_by_n"' in chunked
    assert "torch.empty_like(v)" not in chunked


def test_bounded_spill_chunked_summary_rejects_unbounded_chunk_count():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.bfloat16)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"chunk_blocks must be in \[1,16\]"):
        route_summary_bounded_spill_diagnostic(
            v,
            weight,
            NORM_EPS,
            rope,
            chunk_blocks=17,
        )


def test_bounded_spill_chunked_summary_fails_closed_on_cpu():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.bfloat16)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        route_summary_bounded_spill_diagnostic(
            v,
            weight,
            NORM_EPS,
            rope,
            chunk_blocks=4,
        )


def test_sol_reduction_config_diagnostic_rejects_unsupported_launch_config():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.bfloat16)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        route_centroid_sol_reduction_config_diagnostic(
            v,
            weight,
            NORM_EPS,
            rope,
            num_warps=4,
            num_stages=2,
        )


def test_sol_reduction_k1_fails_closed_on_cpu():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.bfloat16)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        route_summary_sol_reduction(v, weight, NORM_EPS, rope)


def test_sol_reduction_k1_rejects_wrong_dtype_before_gpu_execution():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.float32)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="requires BF16"):
        route_summary_sol_reduction(v, weight, NORM_EPS, rope)
