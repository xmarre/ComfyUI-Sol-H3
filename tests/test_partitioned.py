import math

import pytest
import torch

from sol_h3.partitioned import merge_lse_partitions, sm120_attention_with_lse
from sol_h3.partitioned_request import (
    _force_dense_partitioned_suffix_diagnostic_enabled,
    _weighted_dense,
)



def test_force_dense_partitioned_suffix_diagnostic_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SOL_H3_FORCE_DENSE_PARTITIONED_SUFFIX_DIAGNOSTIC", raising=False)
    assert _force_dense_partitioned_suffix_diagnostic_enabled() is False

    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("SOL_H3_FORCE_DENSE_PARTITIONED_SUFFIX_DIAGNOSTIC", value)
        assert _force_dense_partitioned_suffix_diagnostic_enabled() is True

    for value in ("", "0", "false", "disabled"):
        monkeypatch.setenv("SOL_H3_FORCE_DENSE_PARTITIONED_SUFFIX_DIAGNOSTIC", value)
        assert _force_dense_partitioned_suffix_diagnostic_enabled() is False

def test_lse_partition_merge_matches_explicit_dense_softmax():
    generator = torch.Generator(device="cpu").manual_seed(123)
    q = torch.randn((1, 2, 3, 8), generator=generator, dtype=torch.float64)
    keys = [
        torch.randn((1, 2, 5, 8), generator=generator, dtype=torch.float64),
        torch.randn((1, 2, 7, 8), generator=generator, dtype=torch.float64),
    ]
    values = [torch.randn(k.shape, generator=generator, dtype=torch.float64) for k in keys]
    measures = [math.log(0.4), 0.0]
    scale = q.shape[-1] ** -0.5

    partial_outputs = []
    partial_lses = []
    for key, value in zip(keys, values):
        scores = torch.matmul(q, key.transpose(-1, -2)) * scale
        partial_lses.append(torch.logsumexp(scores, dim=-1))
        partial_outputs.append(torch.matmul(torch.softmax(scores, dim=-1), value))
    merged, merged_lse = merge_lse_partitions(partial_outputs, partial_lses, measures)

    full_key = torch.cat(keys, dim=-2)
    full_value = torch.cat(values, dim=-2)
    bias = torch.cat(
        [
            torch.full((keys[0].shape[-2],), measures[0], dtype=torch.float64),
            torch.full((keys[1].shape[-2],), measures[1], dtype=torch.float64),
        ]
    )
    full_scores = torch.matmul(q, full_key.transpose(-1, -2)) * scale + bias.view(1, 1, 1, -1)
    full_lse = torch.logsumexp(full_scores, dim=-1)
    full_output = torch.matmul(torch.softmax(full_scores, dim=-1), full_value)

    assert torch.allclose(merged, full_output, rtol=1e-12, atol=1e-12)
    assert torch.allclose(merged_lse, full_lse, rtol=1e-12, atol=1e-12)


def test_single_union_weighted_dense_matches_partition_lse_oracle():
    """Production union transport and the design LSE oracle are dense-equivalent."""
    generator = torch.Generator(device="cpu").manual_seed(7341)
    q = torch.randn((4, 3, 8), generator=generator, dtype=torch.float64)
    prefix_k = torch.randn((6, 3, 8), generator=generator, dtype=torch.float64)
    prefix_v = torch.randn(prefix_k.shape, generator=generator, dtype=torch.float64)
    suffix_k = torch.randn((5, 3, 8), generator=generator, dtype=torch.float64)
    suffix_v = torch.randn(suffix_k.shape, generator=generator, dtype=torch.float64)
    prefix_measure = math.log(5.0 / 12.0)
    scale = q.shape[-1] ** -0.5

    q_bhtd = q.transpose(0, 1).unsqueeze(0)
    partial_outputs = []
    partial_lses = []
    for key, value in ((prefix_k, prefix_v), (suffix_k, suffix_v)):
        key_bhtd = key.transpose(0, 1).unsqueeze(0)
        value_bhtd = value.transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(q_bhtd, key_bhtd.transpose(-1, -2)) * scale
        partial_lses.append(torch.logsumexp(scores, dim=-1))
        partial_outputs.append(torch.matmul(torch.softmax(scores, dim=-1), value_bhtd))
    oracle, _ = merge_lse_partitions(partial_outputs, partial_lses, [prefix_measure, 0.0])

    union_k = torch.cat((prefix_k, suffix_k), dim=0)
    union_v = torch.cat((prefix_v, suffix_v), dim=0)
    union_bias = torch.cat(
        (
            torch.full((prefix_k.shape[0],), prefix_measure, dtype=torch.float64),
            torch.zeros(suffix_k.shape[0], dtype=torch.float64),
        )
    )
    union = _weighted_dense(q, union_k, union_v, union_bias, scale=scale)

    assert torch.allclose(union.transpose(0, 1).unsqueeze(0), oracle, rtol=1e-12, atol=1e-12)


def test_lse_partition_merge_accumulates_bfloat16_in_float32():
    outputs = [
        torch.tensor([[[[1.0, 3.0]]]], dtype=torch.bfloat16),
        torch.tensor([[[[5.0, 7.0]]]], dtype=torch.bfloat16),
    ]
    lses = [torch.tensor([[[0.0]]], dtype=torch.float32), torch.tensor([[[0.0]]], dtype=torch.float32)]
    merged, merged_lse = merge_lse_partitions(outputs, lses, [0.0, 0.0])
    assert merged.dtype == torch.bfloat16
    assert merged_lse.dtype == torch.float32
    assert torch.equal(merged, torch.tensor([[[[3.0, 5.0]]]], dtype=torch.bfloat16))
    assert merged_lse.item() == pytest.approx(math.log(2.0), abs=1e-7)


def test_sm120_partition_primitive_fails_closed_on_cpu():
    q = torch.zeros((1, 4, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same CUDA device"):
        sm120_attention_with_lse(q, q, q)
