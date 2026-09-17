import math

import pytest
import torch

from sol_h3.partitioned import merge_lse_partitions, sm120_attention_with_lse


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
