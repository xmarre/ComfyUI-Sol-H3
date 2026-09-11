"""Real-SM120 numerical acceptance for weighted Sol-Attn.

These tests deliberately execute the packaged CuTe kernel. CPU CI only verifies
that they collect; release acceptance requires running them on an SM120 GPU.
"""

import math

import pytest
import torch

from sol_h3 import sparse


_SCALE = 128 ** -0.5
_BLOCK = 64


def _sm120_kernel():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires real SM120")
    kernel = sparse.load_kernel(torch.device("cuda"))
    assert kernel.backend_name == "cute_sm120"
    return kernel


def _weighted_dense_reference(q, k, v, key_bias):
    """Independent dense natural-log key-measure reference in FP32."""
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * _SCALE
    logits = logits + key_bias.view(1, 1, 1, -1)
    probs = logits.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v.float()).bfloat16()


def _forced_sparse_reference(q, k, v, key_bias, sink_start, sink_tokens):
    """Independent tau=inf compressed-softmax oracle.

    Sol-Attn keeps ordinal-neighbour and sink-overlapping blocks exact. Other
    blocks use the BF16 K centroid plus BF16 V sum, represented as one key with
    log(block_rows) mass. This mirrors the mathematical sparse route without
    calling the production kernel or its preprocessing implementation.
    """
    sink_begin = sink_start // _BLOCK
    sink_end = (sink_start + sink_tokens + _BLOCK - 1) // _BLOCK
    outputs = []

    for q_start in range(0, q.shape[1], _BLOCK):
        q_block = q_start // _BLOCK
        keys = []
        values = []
        log_masses = []
        biases = []

        for k_start in range(0, k.shape[1], _BLOCK):
            k_block = k_start // _BLOCK
            kb = k[:, k_start:k_start + _BLOCK]
            vb = v[:, k_start:k_start + _BLOCK]
            rows = kb.shape[1]
            exact = abs(q_block - k_block) <= 1 or sink_begin <= k_block < sink_end

            if exact:
                keys.append(kb.float())
                values.append(vb.float())
                log_masses.extend([0.0] * rows)
                biases.append(key_bias[k_start:k_start + rows].float())
            else:
                keys.append(kb.float().mean(1, keepdim=True).bfloat16().float())
                values.append(
                    vb.float().sum(1, keepdim=True).bfloat16().float() / float(rows)
                )
                log_masses.append(math.log(float(rows)))
                # The test places non-unit key measure only in the exact sink.
                # Approximate blocks therefore carry unit measure.
                biases.append(torch.zeros(1, device=q.device, dtype=torch.float32))

        keys_cat = torch.cat(keys, dim=1)
        values_cat = torch.cat(values, dim=1)
        logits = torch.einsum(
            "bqhd,bkhd->bhqk", q[:, q_start:q_start + _BLOCK].float(), keys_cat
        ) * _SCALE
        logits = logits + torch.tensor(
            log_masses, device=q.device, dtype=torch.float32
        ).view(1, 1, 1, -1)
        logits = logits + torch.cat(biases).view(1, 1, 1, -1)
        outputs.append(
            torch.einsum("bhqk,bkhd->bqhd", logits.softmax(dim=-1), values_cat)
        )

    return torch.cat(outputs, dim=1).bfloat16()


@pytest.mark.gpu
def test_real_sm120_weighted_all_selected_matches_dense_and_changes_output():
    """Compile/execute the weighted specialization and verify log-bias units."""
    kernel = _sm120_kernel()
    torch.manual_seed(211)
    q = torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 193, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    key_bias = torch.zeros(193, device="cuda", dtype=torch.float32)
    key_bias[:64] = math.log(0.25)
    key_bias[64:128] = math.log(2.0)
    key_bias[128:] = math.log(0.6)

    with torch.inference_mode():
        got = kernel(
            q,
            k,
            v,
            tau=float("inf"),
            sink_start=0,
            sink_tokens=k.shape[1],
            key_bias=key_bias,
        )
        want = _weighted_dense_reference(q, k, v, key_bias)
        metrics = sparse.error_metrics(got, want)
        assert sparse.arithmetic_gate_passes(metrics), metrics

        plain = kernel(
            q,
            k,
            v,
            tau=float("inf"),
            sink_start=0,
            sink_tokens=k.shape[1],
        )
        assert not torch.equal(got, plain)


@pytest.mark.gpu
def test_real_sm120_weighted_exact_sink_preserves_sparse_tail():
    """Non-unit key measure stays exact while unrelated far blocks stay sparse."""
    kernel = _sm120_kernel()
    torch.manual_seed(223)
    q = torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 449, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    sink_start = 64
    sink_tokens = 128
    key_bias = torch.zeros(449, device="cuda", dtype=torch.float32)
    key_bias[64:128] = math.log(0.4)
    key_bias[128:192] = math.log(1.8)

    with torch.inference_mode():
        got = kernel(
            q,
            k,
            v,
            tau=float("inf"),
            sink_start=sink_start,
            sink_tokens=sink_tokens,
            key_bias=key_bias,
        )
        want = _forced_sparse_reference(
            q, k, v, key_bias, sink_start=sink_start, sink_tokens=sink_tokens
        )
        metrics = sparse.error_metrics(got, want)
        assert sparse.arithmetic_gate_passes(metrics), metrics
