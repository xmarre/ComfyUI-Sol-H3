from contextlib import nullcontext

import torch

from sol_h3._vendor.sol_attn import interface


def test_sm120_weighted_cache_separates_specialization_and_rebinds_bias_buffer(monkeypatch):
    from sol_h3._vendor.sol_attn import preprocess

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(interface, "_stream", lambda _: None)
    monkeypatch.setattr(interface, "_to_cute_tensors", lambda tensors: tensors)
    monkeypatch.setattr(interface, "_compiled", {})

    thresholds = []

    def prepare(q, k, v, **kwargs):
        blocks = (k.shape[1] + 63) // 64
        kc = torch.empty(1, blocks, q.shape[2], q.shape[3])
        vc = torch.empty_like(kc)
        threshold = torch.empty(1, (q.shape[1] + 63) // 64, q.shape[2])
        thresholds.append(threshold)
        return kc, vc, threshold

    monkeypatch.setattr(preprocess, "prepare", prepare)

    compiles = []
    launches = []

    def compile_(key, tensors, scale, start, end, stream, key_bias_enabled):
        compiles.append((key, key_bias_enabled, start, end))

        def compiled(*args, **kwargs):
            launches.append(args)
            args[3].copy_(args[0])

        interface._compiled[key] = compiled
        return compiled, tensors

    monkeypatch.setattr(interface, "_compile_sm120", compile_)

    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 202, 2, 128, dtype=q.dtype)
    v = torch.zeros_like(k)
    bias_a = torch.linspace(-1.0, 0.0, 202, dtype=torch.float32).contiguous()
    bias_b = torch.linspace(-2.0, 0.0, 202, dtype=torch.float32).contiguous()

    common = dict(
        arch=(12, 0),
        scale=128 ** -0.5,
        tau=1.0,
        thresh_type="diag",
        kv_splits=1,
        sink_tokens=192,
        sink_start=0,
        valid_tokens=65,
    )
    interface._sol_attn_cute(q, k, v, key_bias=bias_a, **common)
    interface._sol_attn_cute(q, k, v, key_bias=bias_b, **common)
    interface._sol_attn_cute(q, k, v, key_bias=None, **common)
    interface._sol_attn_cute(q, k, v, key_bias=None, **common)

    # Biased and unweighted kernels are different CuTe specializations, but
    # changing only values in an identically laid-out immutable bias vector
    # reuses the compiled weighted descriptor and passes the current buffer.
    assert [(enabled, start, end) for _, enabled, start, end in compiles] == [
        (True, 0, 3),
        (False, 0, 3),
    ]
    assert launches[0][7] is bias_a
    assert launches[1][7] is bias_b
    # No-bias specialization never needs a second O(T) allocation; its unused
    # tensor slot is the already-live threshold buffer for that invocation.
    assert launches[2][7] is thresholds[2]
    assert launches[3][7] is thresholds[3]
    assert all(args[3].shape == q.shape for args in launches)
