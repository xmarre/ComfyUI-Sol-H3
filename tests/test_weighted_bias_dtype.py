import torch

from sol_h3 import sparse


def test_dense_reference_casts_fp32_weighted_bias_to_query_dtype(monkeypatch):
    seen = {}

    def fake_sdpa(q, k, v, *, attn_mask=None, **kwargs):
        seen["query_dtype"] = q.dtype
        seen["bias_dtype"] = None if attn_mask is None else attn_mask.dtype
        assert attn_mask is not None
        assert attn_mask.dtype == q.dtype
        assert attn_mask.shape == (1, 1, 1, k.shape[2])
        return torch.zeros_like(q)

    monkeypatch.setattr(sparse.F, "scaled_dot_product_attention", fake_sdpa)
    q = torch.zeros(1, 2, 3, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    key_bias = torch.linspace(-1.0, 0.0, 5, dtype=torch.float32)

    out = sparse._dense_reference(q, k, v, None, key_bias=key_bias)

    assert seen == {"query_dtype": torch.bfloat16, "bias_dtype": torch.bfloat16}
    assert out.shape == (1, 3, 2, 128)
    assert key_bias.dtype == torch.float32
