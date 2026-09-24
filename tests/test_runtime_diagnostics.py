import torch

from sol_h3.runtime_diagnostics import tensor_receipt


def test_tensor_receipt_is_stable_and_layout_bound():
    x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    first = tensor_receipt(x)
    second = tensor_receipt(x.clone())
    assert first == second

    transposed = tensor_receipt(x.t())
    assert transposed["sample_sha256"] != first["sample_sha256"]
    assert transposed["stride"] != first["stride"]


def test_tensor_receipt_detects_sampled_value_change():
    x = torch.arange(64, dtype=torch.float32)
    before = tensor_receipt(x)
    x[0] += 1
    after = tensor_receipt(x)
    assert before["sample_sha256"] != after["sample_sha256"]


def test_tensor_receipt_handles_empty_tensor():
    got = tensor_receipt(torch.empty(0, 3))
    assert got["numel"] == 0
    assert got["sample_count"] == 0
    assert got["sample_finite"] is True
