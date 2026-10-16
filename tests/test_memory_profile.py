"""Allocator-report bookkeeping; CUDA measurements require real hardware."""
import json

import pytest
import torch

from sol_h3.contracts import Config
from sol_h3 import memory_profile as profile


def test_storage_accounting_deduplicates_views_but_counts_their_whole_backing_allocation():
    packed = torch.empty(193, 3 * 2 * 128, dtype=torch.bfloat16)
    q, k, v = packed.split(256, -1)
    assert profile.storage_bytes(q[:1], k[:1], v[:1]) == packed.nbytes
    assert profile.storage_bytes(q, k, v.clone()) == packed.nbytes + v.nbytes


def test_disabled_profile_does_not_initialize_or_query_cuda(monkeypatch):
    monkeypatch.delenv("SOL_H3_MEMORY_PROFILE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("disabled diagnostics queried CUDA"))
    with profile.sampling(Config()):
        profile.mark("unused")
        assert not profile.enabled()


def test_opt_in_profile_is_bounded_nested_and_cleared_on_exception(monkeypatch, caplog):
    monkeypatch.setenv("SOL_H3_MEMORY_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 400)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 300)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 600)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    resets = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: resets.append(1))
    with caplog.at_level("INFO", logger="comfy.sol_h3"):
        with pytest.raises(ValueError, match="cancelled"):
            with profile.sampling(Config()):
                outer = profile._PROFILE.get()
                with profile.sampling(Config()):
                    assert profile._PROFILE.get() is outer
                for evaluation in range(300):
                    profile.mark("evaluation_enter", evaluation=evaluation)
                raise ValueError("cancelled")
    assert not profile.enabled()
    assert resets == [1]
    record = json.loads(caplog.records[-1].message.removeprefix("Sol-H3 memory "))
    assert record["success"] is False
    assert record["incremental_peak_allocated"] == 200
    assert record["peak_reserved"] == 600
    assert len(record["evaluations"]) == 256
