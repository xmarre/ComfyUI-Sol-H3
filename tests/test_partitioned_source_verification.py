"""Source trust follows sampling ownership, including failure and nested calls."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import threading

import pytest
import torch

from sol_h3 import partitioned_request as request, provenance
from sol_h3._vendor.sol_attn import interface, preprocess
from sol_h3.contracts import Config
from sol_h3.provenance import verify_source
from sol_h3.runtime import Request, SamplingWrapper, _REQUEST


@pytest.fixture
def host_kernel(monkeypatch):
    calls = {"verify": 0, "launch": 0}

    def verify():
        calls["verify"] += 1

    def prepare(q, k, v, **kwargs):
        blocks = (len(k[0]) + 63) // 64
        kc = torch.empty((1, blocks, q.shape[2], 128), dtype=q.dtype)
        return kc, torch.empty_like(kc), torch.empty((1, (len(q[0]) + 63) // 64, q.shape[2]))

    def compile_(key, tensors, *args):
        def launch(*arguments, **kwargs):
            calls["launch"] += 1
            arguments[3].copy_(arguments[0])

        interface._compiled[key] = launch
        return launch, tensors

    monkeypatch.setattr(provenance, "verify_source", verify)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(interface, "_validate_cute", lambda *args: None)
    monkeypatch.setattr(interface, "_stream", lambda _: None)
    monkeypatch.setattr(interface, "_compiled", {})
    monkeypatch.setattr(interface, "_compile_sm120", compile_)
    monkeypatch.setattr(interface, "_to_cute_tensors", lambda tensors: tensors)
    monkeypatch.setattr(preprocess, "prepare", prepare)
    return calls


def _kernel(q_rows=17, kv_rows=145):
    q = torch.arange(q_rows * 2 * 128, dtype=torch.float32).reshape(q_rows, 2, 128).bfloat16()
    k = torch.zeros((kv_rows, 2, 128), dtype=q.dtype)
    got = request._sm120_union(
        q,
        k,
        k,
        tau=1.0,
        scale=128**-0.5,
        sink_rows=kv_rows,
        key_bias=None,
        mapped_neighbor_intervals=None,
    )
    torch.testing.assert_close(got, q, rtol=0, atol=0)


def test_source_scan_once_per_request_across_layouts(host_kernel):
    states = [Request(Config()) for _ in range(2)]
    for state in states:
        token = _REQUEST.set(state)
        try:
            for _ in range(3):
                _kernel()
                _kernel(q_rows=33, kv_rows=193)
        finally:
            _REQUEST.reset(token)
    assert host_kernel == {"verify": 2, "launch": 12}
    assert all(state.partitioned_source_verification_calls == 1 for state in states)


def test_standalone_kernel_does_not_keep_global_source_trust(host_kernel):
    assert _REQUEST.get() is None
    _kernel()
    _kernel()
    assert host_kernel == {"verify": 2, "launch": 2}


def test_nested_requests_restore_the_outer_source_owner(host_kernel, caplog):
    outer, inner = SamplingWrapper(Config()), SamplingWrapper(Config())
    owners = []
    caplog.set_level("INFO", logger="comfy.sol_h3")

    def inside():
        owners.append(_REQUEST.get())
        _kernel()
        _kernel()

    def outside():
        owners.append(_REQUEST.get())
        _kernel()
        inner(inside)
        assert _REQUEST.get() is owners[0]
        _kernel()

    outer(outside)
    assert _REQUEST.get() is None
    assert owners[0] is not owners[1]
    assert host_kernel == {"verify": 2, "launch": 4}
    receipts = [json.loads(record.message.removeprefix("Sol-H3 ")) for record in caplog.records]
    assert len(receipts) == 2
    assert all(receipt["partitioned_source_tree_verified"] for receipt in receipts)
    assert all(receipt["partitioned_source_verification_calls"] == 1 for receipt in receipts)
    assert all(receipt["partitioned_source_verification_wall_s"] >= 0 for receipt in receipts)


def test_failed_source_scan_cannot_launch_or_leave_cached_trust(host_kernel, monkeypatch):
    state = Request(Config())
    token = _REQUEST.set(state)
    original = provenance.verify_source
    attempts = []

    def fail():
        attempts.append(True)
        raise RuntimeError("source mismatch")

    try:
        _kernel()
        monkeypatch.setattr(provenance, "verify_source", fail)
        with pytest.raises(RuntimeError, match="source mismatch"):
            _kernel()
        assert state._partitioned_source_identity is None
        assert host_kernel["launch"] == 1
        monkeypatch.setattr(provenance, "verify_source", original)
        _kernel()
        assert len(attempts) == 1
        assert host_kernel == {"verify": 2, "launch": 2}
        assert state.partitioned_source_verification_calls == 3
    finally:
        _REQUEST.reset(token)


def test_new_request_revalidates_and_failure_restores_lifecycle(host_kernel, monkeypatch, caplog):
    wrapper = SamplingWrapper(Config())
    wrapper(_kernel)

    def fail():
        raise RuntimeError("source mismatch")

    monkeypatch.setattr(provenance, "verify_source", fail)
    caplog.set_level("INFO", logger="comfy.sol_h3")
    with pytest.raises(RuntimeError, match="source mismatch"):
        wrapper(_kernel)
    assert _REQUEST.get() is None
    assert host_kernel["launch"] == 1
    receipt = json.loads(caplog.records[-1].message.removeprefix("Sol-H3 "))
    assert receipt["success"] is False
    assert receipt["partitioned_source_tree_verified"] is False
    assert receipt["partitioned_source_verification_calls"] == 1


def test_source_identity_change_revalidates_within_request(host_kernel, monkeypatch):
    state = Request(Config())
    token = _REQUEST.set(state)
    try:
        _kernel()
        monkeypatch.setattr(provenance, "CONTRACT", provenance.CONTRACT + "-changed")
        _kernel()
        _kernel()
        assert host_kernel == {"verify": 2, "launch": 3}
    finally:
        _REQUEST.reset(token)


def test_disk_tampering_is_detected_before_the_next_request_kernel(host_kernel, monkeypatch, tmp_path):
    snapshot = tmp_path / "sol_h3"
    shutil.copytree(Path(provenance.__file__).parent, snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(provenance, "__file__", str(snapshot / "provenance.py"))
    monkeypatch.setattr(provenance, "verify_source", verify_source)
    wrapper = SamplingWrapper(Config())
    wrapper(_kernel)
    changed = snapshot / "_vendor" / "sol_attn" / "sm120" / "kernel.py"
    changed.write_bytes(changed.read_bytes() + b"\n# source mismatch\n")
    with pytest.raises(RuntimeError, match="Packaged Sana source hash mismatch"):
        wrapper(_kernel)
    assert _REQUEST.get() is None
    assert host_kernel["launch"] == 1


def test_executor_failure_cannot_transfer_source_trust_to_the_next_request(host_kernel):
    wrapper = SamplingWrapper(Config())

    def fail():
        _kernel()
        raise RuntimeError("sampler failure")

    with pytest.raises(RuntimeError, match="sampler failure"):
        wrapper(fail)
    assert _REQUEST.get() is None
    wrapper(_kernel)
    assert host_kernel == {"verify": 2, "launch": 2}


def test_concurrent_calls_share_only_their_request_source_check(host_kernel, monkeypatch):
    state = Request(Config())
    entered, finish = threading.Event(), threading.Event()
    source_checks = []

    def verify():
        source_checks.append(True)
        entered.set()
        assert finish.wait(timeout=5)

    monkeypatch.setattr(provenance, "verify_source", verify)

    def invoke():
        token = _REQUEST.set(state)
        try:
            _kernel()
        finally:
            _REQUEST.reset(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke)
        assert entered.wait(timeout=5)
        second = pool.submit(invoke)
        finish.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert len(source_checks) == 1
    assert state.partitioned_source_verification_calls == 1
