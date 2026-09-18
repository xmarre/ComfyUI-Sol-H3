import threading
from types import SimpleNamespace

import pytest
import torch

from sol_h3.contracts import Config
from sol_h3.runtime import Request
from sol_h3 import validation


def _key(
    state,
    *,
    rows=8,
    bias=None,
    physical=None,
    scale=128 ** -0.5,
    mapped_abi=None,
    tensor=None,
    compiler_namespace=("test",),
):
    q = tensor if tensor is not None else torch.zeros((1, 1, rows, 128), dtype=torch.bfloat16)
    state.runtime_lease.source_verified = True
    state.runtime_lease.source_generation = "source"
    state.runtime_lease.implementation_generation = "impl"
    state.runtime_lease.device_identity = {"type": "cpu", "index": None, "process": 1}
    return validation.build_arithmetic_key(
        state=state,
        q=q,
        k=q,
        v=q,
        mode="ordinary_unweighted_v1",
        layout="bhtd",
        scale=scale,
        compiler_namespace=compiler_namespace,
        mapped_abi=mapped_abi,
        bias_identity=bias,
        physical_identity=physical,
    )


def _publish(service, key):
    ticket = service.begin(key)
    assert ticket.validate is True
    service.publish_success(ticket, {"finite": True, "mean_abs": 0.0})
    return ticket


def test_success_is_reused_only_within_one_request():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state)
    _publish(state.validation_state, key)
    second = state.validation_state.begin(key)
    assert second.validate is False
    assert state.validation_state.summary()["hits"] == 1

    other = Request(Config(exact=False, backend="sol"))
    other_key = _key(other)
    assert other.validation_state.begin(other_key).validate is True


def test_invalidation_forces_a_new_generation():
    state = Request(Config(exact=False, backend="sol"))
    _publish(state.validation_state, _key(state))
    state.validation_state.invalidate()
    changed = _key(state)
    ticket2 = state.validation_state.begin(changed)
    assert ticket2.validate is True
    assert state.validation_state.summary()["miss_reasons"]["numerical_transition"] == 1
    state.validation_state.publish_failure(ticket2)


def test_generation_change_rejects_stale_publication():
    state = Request(Config(exact=False, backend="sol"))
    first = state.validation_state.begin(_key(state))
    state.validation_state.invalidate()
    state.validation_state.publish_success(first, {"finite": True})
    summary = state.validation_state.summary()
    assert summary["cache_entries"] == 0
    assert state.validation_state.stats["generation_publish_rejected"] == 1

    current = state.validation_state.begin(_key(state))
    assert current.validate is True
    state.validation_state.publish_failure(current)


def test_lru_is_bounded_by_entry_count():
    service = validation.ArithmeticValidationState(max_entries=2, max_bytes=65536)
    state = Request(Config(exact=False, backend="sol"))
    state.validation_state = service
    for rows in (8, 9, 10):
        _publish(service, _key(state, rows=rows))
    summary = service.summary()
    assert summary["cache_entries"] == 2
    assert summary["evictions"] == 1


def test_lru_is_bounded_by_payload_bytes():
    service = validation.ArithmeticValidationState(max_entries=256, max_bytes=2200)
    state = Request(Config(exact=False, backend="sol"))
    state.validation_state = service
    for rows in range(8, 20):
        _publish(service, _key(state, rows=rows))
    summary = service.summary()
    assert summary["cache_bytes"] <= 2200
    assert summary["evictions"] > 0


def test_oversized_key_bypasses_reuse_without_publishing():
    service = validation.ArithmeticValidationState(max_entries=256, max_bytes=64)
    state = Request(Config(exact=False, backend="sol"))
    state.validation_state = service
    ticket = service.begin(_key(state))
    assert ticket.validate is True
    assert ticket.publish is False
    service.publish_success(ticket, {"finite": True})
    summary = service.summary()
    assert summary["cache_entries"] == 0
    assert summary["miss_reasons"]["unrepresentable_identity"] == 1


def test_success_cache_retains_no_tensor_objects():
    state = Request(Config(exact=False, backend="sol"))
    _publish(state.validation_state, _key(state))
    entry = next(iter(state.validation_state._entries.values()))

    def contains_tensor(value):
        if torch.is_tensor(value):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(k) or contains_tensor(v) for k, v in value.items())
        if isinstance(value, (tuple, list, set, frozenset)):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(entry)


def test_arithmetic_key_changes_for_bias_scale_mapped_abi_and_stride():
    state = Request(Config(exact=False, backend="sol"))
    base = _key(state)
    base_ticket = state.validation_state.begin(base)
    base_digest = base_ticket.digest
    state.validation_state.publish_failure(base_ticket)

    keys = [
        _key(state, bias="other"),
        _key(state, scale=0.25),
        _key(state, mapped_abi="mapped-v1"),
        _key(
            state,
            tensor=torch.zeros((1, 1, 16, 128), dtype=torch.bfloat16)[:, :, ::2, :],
        ),
    ]
    digests = {validation._digest(key) for key in keys}
    assert base_digest not in digests
    assert len(digests) == len(keys)


def test_same_thread_reentrant_key_never_self_waits():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state)
    first = state.validation_state.begin(key)
    second = state.validation_state.begin(key)
    assert first.validate and first.publish
    assert second.validate and not second.publish
    state.validation_state.publish_failure(first)


def test_waiter_reuses_successful_inflight_proof():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state)
    first = state.validation_state.begin(key)
    result = []

    def waiter():
        result.append(state.validation_state.begin(key))

    thread = threading.Thread(target=waiter)
    thread.start()
    state.validation_state.publish_success(first, {"finite": True})
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result and result[0].validate is False


def test_waiter_retries_after_owner_failure():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state)
    first = state.validation_state.begin(key)
    result = []

    def waiter():
        result.append(state.validation_state.begin(key))

    thread = threading.Thread(target=waiter)
    thread.start()
    state.validation_state.publish_failure(first)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result and result[0].validate is True and result[0].publish is True
    state.validation_state.publish_failure(result[0])


def test_runtime_routes_bind_independently_without_identity_drift():
    lease = validation.RuntimeLease()
    lease.source_verified = True
    lease.source_generation = "source"
    lease.implementation_generation = "impl"
    lease.device_identity = validation._device_identity(torch.device("cpu"))

    def kernel():
        pass

    kernel.backend_name = "cute_sm120"
    kernel.block_size = 64

    def compile_fn():
        pass

    interface = SimpleNamespace(
        __file__="interface.py",
        _compile_sm120=compile_fn,
        _compiled={},
        MAPPED_NEIGHBOR_CONTRACT="mapped-v4",
    )

    ordinary = lease.bind_kernel(kernel, torch.device("cpu"))
    partitioned = lease.bind_partitioned(interface, torch.device("cpu"))
    assert lease.bind_kernel(kernel, torch.device("cpu")) == ordinary
    assert lease.bind_partitioned(interface, torch.device("cpu")) == partitioned
    assert ordinary != partitioned
    assert set(lease.runtime_identities) == {"ordinary", "partitioned"}


def test_runtime_route_replacement_fails_closed():
    lease = validation.RuntimeLease()
    lease.source_verified = True
    lease.source_generation = "source"
    lease.implementation_generation = "impl"
    lease.device_identity = validation._device_identity(torch.device("cpu"))

    def compile_fn():
        pass

    interface = SimpleNamespace(
        __file__="interface.py",
        _compile_sm120=compile_fn,
        _compiled={},
        MAPPED_NEIGHBOR_CONTRACT="mapped-v4",
    )
    lease.bind_partitioned(interface, torch.device("cpu"))

    def replacement_compile_fn():
        pass

    interface._compile_sm120 = replacement_compile_fn
    with pytest.raises(RuntimeError, match="partitioned runtime identity changed"):
        lease.bind_partitioned(interface, torch.device("cpu"))


def test_runtime_lease_verifies_source_once_across_routes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validation,
        "_device_identity",
        lambda device: {"type": "cpu", "index": None, "process": 1},
    )
    import sol_h3.provenance as provenance

    monkeypatch.setattr(provenance, "verify_source", lambda: calls.append(True) or {
        "source": provenance.SOURCE,
        "revision": provenance.REVISION,
        "contract": provenance.CONTRACT,
        "files": {},
    })

    def kernel():
        pass

    kernel.backend_name = "cute_sm120"
    kernel.block_size = 64

    def compile_fn():
        pass

    interface = SimpleNamespace(
        __file__="interface.py",
        _compile_sm120=compile_fn,
        _compiled={},
        MAPPED_NEIGHBOR_CONTRACT="mapped-v4",
    )
    lease = validation.RuntimeLease()
    lease.bind_kernel(kernel, torch.device("cpu"))
    lease.bind_partitioned(interface, torch.device("cpu"))
    lease.bind_kernel(kernel, torch.device("cpu"))
    assert calls == [True]
    assert lease.source_verify_count == 1
