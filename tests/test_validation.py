import threading

import torch

from sol_h3.contracts import Config
from sol_h3.runtime import Request
from sol_h3 import validation


def _key(state, *, rows=8, bias=None, physical=None):
    q = torch.zeros((1, 1, rows, 128), dtype=torch.bfloat16)
    state.runtime_lease.source_verified = True
    state.runtime_lease.source_generation = "source"
    state.runtime_lease.implementation_generation = "impl"
    state.runtime_lease.ordinary_runtime_identity = ("test-runtime",)
    state.runtime_lease.ordinary_compiler_namespace = ("test-compiler",)
    state.runtime_lease.device_identity = {"type": "cpu", "index": None, "process": 1}
    return validation.build_arithmetic_key(
        state=state,
        q=q,
        k=q,
        v=q,
        mode="ordinary_unweighted_v1",
        layout="bhtd",
        scale=128 ** -0.5,
        bias_identity=bias,
        physical_identity=physical,
    )


def test_success_is_reused_only_within_one_request():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state)
    first = state.validation_state.begin(key)
    assert first.validate is True
    state.validation_state.publish_success(first, {"finite": True, "mean_abs": 0.0})
    second = state.validation_state.begin(key)
    assert second.validate is False
    assert state.validation_state.summary()["hits"] == 1

    other = Request(Config(exact=False, backend="sol"))
    other_key = _key(other)
    assert other.validation_state.begin(other_key).validate is True


def test_invalidation_forces_a_new_generation():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state)
    ticket = state.validation_state.begin(key)
    state.validation_state.publish_success(ticket, {"finite": True})
    state.validation_state.invalidate()
    changed = _key(state)
    ticket2 = state.validation_state.begin(changed)
    assert ticket2.validate is True
    assert state.validation_state.summary()["miss_reasons"]["numerical_transition"] == 1


def test_lru_is_bounded_by_entry_count():
    service = validation.ArithmeticValidationState(max_entries=2, max_bytes=65536)
    state = Request(Config(exact=False, backend="sol"))
    state.validation_state = service
    for rows in (8, 9, 10):
        key = _key(state, rows=rows)
        ticket = service.begin(key)
        service.publish_success(ticket, {"finite": True})
    summary = service.summary()
    assert summary["cache_entries"] == 2
    assert summary["evictions"] == 1


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


def test_runtime_lease_verifies_source_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validation,
        "_device_identity",
        lambda device: {"type": "cpu", "index": None, "process": 1},
    )
    monkeypatch.setattr(
        validation,
        "_distribution_version",
        lambda name: f"version:{name}",
    )
    import sol_h3.provenance as provenance
    monkeypatch.setattr(provenance, "verify_source", lambda: calls.append(True) or {
        "source": provenance.SOURCE,
        "revision": provenance.REVISION,
        "contract": provenance.CONTRACT,
        "files": {},
    })
    lease = validation.RuntimeLease()
    lease.ensure_source_verified(torch.device("cpu"))
    lease.ensure_source_verified(torch.device("cpu"))
    assert calls == [True]
    assert lease.source_verify_count == 1
    assert lease.compiler_environment["triton"] == "version:triton"
    assert lease.compiler_environment["nvidia_cutlass_dsl"] == "version:nvidia-cutlass-dsl"
    assert lease.compiler_environment["cuda_python"] == "version:cuda-python"
    assert lease.compiler_environment["apache_tvm_ffi"] == "version:apache-tvm-ffi"


def test_partitioned_binding_does_not_change_ordinary_namespace(monkeypatch):
    state = Request(Config(exact=False, backend="sol"))
    lease = state.runtime_lease
    lease.source_verified = True
    lease.source_generation = "source"
    lease.implementation_generation = "impl"
    lease.device_identity = {"type": "cpu", "index": None, "process": 1}

    class Kernel:
        compiler_namespace = ("ordinary",)
        backend_name = "cute_sm120"
        block_size = 64

    monkeypatch.setattr(
        validation,
        "_device_identity",
        lambda device: {"type": "cpu", "index": None, "process": 1},
    )
    lease.bind_kernel(Kernel(), torch.device("cpu"))
    ordinary = lease.compiler_namespace_for("ordinary_unweighted_v1")
    assert ordinary == ("ordinary",)


def test_ordinary_compiler_cache_generation_reports_transition(monkeypatch):
    lease = validation.RuntimeLease()
    monkeypatch.setattr(
        validation,
        "_device_identity",
        lambda device: {"type": "cpu", "index": None, "process": 1},
    )
    lease.source_verified = True
    lease.source_generation = "source"
    lease.implementation_generation = "impl"
    lease.device_identity = {"type": "cpu", "index": None, "process": 1}

    class Kernel:
        backend_name = "cute_sm120"
        block_size = 64

        def __init__(self):
            self.generation = 0
            self.compiler_namespace_provider = (
                lambda: ("compiler", self.generation)
            )

    kernel = Kernel()
    assert lease.bind_kernel(kernel, torch.device("cpu")) is False
    kernel.generation += 1
    assert lease.bind_kernel(kernel, torch.device("cpu")) is True


def test_loaded_ordinary_runtime_replacement_reports_transition(monkeypatch):
    lease = validation.RuntimeLease()
    monkeypatch.setattr(
        validation,
        "_device_identity",
        lambda device: {"type": "cpu", "index": None, "process": 1},
    )
    lease.source_verified = True
    lease.source_generation = "source"
    lease.implementation_generation = "impl"
    lease.device_identity = {"type": "cpu", "index": None, "process": 1}

    class Kernel:
        backend_name = "cute_sm120"
        block_size = 64

        def __init__(self, generation):
            self.compiler_namespace = ("compiler", generation)

    assert lease.bind_kernel(Kernel(1), torch.device("cpu")) is False
    assert lease.bind_kernel(Kernel(2), torch.device("cpu")) is True


def test_failed_owner_wakes_waiter_and_requires_fresh_validation():
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
    assert result and result[0].validate is True
    assert result[0].publish is True
    state.validation_state.publish_failure(result[0])


def test_generation_change_during_validation_rejects_stale_publish_and_wakes_waiter():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state)
    first = state.validation_state.begin(key)
    result = []

    def waiter():
        result.append(state.validation_state.begin(key))

    thread = threading.Thread(target=waiter)
    thread.start()
    state.validation_state.invalidate("numerical_transition")
    state.validation_state.publish_success(first, {"finite": True})
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result and result[0].validate is True
    assert result[0].generation == 1
    assert state.validation_state.stats["generation_publish_rejected"] == 1
    state.validation_state.publish_failure(result[0])


def test_oversized_identity_bypasses_success_storage():
    service = validation.ArithmeticValidationState(max_entries=4, max_bytes=64)
    state = Request(Config(exact=False, backend="sol"))
    state.validation_state = service
    key = _key(state)
    ticket = service.begin(key)

    assert ticket.validate is True
    assert ticket.publish is False
    service.publish_success(ticket, {"finite": True})
    summary = service.summary()
    assert summary["cache_entries"] == 0
    assert summary["miss_reasons"]["unrepresentable_identity"] == 1


def test_success_entries_retain_no_tensors():
    state = Request(Config(exact=False, backend="sol"))
    key = _key(state, bias={"segments": [{"start": 0, "stop": 8, "value": "0x0.0p+0"}]})
    ticket = state.validation_state.begin(key)
    state.validation_state.publish_success(ticket, {"finite": True, "mean_abs": 0.0})

    def contains_tensor(value):
        if torch.is_tensor(value):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(k) or contains_tensor(v) for k, v in value.items())
        if isinstance(value, (tuple, list, set)):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(state.validation_state._entries)


def test_bias_scale_and_physical_identity_mutations_do_not_hit_prior_success():
    state = Request(Config(exact=False, backend="sol"))
    service = state.validation_state
    base_key = _key(state, bias={"value": "a"}, physical={"map": "a"})
    ticket = service.begin(base_key)
    service.publish_success(ticket, {"finite": True})

    bias_key = _key(state, bias={"value": "b"}, physical={"map": "a"})
    assert service.begin(bias_key).validate is True

    # Use a fresh service for each mutation so the miss classification is not
    # affected by an intentionally left in-flight validator above.
    state.validation_state = validation.ArithmeticValidationState()
    service = state.validation_state
    ticket = service.begin(base_key)
    service.publish_success(ticket, {"finite": True})
    scale_key = dict(base_key)
    scale_key["scale"] = float(0.123).hex()
    assert service.begin(scale_key).validate is True

    state.validation_state = validation.ArithmeticValidationState()
    service = state.validation_state
    ticket = service.begin(base_key)
    service.publish_success(ticket, {"finite": True})
    physical_key = dict(base_key)
    physical_key["physical_identity"] = {"map": "b"}
    assert service.begin(physical_key).validate is True


def test_ordinary_and_partitioned_bindings_share_one_source_verification(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validation,
        "_device_identity",
        lambda device: {"type": "cpu", "index": None, "process": 1},
    )
    import sol_h3.provenance as provenance
    from sol_h3 import compiler_attribution

    monkeypatch.setattr(
        provenance,
        "verify_source",
        lambda: calls.append(True)
        or {
            "source": provenance.SOURCE,
            "revision": provenance.REVISION,
            "contract": provenance.CONTRACT,
            "files": {},
        },
    )
    monkeypatch.setattr(
        compiler_attribution,
        "install_hooks",
        lambda interface: ("partitioned-compiler",),
    )
    monkeypatch.setattr(
        compiler_attribution,
        "compiler_namespace",
        lambda interface: ("partitioned-compiler",),
    )

    class Kernel:
        backend_name = "cute_sm120"
        block_size = 64
        compiler_namespace = ("ordinary-compiler",)

    class Interface:
        __file__ = "fake-interface.py"
        MAPPED_NEIGHBOR_CONTRACT = "mapped-v4"

        @staticmethod
        def _compile_sm120():
            raise AssertionError("compile is not part of lease binding")

    lease = validation.RuntimeLease()
    assert lease.bind_kernel(Kernel(), torch.device("cpu")) is False
    assert lease.bind_partitioned(Interface(), torch.device("cpu")) is False
    assert calls == [True]
    assert lease.source_verify_count == 1



def test_device_identity_carries_stable_process_generation():
    from sol_h3.validation import _device_identity

    first = _device_identity("cpu")
    second = _device_identity("cpu")

    assert first["process"] == second["process"]
    assert first["process_generation"] == second["process_generation"]
    assert isinstance(first["process_generation"], str)
    assert len(first["process_generation"]) == 32
