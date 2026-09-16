from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sol_h3.mapped_neighbors import (
    MappingUnavailable,
    compile_descriptor,
    device_descriptor,
    validate_wire_map,
)


def _descriptor():
    wire = (
        "vdn_query_positions",
        1,
        "vdn-cpu-descriptor-test",
        "a" * 64,
        0,
        64,
        256,
        1,
        ((0, 64, 65),),
    )
    validated = validate_wire_map(wire, q_rows=64, kv_rows=256, sink_rows=1)
    descriptor = compile_descriptor(validated)
    assert descriptor is not None
    return descriptor


def _state():
    return SimpleNamespace(
        mapped_descriptor_cache=OrderedDict(),
        mapped_descriptor_bytes=0,
    )


def test_cpu_device_descriptor_cache_never_touches_cuda(monkeypatch):
    descriptor = _descriptor()
    state = _state()

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("CPU mapped descriptor touched a CUDA API")

    monkeypatch.setattr(torch.cuda, "current_stream", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "Event", forbidden_cuda)

    first = device_descriptor(state, descriptor, torch.device("cpu"))
    second = device_descriptor(state, descriptor, torch.device("cpu"))

    assert first is second
    assert first.dtype == torch.int32
    assert first.device.type == "cpu"
    assert first.tolist() == [[0, 4]]
    assert state.mapped_descriptor_bytes == first.numel() * first.element_size()
    assert len(state.mapped_descriptor_cache) == 1


def test_cuda_capture_miss_checks_descriptor_device_before_allocation(monkeypatch):
    descriptor = _descriptor()
    state = _state()
    active_devices = []

    @contextmanager
    def target_device(device):
        device = torch.device(device)
        active_devices.append(device)
        try:
            yield
        finally:
            assert active_devices.pop() == device

    monkeypatch.setattr(torch.cuda, "device", target_device)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())

    def capture_status():
        assert active_devices == [torch.device("cuda:1")]
        return True

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capture_status)

    def forbidden_tensor(*args, **kwargs):
        raise AssertionError("capture miss allocated mapped metadata")

    monkeypatch.setattr(torch, "tensor", forbidden_tensor)

    with pytest.raises(MappingUnavailable, match="backend") as exc_info:
        device_descriptor(state, descriptor, torch.device("cuda:1"))

    assert exc_info.value.reason == "backend"
    assert not state.mapped_descriptor_cache
    assert state.mapped_descriptor_bytes == 0
