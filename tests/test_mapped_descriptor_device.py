from collections import OrderedDict
from types import SimpleNamespace

import torch

from sol_h3.mapped_neighbors import (
    compile_descriptor,
    device_descriptor,
    validate_wire_map,
)


def test_cpu_device_descriptor_cache_never_touches_cuda(monkeypatch):
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
    state = SimpleNamespace(
        mapped_descriptor_cache=OrderedDict(),
        mapped_descriptor_bytes=0,
    )

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
