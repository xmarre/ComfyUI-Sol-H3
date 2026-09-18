from sol_h3.diagnostics import CudaDiagnosticState


class _Event:
    clock = 0.0

    def __init__(self):
        self.value = None

    def record(self):
        type(self).clock += 1.0
        self.value = type(self).clock

    def elapsed_time(self, other):
        return (other.value - self.value) * 2.5


def test_disabled_diagnostics_have_no_samples():
    state = CudaDiagnosticState(enabled=False)
    assert state.begin_sample("gate", "cuda:0", important=True) is None
    state.resolve()
    assert state.summary()["selected_samples"] == 0


def test_bounded_diagnostics_resolve_events_and_report_drain_separately():
    syncs = []
    state = CudaDiagnosticState(
        enabled=True,
        max_samples=2,
        event_factory=_Event,
        synchronize=lambda device: syncs.append(str(device)),
    )
    first = state.begin_sample("gate", "cuda:0", context={"stage": "low"}, important=True)
    state.initial_stream_drain(first)
    with state.span(first, "all_selected"):
        pass
    second = state.begin_sample("production", "cuda:0")
    with state.span(second, "production_call"):
        pass
    assert state.begin_sample("gate", "cuda:0", important=True) is None

    state.resolve()
    summary = state.summary()
    assert summary["selected_samples"] == 2
    assert summary["resolved_samples"] == 2
    assert summary["overflow"]["gate"] == 1
    assert summary["details"][0]["context"] == {"stage": "low"}
    assert summary["details"][0]["initial_stream_drain_host_wall_s"] >= 0.0
    assert summary["details"][0]["cuda_event_ms"]["all_selected"] == 2.5
    assert summary["details"][1]["cuda_event_ms"]["production_call"] == 2.5
    assert syncs == ["cuda:0", "cuda:0"]
