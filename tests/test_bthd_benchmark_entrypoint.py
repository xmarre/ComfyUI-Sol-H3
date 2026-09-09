import runpy
import sys
from pathlib import Path


def test_bthd_layout_benchmark_bootstraps_repo_root_for_direct_execution(monkeypatch):
    """Direct script execution must make the checkout's sol_h3 package importable."""
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "bthd_layout_benchmark.py"
    tools = script.parent

    # Reproduce ``python tools/bthd_layout_benchmark.py``: Python starts with
    # tools/ at sys.path[0], not the repository root. Keep normal site paths so
    # torch remains importable, but remove the checkout root itself.
    filtered = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == root:
                continue
        except (OSError, RuntimeError):
            pass
        filtered.append(entry)
    monkeypatch.setattr(sys, "path", [str(tools), *filtered])

    runpy.run_path(str(script), run_name="__bthd_entrypoint_test__")

    assert Path(sys.path[0]).resolve() == root
