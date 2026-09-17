from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_comfy_custom_node_load_exposes_canonical_sol_h3(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = r'''
import importlib
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location(
    "comfy_custom_sol_h3",
    root / "__init__.py",
    submodule_search_locations=[str(root)],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

history = importlib.import_module("sol_h3.partitioned_history")
request = importlib.import_module("sol_h3.partitioned_request")
assert callable(history.install_partitioned_history_bridge)
assert isinstance(request.PARTITIONED_REQUEST_ABI, str)
assert request.PARTITIONED_REQUEST_ABI
'''
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(root)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
