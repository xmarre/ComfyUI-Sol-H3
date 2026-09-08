"""Reject changes to the exact arithmetic being replaced, allowing unrelated upstream edits."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


def function_digest(function):
    source = textwrap.dedent(inspect.getsource(function))
    canonical = ast.unparse(ast.parse(source))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_native():
    from comfy.ldm.minimax import model as native
    expected = json.loads(Path(__file__).with_name("native_manifest.json").read_text())
    for path, digest in expected.items():
        function = native
        for name in path.split("."):
            function = getattr(function, name)
        try:
            actual = function_digest(function)
        except (TypeError, OSError) as exc:
            raise RuntimeError(f"Cannot verify native {path}; exact fusion requires an audited source implementation") from exc
        if actual != digest:
            raise RuntimeError(f"Native {path} changed since the Sol-H3 audit; review compatibility before exact fusion")
