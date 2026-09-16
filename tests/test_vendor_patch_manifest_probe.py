"""Temporary provenance probe; removed once the checked-in manifest is regenerated."""
import hashlib
import os
from pathlib import Path
import subprocess

from sol_h3._vendor.sol_attn import interface
from tools.vendor_sol_attn import PATCHES, REVISION, SUBTREE, package_sources


def test_report_canonical_vendor_hashes():
    source = os.environ.get("SANA_PATH")
    if not source:
        return
    raw_files = {}
    for full in subprocess.check_output(
        ["git", "-C", source, "ls-tree", "-r", "--name-only", REVISION, "--", SUBTREE]
    ).decode().splitlines():
        name = full[len(SUBTREE) + 1:]
        raw_files[name] = subprocess.check_output(
            ["git", "-C", source, "show", f"{REVISION}:{full}"]
        )
    packaged = package_sources(raw_files)
    root = Path(interface.__file__).parent
    records = {
        name: {
            "canonical": hashlib.sha256(packaged[name]).hexdigest(),
            "checked_in": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            "equal": packaged[name] == (root / name).read_bytes(),
        }
        for name in ("interface.py", "sm120/kernel.py", "sm120/mainloop.py")
    }
    patches = [(label, hashlib.sha256(path.read_bytes()).hexdigest()) for label, path in PATCHES]
    raise AssertionError(f"canonical vendor hashes: files={records}; patches={patches}")
