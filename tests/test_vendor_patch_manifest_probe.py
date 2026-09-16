"""Temporary provenance probe; removed once the checked-in manifest is regenerated."""
import hashlib

from tools.vendor_sol_attn import PATCHES


def test_report_ordered_vendor_patch_hashes():
    records = [(label, hashlib.sha256(path.read_bytes()).hexdigest()) for label, path in PATCHES]
    raise AssertionError(f"ordered vendor patch hashes: {records}")
