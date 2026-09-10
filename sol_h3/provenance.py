"""Verify packaged source before importing the optional GPU runtime."""
import hashlib
import json
from pathlib import Path

SOURCE = 'sana-sol-engine'
REVISION = '2936c47637380842aaa4a4488fac5006cc542b70'
CONTRACT = 'sana-sol-engine-sol-attn-64-rect-sm120-v3'


def _manifest_name(path, source):
    """Return a platform-independent manifest key for a source-tree path."""
    return path.relative_to(source).as_posix()


def _matches_packaged_hash(path, expected):
    """Match exact bytes, or the same text after Git-style CRLF normalization.

    The manifest records the canonical LF bytes committed by this project. Git
    for Windows can materialize tracked text as CRLF depending on the user's
    checkout configuration. Treat only CRLF -> LF as transport normalization;
    every other byte difference remains a hard provenance failure.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    if b'\r\n' not in data:
        return False
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == expected


def verify_source():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'sol_manifest.json').read_text(encoding='utf-8'))
    if manifest['revision'] != REVISION or manifest['source'] != SOURCE:
        raise RuntimeError('Packaged Sana source identity mismatch')
    source = root / '_vendor' / 'sol_attn'
    actual = {_manifest_name(p, source) for p in source.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    expected = set(manifest['files'])
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            'Packaged Sana source file set mismatch: '
            f'missing={missing or []}, unexpected={unexpected or []}'
        )
    for name, hashes in manifest['files'].items():
        if not _matches_packaged_hash(source / name, hashes['packaged_sha256']):
            raise RuntimeError(f'Packaged Sana source hash mismatch: {name}')
    return manifest
