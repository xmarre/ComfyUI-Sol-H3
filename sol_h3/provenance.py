"""Verify packaged source before importing the optional GPU runtime."""
import hashlib
import json
from pathlib import Path

SOURCE = 'sana-sol-engine'
REVISION = '2936c47637380842aaa4a4488fac5006cc542b70'
CONTRACT = 'sana-sol-engine-sol-attn-64-v1'


def verify_source():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'sol_manifest.json').read_text())
    if manifest['revision'] != REVISION or manifest['source'] != SOURCE:
        raise RuntimeError('Packaged Sana source identity mismatch')
    source = root / '_vendor' / 'sol_attn'
    actual = {str(p.relative_to(source)) for p in source.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != set(manifest['files']):
        raise RuntimeError('Packaged Sana source file set mismatch')
    for name, hashes in manifest['files'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != hashes['packaged_sha256']:
            raise RuntimeError(f'Packaged Sana source hash mismatch: {name}')
    return manifest
