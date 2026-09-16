"""Reproduce the node-local Sana snapshot from an explicitly pinned git checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

REVISION = '2936c47637380842aaa4a4488fac5006cc542b70'
SUBTREE = 'models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn'
ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4'
PATCHES = (
    ('rectangular-sm120-v3', ROOT / 'tools/rectangular_sm120.patch'),
    ('mapped-neighbor-sm120-v4', ROOT / 'tools/mapped_neighbor_sm120.patch'),
)


def relocate(data, path):
    if not path.endswith('.py'):
        return data
    text = data.decode()
    dots = '.' * len(Path(path).parts)
    text = re.sub(r'(?m)^(\s*)from sol_attn\.', rf'\1from {dots}', text)
    text = re.sub(
        r'(?m)^(\s*)import sol_attn\.(\S+)\.(\w+) as (\w+)$',
        lambda m: f'{m[1]}from {dots}{m[2]} import {m[3]} as {m[4]}', text)
    if text != data.decode():
        text = '# Modified by ComfyUI-Sol-H3: node-local relative imports only.\n' + text
    return text.encode()


def _package_sources(raw_files):
    """Apply the recorded functional patch stack after import relocation."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / 'sol_h3/_vendor/sol_attn'
        for name, raw in raw_files.items():
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(relocate(raw, name))

        modifications = {name: [] for name in raw_files}
        previous = {name: (target / name).read_bytes() for name in raw_files}
        for label, patch in PATCHES:
            subprocess.run(['git', 'apply', '--check', str(patch)], cwd=root, check=True)
            subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
            current = {name: (target / name).read_bytes() for name in raw_files}
            for name in raw_files:
                if current[name] != previous[name]:
                    modifications[name].append(label)
            previous = current
        return previous, modifications


def package_sources(raw_files):
    """Return canonical packaged bytes for compatibility with existing tests/tools."""
    packaged, _ = _package_sources(raw_files)
    return packaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()

    def git(*argv):
        return subprocess.check_output(['git', '-C', str(args.checkout), *argv])

    if git('rev-parse', 'HEAD').decode().strip() != REVISION:
        raise SystemExit('Checkout must be at the pinned Sana revision')
    target = ROOT / 'sol_h3' / '_vendor' / 'sol_attn'
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    raw_files = {}
    for full in git('ls-tree', '-r', '--name-only', REVISION, '--', SUBTREE).decode().splitlines():
        path = full[len(SUBTREE) + 1:]
        raw = git('show', f'{REVISION}:{full}')
        raw_files[path] = raw
    packaged_files, modifications = _package_sources(raw_files)
    for path, raw in raw_files.items():
        packaged = packaged_files[path]
        dest = target / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(packaged)
        records[path] = {
            'upstream_sha256': hashlib.sha256(raw).hexdigest(),
            'packaged_sha256': hashlib.sha256(packaged).hexdigest(),
        }
        if modifications[path]:
            records[path]['modifications'] = modifications[path]

    patch_records = [
        {
            'label': label,
            'path': patch.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
        }
        for label, patch in PATCHES
    ]
    manifest = {
        'source': 'sana-sol-engine',
        'repository': 'https://github.com/xmarre/Sana',
        'branch': 'sol-engine',
        'revision': REVISION,
        'subtree': SUBTREE,
        'contract': CONTRACT,
        'transformation': (
            'relative-imports-v1; rectangular-sm120-v3 (tools/rectangular_sm120.patch); '
            'mapped-neighbor-sm120-v4 (tools/mapped_neighbor_sm120.patch)'
        ),
        # Legacy audit field remains the rectangular-v3 patch hash.
        'packaged_patch_sha256': patch_records[0]['sha256'],
        'patches': patch_records,
        'files': records,
    }
    (ROOT / 'sol_h3' / 'sol_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
