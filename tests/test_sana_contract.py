"""Exercise the real packaged interface; CUDA execution is deliberately separate."""
import ast
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from sol_h3._vendor.sol_attn import interface
from sol_h3.provenance import verify_source, REVISION
from tools.vendor_sol_attn import package_sources


def test_provenance_and_node_local_imports():
    manifest = verify_source()
    assert manifest['revision'] == REVISION
    patch = Path(__file__).resolve().parents[1] / 'tools/rectangular_sm120.patch'
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest['packaged_patch_sha256']
    root = Path(interface.__file__).parent
    assert len(manifest['files']) == 51
    raw_files = {}
    for name, hashes in manifest['files'].items():
        if name.endswith('.py'):
            tree = ast.parse((root / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith(('sol_attn', 'comfy_kitchen'))
                elif isinstance(node, ast.Import):
                    assert not any(n.name.startswith(('sol_attn', 'comfy_kitchen')) for n in node.names)
        source = os.environ.get('SANA_PATH')
        if source:
            raw = subprocess.check_output(['git', '-C', source, 'show',
                                          f"{REVISION}:{manifest['subtree']}/{name}"])
            assert hashlib.sha256(raw).hexdigest() == hashes['upstream_sha256']
            raw_files[name] = raw
    if raw_files:
        for name, data in package_sources(raw_files).items():
            assert data == (root / name).read_bytes()
    assert 'BSD 3-Clause' in (root / 'sm100/LICENSE.flash-attention').read_text()
    assert 'Apache License' in (root.parent / 'LICENSE.Apache-2.0').read_text()
    assert 'NVIDIA' in (root / 'THIRD_PARTY_NOTICES.md').read_text()


def test_public_api():
    assert tuple(inspect.signature(interface.sol_attn).parameters) == (
        'q', 'k', 'v', 'scale', 'tau', 'thresh_type', 'kv_splits',
        'sink_tokens', 'sink_start', 'compile_bucket_size')
    assert interface._backend_for_arch((12, 0), cute_available=True) == 'cute_sm120'
    assert interface._backend_for_arch((12, 0), cute_available=False) == 'triton'
    assert interface._backend_for_arch((10, 3), cute_available=True) == 'cute_sm100'


@pytest.mark.parametrize('prefix,end', [(0, 3), (1, 1), (64, 1), (65, 2), (130, 3)])
def test_actual_sink_block_contract(prefix, end):
    start = 3 if prefix == 0 else 0
    assert interface._sink_block_range(130, 0, prefix) == (start, end)
    if prefix == 65:
        assert interface._sink_block_range(130, None, prefix) == (1, 3)


def test_real_interface_validates_dtype_shape_and_device():
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='same CUDA device'):
        interface.sol_attn(q, q, q, sink_start=0, sink_tokens=1)
    with pytest.raises(TypeError, match='bfloat16'):
        interface.sol_attn(q.float(), q.float(), q.float())
    with pytest.raises(ValueError, match='head dimension'):
        interface.sol_attn(q[..., :64], q[..., :64], q[..., :64])


def test_bridge_calls_real_public_interface(monkeypatch):
    # Preserve the real public entrypoint and dispatch. Substitute only device
    # validation and the GPU executor because these tensors are on CPU.
    from sol_h3 import sparse
    from sol_h3.contracts import Config
    from sol_h3.runtime import Request
    monkeypatch.setattr(interface, '_validate_inputs', lambda *a, **k: (12, 0))
    monkeypatch.setattr(interface, '_cute_runtime_available', lambda: True)
    calls = []
    def execute(q, k, v, **kw):
        calls.append((q.shape, q.dtype, kw))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=kw['scale']).transpose(1, 2)
    monkeypatch.setattr(interface, '_sol_attn_cute', execute)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (12, 0))
    kernel = sparse.load_kernel(torch.device('cuda'))
    assert kernel.backend_name == 'cute_sm120'
    assert kernel.source_tree_verified
    monkeypatch.setattr(sparse, 'load_kernel', lambda device: kernel)
    q = torch.randn(1, 2, 65, 128, dtype=torch.bfloat16)
    state = Request(Config(exact=False, backend='sol'))
    out = sparse.attention(q, q, q, 1, state.config, state)
    assert out.shape == (1, 65, 256)
    assert out.dtype == torch.bfloat16
    assert [c[2]['sink_tokens'] for c in calls] == [65, 1]
    assert all(c[0] == (1, 65, 2, 128) and c[1] == torch.bfloat16 for c in calls)
    assert all(c[2]['scale'] == 128 ** -0.5 and c[2]['sink_start'] == 0 for c in calls)
    assert all(c[2]['thresh_type'] == 'diag' and c[2]['tau'] == 1.0 for c in calls)
    assert state.sparse_calls == 1


def test_import_without_global_checkout_or_pythonpath(tmp_path):
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-c',
                    f'import sys; sys.path.insert(0, {root!r}); '
                    'from sol_h3._vendor.sol_attn import sol_attn; '
                    'assert sol_attn.__module__ == "sol_h3._vendor.sol_attn.interface"; '
                    'assert "sol_attn" not in sys.modules; '
                    'assert "comfy_kitchen" not in sys.modules'],
                   cwd=tmp_path, env=env, check=True)


def test_tampered_source_cannot_claim_verified(tmp_path, monkeypatch):
    import shutil
    from sol_h3 import provenance
    source = Path(provenance.__file__).parent
    shutil.copytree(source / '_vendor', tmp_path / '_vendor')
    shutil.copyfile(source / 'sol_manifest.json', tmp_path / 'sol_manifest.json')
    monkeypatch.setattr(provenance, '__file__', str(tmp_path / 'provenance.py'))
    target = tmp_path / '_vendor/sol_attn/interface.py'
    target.write_text(target.read_text() + '\n# corrupt\n')
    with pytest.raises(RuntimeError, match='hash mismatch: interface.py'):
        provenance.verify_source()


def test_missing_cute_counts_no_sparse_and_caches_reason(monkeypatch):
    from sol_h3 import sparse
    from sol_h3.contracts import Config
    from sol_h3.runtime import Request
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (12, 0))
    monkeypatch.setattr(interface, '_cute_runtime_available', lambda: False)
    real_load = sparse.load_kernel
    attempts = []
    def load(device):
        attempts.append(device)
        return real_load(torch.device('cuda'))
    monkeypatch.setattr(sparse, 'load_kernel', load)
    q = torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16)
    state = Request(Config(exact=False, backend='sol'))
    for _ in range(2):
        with pytest.raises(sparse.KernelUnavailable, match='SM120 requires CuTe'):
            sparse.attention(q, q, q, 1, state.config, state)
    assert len(attempts) == 1
    assert state.sparse_calls == 0
    assert state.kernel is None
