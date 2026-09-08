import hashlib
from types import ModuleType
import sys

import pytest
import torch
import torch.nn.functional as F

from sol_h3 import sparse
from sol_h3.contracts import Config
from sol_h3.runtime import Request


def test_real_loader_rejects_cpu():
    with pytest.raises(RuntimeError, match="SM120"):
        sparse.load_kernel(torch.device("cpu"))


def test_pinned_source_checks(tmp_path):
    raw = b"example source\n"
    manifest = {
        "git_blobs": {"interface.py": "0" * 40},
        "sha256": {"interface.py": hashlib.sha256(raw).hexdigest()},
    }
    with pytest.raises(RuntimeError, match="missing"):
        sparse.verify_sources(tmp_path, manifest)
    (tmp_path / "interface.py").write_bytes(raw)
    sparse.verify_sources(tmp_path, manifest)
    (tmp_path / "interface.py").write_bytes(raw + b"changed")
    with pytest.raises(RuntimeError, match="mismatch"):
        sparse.verify_sources(tmp_path, manifest)


def test_pinned_manifest_requires_matching_security_file_set(tmp_path):
    (tmp_path / "interface.py").write_text("x")
    with pytest.raises(RuntimeError, match="file sets differ"):
        sparse.verify_sources(
            tmp_path,
            {"git_blobs": {"interface.py": "0" * 40}, "sha256": {}},
        )


def test_package_lookup_does_not_execute_init(monkeypatch, tmp_path):
    package = tmp_path / "sol_attn"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('package code executed')\n")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    assert sparse._find_package_root() == package.resolve()
    assert "sol_attn" not in sys.modules


def test_preimported_package_is_rejected_before_use(monkeypatch, tmp_path):
    package = tmp_path / "sol_attn"
    package.mkdir()
    raw = b"# pinned\n"
    (package / "__init__.py").write_bytes(raw)
    manifest = {
        "git_blobs": {"__init__.py": "0" * 40},
        "sha256": {"__init__.py": hashlib.sha256(raw).hexdigest()},
    }
    monkeypatch.setitem(sys.modules, "sol_attn", ModuleType("sol_attn"))
    sparse._VERIFIED_ROOTS.discard(package.resolve())
    with pytest.raises(RuntimeError, match="imported before"):
        sparse._load_verified_interface(package, manifest)


def test_bridge_protects_prefix_and_uses_full_sink_gate(monkeypatch):
    # CPU bridge oracle only: deliberately fake the sparse output after the
    # all-selected call to verify that ALL prefix queries are overwritten.
    calls = []
    dense_calls = []

    def kernel(q, k, v, **kw):
        calls.append(kw)
        dense = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return dense if kw["sink_tokens"] == q.shape[1] else torch.full_like(dense, 9)

    def dense_attention(q, k, v):
        dense_calls.append(q.shape[2])
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    q, k, v = (torch.randn(1, 2, 11, 128).to(torch.bfloat16) for _ in range(3))
    config = Config(exact=False, backend="sol")
    state = Request(config)
    out = sparse.attention(q, k, v, 5, config, state, dense_attention=dense_attention).reshape(1, 11, 2, 128)
    want = F.scaled_dot_product_attention(q[:, :, :5], k, v).transpose(1, 2)
    assert torch.equal(out[:, :5], want)
    assert (out[:, 5:] == 9).all()
    assert [c["sink_tokens"] for c in calls] == [11, 5]
    assert all(c["sink_start"] == 0 for c in calls)
    assert dense_calls == [5]
    assert state.sparse_calls == 1
    sparse.attention(q, k, v, 5, config, state, dense_attention=dense_attention)
    assert len(calls) == 3  # same-request shape gate reused
    assert dense_calls == [5, 5]


def test_failed_arithmetic_never_counts_sparse(monkeypatch):
    monkeypatch.setattr(
        sparse,
        "load_kernel",
        lambda device: lambda q, k, v, **kw: torch.full_like(v, float("nan")),
    )
    state = Request(Config(exact=False, backend="sol"))
    q = torch.ones(1, 1, 3, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="arithmetic gate failed"):
        sparse.attention(q, q, q, 1, state.config, state)
    assert state.sparse_calls == 0
