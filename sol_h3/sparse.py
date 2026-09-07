"""Native attention bridge to the separately installed, pinned Sol kernel."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


_VERIFIED_ROOTS = set()


def verify_sources(root, manifest):
    root = Path(root).resolve()
    git_blobs = manifest.get("git_blobs")
    sha256 = manifest.get("sha256")
    if not isinstance(git_blobs, dict) or not isinstance(sha256, dict) or not git_blobs:
        raise RuntimeError("Pinned sol_attn manifest is incomplete")
    if set(git_blobs) != set(sha256):
        raise RuntimeError("Pinned sol_attn manifest provenance/security file sets differ")
    for name, expected in sha256.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError("Pinned sol_attn manifest contains an invalid SHA-256 entry")
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Pinned sol_attn path escapes package root: {name}") from exc
        if not path.is_file():
            raise RuntimeError(f"Pinned sol_attn file missing: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"sol_attn source mismatch: {name}; install the documented pinned revision")


def _find_package_root():
    roots = []
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        candidate = Path(entry or ".") / "sol_attn"
        try:
            if not (candidate / "__init__.py").is_file():
                continue
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved not in roots:
            roots.append(resolved)
    if not roots:
        raise RuntimeError(
            "Install the pinned Sol-Attn backend and expose its parent directory on PYTHONPATH; "
            "see README.md. No dense fallback."
        )
    if len(roots) != 1:
        raise RuntimeError(f"Multiple sol_attn package roots are visible on PYTHONPATH: {roots}")
    return roots[0]


def _load_verified_interface(root, manifest):
    root = Path(root).resolve()
    # Verify every pinned package file before Python is allowed to execute package
    # __init__ or interface code. The Git blob IDs remain in the manifest only as
    # provenance; SHA-256 is the runtime integrity gate.
    verify_sources(root, manifest)
    loaded = any(name == "sol_attn" or name.startswith("sol_attn.") for name in sys.modules)
    if loaded:
        if root not in _VERIFIED_ROOTS or "sol_attn.interface" not in sys.modules:
            raise RuntimeError(
                "sol_attn was imported before Sol-H3 integrity verification; restart ComfyUI "
                "with the pinned package exposed on PYTHONPATH"
            )
        module = sys.modules["sol_attn.interface"]
    else:
        try:
            module = importlib.import_module("sol_attn.interface")
        except ImportError as exc:
            raise RuntimeError("Install the pinned Sol-Attn backend; see README.md. No dense fallback.") from exc
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve().parent != root:
            raise RuntimeError("Imported sol_attn.interface does not come from the verified package root")
        # Recheck after import so a package that mutates lazy-import source files
        # cannot turn a successful pre-import verification into a later code load.
        verify_sources(root, manifest)
        _VERIFIED_ROOTS.add(root)
    return module


def load_kernel(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("This experimental SOL integration currently targets single-GPU SM120 only")
    root = _find_package_root()
    manifest = json.loads(Path(__file__).with_name("sol_manifest.json").read_text())
    module = _load_verified_interface(root, manifest)
    backend = module.get_sol_attn_backend(device)
    if backend != "cute_sm120":
        raise RuntimeError(f"SOL requires cute_sm120, selected {backend}; install CuTe DSL/CUDA dependencies")
    return module.sol_attn


def error_metrics(got, want):
    delta = got.float() - want.float()
    return {"max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean()),
            "rel_l2": float(torch.linalg.vector_norm(delta) /
                            torch.linalg.vector_norm(want.float()).clamp_min(1e-12))}


def _dense_reference(q, k, v, dense_attention):
    if dense_attention is not None:
        out = dense_attention(q, k, v)
        expected = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        if out.shape != expected:
            raise RuntimeError(f"Dense SOL reference returned {tuple(out.shape)}, expected {expected}")
        return out
    # Test/reference fallback only. Production passes the active Comfy dense provider
    # from the optimized-attention wrapper so prefix rows preserve current semantics.
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def attention(q, k, v, prefix, config, state, dense_attention=None):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[-1] != 128:
        raise RuntimeError("SOL requires matching QKV [1, heads, packed rows, 128]")
    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
        raise RuntimeError("SOL requires BF16 QKV on the same device")
    if state.kernel is None:
        state.kernel = load_kernel(q.device)
        state.kernel_device = q.device
    elif q.device != state.kernel_device:
        raise RuntimeError("SOL compute device changed within a sampling request")
    # SM120 interleaved views remain unvalidated upstream; contiguous BTHD
    # buffers avoid inheriting SM100-specific zero-copy assumptions.
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    key = (q.device, q.dtype, tuple(q.shape))
    if key not in state.sparse_verified:
        # Make EVERY KV block exact using the sink, rather than assuming an
        # extreme tau necessarily selects every block on arbitrary activations.
        got = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                           sink_start=0, sink_tokens=qb.shape[1])
        want = _dense_reference(q, k, v, dense_attention)
        metrics = error_metrics(got, want)
        limits = {"max_abs": 0.15 if qb.shape[1] >= 32768 else 0.08,
                  "mean_abs": 0.002, "rel_l2": 0.005}
        if not all(metrics[n] <= limits[n] for n in limits):
            raise RuntimeError(f"SOL all-selected arithmetic gate failed: {metrics}")
        state.sparse_verified.add(key)
        state.gates.append({"shape": list(q.shape), **metrics})
        del got, want
    out = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                       sink_start=0, sink_tokens=prefix)
    # Generated audio, references, conditioning and text query rows remain dense
    # through the same active Comfy dense provider used by the baseline path.
    out[:, :prefix] = _dense_reference(q[:, :, :prefix], k, v, dense_attention)
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], -1)
