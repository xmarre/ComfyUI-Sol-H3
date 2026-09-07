"""Native attention bridge to the separately installed, pinned Sol kernel."""
import hashlib
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_kernel(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("This experimental SOL integration currently targets single-GPU SM120 only")
    try:
        module = importlib.import_module("sol_attn.interface")
    except ImportError as exc:
        raise RuntimeError("Install the pinned Sol-Attn backend; see README.md. No dense fallback.") from exc
    root = Path(module.__file__).resolve().parent
    manifest = json.loads(Path(__file__).with_name("sol_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Pinned sol_attn file missing: {name}")
        raw = path.read_bytes()
        digest = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if digest != sha:
            raise RuntimeError(f"sol_attn source mismatch: {name}; install the documented pinned revision")
    backend = module.get_sol_attn_backend(device)
    if backend != "cute_sm120":
        raise RuntimeError(f"SOL requires cute_sm120, selected {backend}; install CuTe DSL/CUDA dependencies")
    return module.sol_attn


def error_metrics(got, want):
    delta = got.float() - want.float()
    return {"max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean()),
            "rel_l2": float(torch.linalg.vector_norm(delta) /
                            torch.linalg.vector_norm(want.float()).clamp_min(1e-12))}


def attention(q, k, v, prefix, config, state):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[-1] != 128:
        raise RuntimeError("SOL requires matching QKV [1, heads, packed rows, 128]")
    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
        raise RuntimeError("SOL requires BF16 QKV on the same device")
    if state.kernel is None:
        state.kernel = load_kernel(q.device)
    # SM120 interleaved views remain unvalidated upstream; contiguous BTHD
    # buffers avoid inheriting SM100-specific zero-copy assumptions.
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    key = (q.device, q.dtype, tuple(q.shape))
    if key not in state.sparse_verified:
        # Make EVERY KV block exact using the sink, rather than assuming an
        # extreme tau necessarily selects every block on arbitrary activations.
        got = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                           sink_start=0, sink_tokens=qb.shape[1])
        want = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
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
    # Generated audio, references, conditioning and text query rows remain dense.
    out[:, :prefix] = F.scaled_dot_product_attention(q[:, :, :prefix], k, v).transpose(1, 2)
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], -1)
