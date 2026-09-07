from dataclasses import asdict, dataclass
import hashlib
import json
import math

KEY = "sol_h3_runtime_v1"
OWNER = "comfyui_sol_h3"


@dataclass(frozen=True)
class Config:
    exact: bool = True
    backend: str = "inherit"
    tau: float = 1.0
    dense_evaluations: int = 1
    dense_layers: int = 2

    def __post_init__(self):
        if self.backend not in {"inherit", "sol"}:
            raise ValueError("Supported backends: inherit, sol. SOL-BSA is not validated on SM120.")
        if not math.isfinite(self.tau) or not 0 <= self.tau <= 3:
            raise ValueError("tau must be finite and between 0 and 3")
        for name in ("dense_evaluations", "dense_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def metadata(self):
        owns_sol = self.backend == "sol"
        data = {"api": 1, "owner": OWNER, **asdict(self),
                "approximate": owns_sol,
                "attention_ownership": "sol" if owns_sol else "inherit",
                "sink_mode": "prefix" if owns_sol else None,
                "threshold": "diag" if owns_sol else None,
                "kernel_contract": "sana-2936c476-sol-sm120" if owns_sol else None,
                "exact_kernel": "rounded-affine-v1" if self.exact else "native",
                "history_policy": "sparse_forecasting_blocked" if owns_sol else "inherits_upstream_numerics"}
        data["fingerprint"] = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return data


def prefix_length(layout, rows):
    """Validate the current packed spans; never infer geometry from a stale signature."""
    segments = getattr(layout, "segments", None)
    if not segments or getattr(layout, "seq_len", None) != rows:
        raise RuntimeError("SOL needs the current native packed layout and matching row count")
    cursor = 0
    video_start = None
    for a, b, kind in segments:
        if kind not in {"text", "cond", "cond_audio", "ref_img", "ref_audio", "audio", "video"}:
            raise RuntimeError(f"Unrecognized packed segment kind: {kind}")
        if type(a) is not int or type(b) is not int or a != cursor or b < a:
            raise RuntimeError("Packed segments must cover the sequence contiguously")
        if kind == "video":
            if video_start is not None or b != rows:
                raise RuntimeError("SOL requires one contiguous target-video tail")
            video_start = a
        cursor = b
    if cursor != rows or video_start is None or not 0 < video_start < rows:
        raise RuntimeError("SOL requires a nonempty prefix and target-video tail")
    return video_start


def adaln_status(model):
    if getattr(model, "use_adaln_curves", False):
        if getattr(model, "adaln_t_table", None) is None:
            raise RuntimeError("Curve AdaLN model is missing adaln_t_table")
        return "not_applicable_compact_curve"
    return "native_full_width_no_schedule_precompute"


def provider_names(options):
    """Flatten provider keys without depending on private wrapper/callback object types."""
    names = [str(k).lower() for k in options]
    names += [str(k).lower() for group in options.get("wrappers", {}).values() for k in group]
    names += [str(k).lower() for group in options.get("callbacks", {}).values() for k in group]
    return names


def reject_forecasting_conflicts(options):
    """Reject known numerical-backend transitions that an active forecaster cannot identify."""
    names = provider_names(options)
    spectrum = any("spectrum" in name for name in names)
    scheduled_sparse = any("block_sparse_attention" in name for name in names)
    if spectrum and scheduled_sparse:
        raise RuntimeError(
            "Spectrum + ComfyUI Block Sparse Attention is gated: the sparse schedule can mix dense and sparse "
            "actual anchors, while the audited Spectrum consumer does not track that backend transition"
        )


def reject_sparse_conflicts(options):
    # No consumer currently acknowledges the v1 history fingerprint. Check on every
    # evaluation as well as installation, including patches applied after this node.
    names = provider_names(options)
    if any("spectrum" in name for name in names):
        raise RuntimeError("SOL + Spectrum is gated until Spectrum consumes sol_h3_runtime_v1 history identity")
    if "vdn_h3_external_sequence_v1" in options:
        raise RuntimeError("SOL does not yet support VDN external/reduced or mixed-grid sequences")
    if any("block_sparse_attention" in name for name in names):
        raise RuntimeError("SOL cannot be stacked with ComfyUI Block Sparse Attention; select exactly one sparse provider")
    if options.get("optimized_attention_override") is not None:
        raise RuntimeError("SOL cannot own attention alongside an existing optimized attention override")
