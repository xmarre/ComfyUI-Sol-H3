from dataclasses import asdict, dataclass
import hashlib
import json
import math

KEY = "sol_h3_runtime_v1"
OWNER = "comfyui_sol_h3"


@dataclass(frozen=True)
class Config:
    exact: bool = True
    backend: str = "dense"
    tau: float = 1.0
    dense_evaluations: int = 1
    dense_layers: int = 2

    def __post_init__(self):
        if self.backend not in {"dense", "sol"}:
            raise ValueError("Supported backends: dense, sol. SOL-BSA is not validated on SM120.")
        if not math.isfinite(self.tau) or not 0 <= self.tau <= 3:
            raise ValueError("tau must be finite and between 0 and 3")
        for name in ("dense_evaluations", "dense_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def metadata(self):
        data = {"api": 1, "owner": OWNER, **asdict(self),
                "approximate": self.backend != "dense", "sink_mode": "prefix",
                "threshold": "diag", "kernel_contract": "sana-2936c476-sol-sm120",
                "exact_kernel": "rounded-affine-v1" if self.exact else "native",
                "history_policy": "sparse_forecasting_blocked" if self.backend == "sol" else "dense_compatible"}
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


def reject_sparse_conflicts(options):
    # No consumer currently acknowledges the v1 history fingerprint. Check on every
    # evaluation as well as installation, including patches applied after this node.
    names = [str(k).lower() for k in options]
    names += [str(k).lower() for group in options.get("wrappers", {}).values() for k in group]
    if any("spectrum" in name for name in names):
        raise RuntimeError("SOL + Spectrum is gated until Spectrum consumes sol_h3_runtime_v1 history identity")
    if "vdn_h3_external_sequence_v1" in options:
        raise RuntimeError("SOL does not yet support VDN external/reduced or mixed-grid sequences")
