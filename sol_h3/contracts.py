from dataclasses import asdict, dataclass
from .provenance import CONTRACT, SOURCE, REVISION
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
                "kernel_contract": CONTRACT if owns_sol else None,
                "sol_source": SOURCE if owns_sol else None,
                "sana_revision": REVISION if owns_sol else None,
                "exact_kernel": "rounded-affine-v1" if self.exact else "native",
                "history_policy": "attention_backend_history_v1"}
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
