from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_json_object(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if stdout[index + consumed :].strip():
            continue
        if not isinstance(value, dict):
            raise RuntimeError("real-H3 replay probe returned a non-object JSON payload")
        return value
    raise RuntimeError("real-H3 replay probe did not return a JSON object")


def _resolve_keyless_repo(comfy_root: Path) -> Path:
    candidate = comfy_root / "custom_nodes" / "minimax-h3-keyless"
    if not candidate.is_dir():
        raise RuntimeError(
            "MiniMax-H3-Keyless checkout was not found at "
            f"{candidate}; the replay receipt loader is required"
        )
    return candidate


def _report_path(output_root: Path, payload: dict[str, Any], receipt_sha256: str) -> Path:
    block_part = "-".join(
        str(int(row["block_index"]))
        for row in payload.get("results", [])
        if isinstance(row, dict) and "block_index" in row
    ) or "unknown"
    checkpoint_kind = str(payload.get("checkpoint_kind", "unknown"))
    directory = output_root / "keyless_real_h3_replay_reports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (
        f"real-h3-replay-{receipt_sha256[:12]}.{checkpoint_kind}."
        f"blocks-{block_part}.json"
    )


class SolH3KeylessRealH3Replay:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
        except ImportError:
            checkpoint_names = []
        else:
            checkpoint_names = folder_paths.get_filename_list("diffusion_models")

        return {
            "required": {
                "capture_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "checkpoint_name": (
                    checkpoint_names,
                ),
                "checkpoint_kind": (
                    ["teacher", "keyless"],
                    {"default": "teacher"},
                ),
                "q_start": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1_000_000, "step": 1},
                ),
                "q_rows": (
                    "INT",
                    {"default": 257, "min": 1, "max": 2048, "step": 1},
                ),
                "v_start": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1_000_000, "step": 1},
                ),
                "v_rows": (
                    "INT",
                    {"default": 511, "min": 1, "max": 2048, "step": 1},
                ),
                "repeats": (
                    "INT",
                    {"default": 10, "min": 1, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report_json", "report_path")
    FUNCTION = "run"
    CATEGORY = "model/optimizations/Sol-H3/diagnostics"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Run the bounded Keyless K1/K2 real-H3 replay against a local capture bundle. "
        "The 1+ GB capture never leaves the machine; a small JSON report is written under "
        "ComfyUI/output/keyless_real_h3_replay_reports and returned as text."
    )

    def run(
        self,
        capture_path: str,
        checkpoint_name: str,
        checkpoint_kind: str,
        q_start: int,
        q_rows: int,
        v_start: int,
        v_rows: int,
        repeats: int,
    ):
        import folder_paths

        capture = Path(capture_path).expanduser().resolve()
        if not capture.is_file():
            raise FileNotFoundError(f"real-H3 capture bundle does not exist: {capture}")
        receipt = capture.with_suffix(capture.suffix + ".receipt.json")
        if not receipt.is_file():
            raise FileNotFoundError(f"real-H3 capture receipt does not exist: {receipt}")
        receipt_sha256 = _sha256_file(receipt)

        checkpoint = Path(
            folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_name)
        ).resolve()
        comfy_root = Path(folder_paths.__file__).resolve().parent
        keyless_repo = _resolve_keyless_repo(comfy_root)
        repo_root = Path(__file__).resolve().parents[1]
        probe = repo_root / "tools" / "keyless_real_h3_replay_probe.py"
        if not probe.is_file():
            raise RuntimeError(f"real-H3 replay probe is missing: {probe}")

        command = [
            sys.executable,
            str(probe),
            "--capture-bundle",
            str(capture),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-kind",
            str(checkpoint_kind),
            "--keyless-repo",
            str(keyless_repo),
            "--expected-receipt-sha256",
            receipt_sha256,
            "--device",
            "cuda:0",
            "--q-start",
            str(int(q_start)),
            "--q-rows",
            str(int(q_rows)),
            "--v-start",
            str(int(v_start)),
            "--v-rows",
            str(int(v_rows)),
            "--repeats",
            str(int(repeats)),
            "--blocks",
            "0",
            "25",
            "49",
        ]
        env = dict(os.environ)
        prior = env.get("PYTHONPATH", "")
        roots = [str(comfy_root), str(keyless_repo)]
        if prior:
            roots.append(prior)
        env["PYTHONPATH"] = os.pathsep.join(roots)

        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            raise RuntimeError(
                "real-H3 replay probe failed "
                f"(exit {completed.returncode}):\n{detail}"
            )

        payload = _extract_json_object(completed.stdout)
        report = _report_path(
            Path(folder_paths.get_output_directory()).resolve(),
            payload,
            receipt_sha256,
        )
        if report.exists():
            raise FileExistsError(
                "real-H3 replay report already exists; evidence is immutable: "
                f"{report}"
            )
        report_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        report.write_text(report_text, encoding="utf-8")
        return (report_text, str(report))
