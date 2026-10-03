#!/usr/bin/env python3
"""Check Patcher's sequential preflight and declared-base overlay application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def git(path: Path, *args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=ComfyUI Patcher",
            "-c",
            "user.email=patcher@local.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=path,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Tracked and declared base revision")
    parser.add_argument("--overlay", required=True, help="Earlier independent overlay revision")
    parser.add_argument("--candidate", default="HEAD", help="Later independent overlay revision")
    parser.add_argument("--tests", nargs="*", default=[], help="Existing pytest cases to run on the composed tree")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    base, earlier, candidate = (
        git(repository, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
        for revision in (args.base, args.overlay, args.candidate)
    )
    worktrees = []
    with tempfile.TemporaryDirectory(prefix="sol-overlay-", dir=repository.parent) as directory:
        try:
            preview = Path(directory) / "preview"
            composed = Path(directory) / "composed"
            for path in (preview, composed):
                git(repository, "worktree", "add", "--detach", str(path), base)
                worktrees.append(path)
            for head in (earlier, candidate):
                git(preview, "merge", "--no-ff", "-m", "comfyui-patcher sequential merge preview", head)
                branch_point = git(repository, "merge-base", base, head).decode().strip()
                patch = git(repository, "diff", "--binary", "--full-index", "--find-renames", branch_point, head, "--")
                if patch:
                    git(composed, "apply", "--3way", "--index", "-", data=patch)
                git(composed, "commit", "--no-verify", "--allow-empty", "-m", "comfyui-patcher overlay delta")
            preview_tree = git(preview, "rev-parse", "HEAD^{tree}").decode().strip()
            composed_tree = git(composed, "rev-parse", "HEAD^{tree}").decode().strip()
            if preview_tree != composed_tree:
                raise RuntimeError(f"preflight tree {preview_tree} differs from applied tree {composed_tree}")
            print(json.dumps({"preflight": "clean", "composed_tree": composed_tree}), flush=True)
            if args.tests:
                subprocess.run([sys.executable, "-m", "pytest", "-q", *args.tests], cwd=composed, check=True)
        finally:
            for path in reversed(worktrees):
                git(repository, "worktree", "remove", "--force", str(path))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stderr.write(exc.stdout.decode(errors="replace"))
        if exc.stderr:
            sys.stderr.write(exc.stderr.decode(errors="replace"))
        raise SystemExit(exc.returncode) from exc
