from pathlib import Path
import shutil
import subprocess
import tempfile

from vendor_sol_attn import REVISION, SUBTREE, relocate


repo = Path(__file__).resolve().parents[1]
sana = repo / "_deps/Sana"
vendor = repo / "sol_h3/_vendor/sol_attn"
patch = repo / "tools/rectangular_sm120.patch"


def git_sana(*args):
    return subprocess.check_output(["git", "-C", str(sana), *args])


names = git_sana("ls-tree", "-r", "--name-only", REVISION, "--", SUBTREE).decode().splitlines()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    target = root / "sol_h3/_vendor/sol_attn"
    for full in names:
        name = full[len(SUBTREE) + 1 :]
        raw = git_sana("show", f"{REVISION}:{full}")
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(relocate(raw, name))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "ci"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "relocated upstream"], cwd=root, check=True)
    shutil.copytree(vendor, target, dirs_exist_ok=True)
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", "sol_h3/_vendor/sol_attn"], cwd=root
    )
    patch.write_bytes(diff)
