import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def git_state(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    # Record a hash of the tracked working diff as well as the revision; untracked
    # research source is also hashed by the runner's source inventory.
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()}
