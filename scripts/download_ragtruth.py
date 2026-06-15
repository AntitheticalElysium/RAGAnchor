"""Fetch the RAGTruth dataset into data/ragtruth/. Idempotent."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from raganchor.config import RAGTRUTH_DIR

REPO = "https://github.com/ParticleMedia/RAGTruth"
FILES = ["response.jsonl", "source_info.jsonl"]


def main() -> None:
    RAGTRUTH_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAGTRUTH_DIR / f).exists() for f in FILES):
        print(f"[ragtruth] already present in {RAGTRUTH_DIR}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", REPO, tmp], check=True)
        for f in FILES:
            (RAGTRUTH_DIR / f).write_bytes((Path(tmp) / "dataset" / f).read_bytes())
    print(f"[ragtruth] wrote {FILES} to {RAGTRUTH_DIR}")


if __name__ == "__main__":
    sys.exit(main())
