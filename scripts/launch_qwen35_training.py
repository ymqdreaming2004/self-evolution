"""Launch the verified Qwen3.5 QLoRA job as a detached Windows process."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "train_venv" / "venv" / "Scripts" / "llamafactory-cli.exe"
CONFIG = ROOT / "outputs" / "qwen35-fridge-lora" / "training_args.yaml"
OUTPUT = ROOT / "outputs" / "qwen35-fridge-qlora"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_DATASETS_CACHE"] = str(ROOT / "train_venv" / "hf_cache_qwen35")
    env["PYTHONUNBUFFERED"] = "1"

    with (OUTPUT / "train.stdout.log").open("ab", buffering=0) as stdout, (
        OUTPUT / "train.stderr.log"
    ).open("ab", buffering=0) as stderr:
        process = subprocess.Popen(
            [str(CLI), "train", str(CONFIG)],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NO_WINDOW
            ),
            close_fds=True,
        )

    (OUTPUT / "train.pid").write_text(str(process.pid), encoding="ascii")
    print(f"Started Qwen3.5 QLoRA training (PID {process.pid}).")


if __name__ == "__main__":
    main()
