"""Numerically correct Hugging Face Trainer checkpoint discovery."""

from __future__ import annotations

import json
import re
from pathlib import Path

_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(path: str | Path) -> int:
    match = _CHECKPOINT.match(Path(path).name)
    if not match:
        raise ValueError(f"Not a checkpoint directory name: {path}")
    return int(match.group(1))


def is_valid_checkpoint(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_dir() or not _CHECKPOINT.match(candidate.name):
        return False
    state = candidate / "trainer_state.json"
    if not state.is_file():
        return False
    try:
        value = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(value.get("global_step", -1)) == checkpoint_step(candidate)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    candidates = [path for path in root.glob("checkpoint-*") if is_valid_checkpoint(path)]
    return max(candidates, key=checkpoint_step) if candidates else None


def resolve_resume_checkpoint(value: str | None, output_dir: str | Path) -> str | None:
    if value is None:
        return None
    if value.casefold() in {"auto", "latest"}:
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            raise FileNotFoundError(f"No valid checkpoint found under {output_dir}")
        return str(latest)
    if value.casefold() == "auto-if-present":
        latest = find_latest_checkpoint(output_dir)
        return str(latest) if latest is not None else None
    path = Path(value).expanduser().resolve()
    if not is_valid_checkpoint(path):
        raise ValueError(f"Checkpoint is incomplete or inconsistent: {path}")
    return str(path)
