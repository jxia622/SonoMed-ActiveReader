"""Consistent structured logging without secret leakage."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_SECRET_PATTERN = re.compile(r"(?i)(hf_token|token|secret|password)=([^\s]+)")


class SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET_PATTERN.sub(r"\1=<redacted>", str(record.msg))
        return True


def configure_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
    for handler in handlers:
        handler.addFilter(SecretFilter())


def safe_environment() -> dict[str, str]:
    allowed_prefixes = ("SLURM_", "CUDA_", "NCCL_", "SONOMED_")
    blocked_fragments = ("TOKEN", "SECRET", "PASSWORD", "KEY")
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(allowed_prefixes)
        and not any(fragment in key.upper() for fragment in blocked_fragments)
    }
