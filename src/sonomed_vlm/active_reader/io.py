"""Private, atomic experiment records and reproducibility fingerprints."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "pillow", "numpy", "pyyaml"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def code_revision() -> str:
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})
