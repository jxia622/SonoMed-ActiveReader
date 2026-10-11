"""Run provenance and compute metadata."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonomed_vlm.training.distributed import world_info
from sonomed_vlm.utils.io import sha256_file, write_json
from sonomed_vlm.utils.logging import safe_environment


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted/no-git-revision"


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "accelerate", "peft", "datasets", "pyarrow", "pillow")
    output = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def resolve_dataset_revision(data_root: Path, manifest: Path) -> str:
    """Prefer an upstream Git revision; otherwise use an explicit or manifest identity."""
    result = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=data_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if (data_root / ".git").exists()
        else None
    )
    if result is not None and result.returncode == 0:
        return f"git:{result.stdout.strip()}"
    explicit = data_root / "dataset_revision.txt"
    declared = explicit.read_text(encoding="utf-8").strip() if explicit.is_file() else ""
    if declared:
        return "declared:" + declared
    return "manifest-sha256:" + sha256_file(manifest)


def gpu_metadata() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "count": 0}
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name, "memory_bytes": props.total_memory})
        return {"available": True, "count": len(devices), "devices": devices}
    except ImportError:
        return {"available": False, "count": 0, "reason": "torch-not-installed"}


def collect_metadata(
    *,
    run_id: str,
    project_root: Path,
    train_manifest: Path,
    seed: int,
    model_revision: str | None,
    dataset_revision: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "git_commit": git_commit(project_root),
        "dataset_manifest": str(train_manifest),
        "dataset_manifest_sha256": sha256_file(train_manifest),
        "model_revision": model_revision,
        "dataset_revision": dataset_revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "distributed": world_info().to_dict(),
        "gpu": gpu_metadata(),
        "environment": safe_environment(),
    }


def save_environment_files(run_dir: Path, metadata: dict[str, Any]) -> None:
    write_json(run_dir / "metadata.json", metadata)
    (run_dir / "environment.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in metadata["packages"].items()) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(str(metadata["git_commit"]) + "\n", encoding="utf-8")
    (run_dir / "model_revision.txt").write_text(
        str(metadata["model_revision"]) + "\n", encoding="utf-8"
    )
    (run_dir / "dataset_revision.txt").write_text(
        str(metadata["dataset_revision"]) + "\n", encoding="utf-8"
    )
    (run_dir / "slurm_info.txt").write_text(
        json.dumps(metadata["distributed"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_compute(metadata_path: Path, started_at: float, examples: int | None = None) -> None:
    import time

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    seconds = time.monotonic() - started_at
    gpu_count = int(metadata.get("gpu", {}).get("count", 0))
    metadata["completed_at"] = datetime.now(UTC).isoformat()
    metadata["wall_seconds"] = seconds
    metadata["gpu_hours"] = seconds * gpu_count / 3600
    if examples is not None and seconds:
        metadata["examples_per_second_wall"] = examples / seconds
    try:
        import torch

        if torch.cuda.is_available():
            metadata["max_allocated_gpu_memory_bytes"] = max(
                torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
            )
    except ImportError:
        pass
    write_json(metadata_path, metadata)
