"""Internal examples and image identities."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from sonomed_vlm.utils.io import sha256_bytes


@dataclass(slots=True)
class ImageAsset:
    """One image stored inline or referenced by a local path."""

    image_bytes: bytes | None = None
    image_path: Path | None = None
    image_id: str | None = None

    def read_bytes(self) -> bytes:
        if self.image_bytes is not None:
            return self.image_bytes
        if self.image_path is None:
            raise ValueError("Image asset contains neither bytes nor a path")
        try:
            return self.image_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Unable to read image: {self.image_path}") from exc

    def compute_id(self) -> str:
        if self.image_id is None:
            self.image_id = sha256_bytes(self.read_bytes())
        return self.image_id

    def open(self) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(self.read_bytes()))
            image.load()
            return image
        except Exception as exc:
            location = str(self.image_path) if self.image_path else "embedded bytes"
            raise ValueError(f"Unreadable image from {location}: {exc}") from exc


def image_group_id(image_ids: list[str]) -> str:
    """Create an ordered multi-image identity; a single image keeps its SHA-256."""
    if not image_ids:
        raise ValueError("At least one image identity is required")
    if len(image_ids) == 1:
        return image_ids[0]
    digest = hashlib.sha256(b"sonomed-vlm-multi-image-v1\0")
    for image_id in image_ids:
        digest.update(bytes.fromhex(image_id))
    return digest.hexdigest()


@dataclass(slots=True)
class NormalizedExample:
    example_id: str
    image_id: str
    images: list[ImageAsset]
    source_dataset: str | None
    task_family: str | None
    task_type: str | None
    focus: str | None
    system_prompt: str | None
    user_prompt: str
    assistant_response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_ids(self) -> list[str]:
        return [image.compute_id() for image in self.images]

    def to_manifest_record(self) -> dict[str, Any]:
        """Return identifiers/metadata only, never embedded bytes or clinical text."""
        locator_keys = ("shard", "row_index", "qa_index", "image_paths")
        record: dict[str, Any] = {
            "example_id": self.example_id,
            "image_id": self.image_id,
            "image_ids": self.image_ids,
            "source_dataset": self.source_dataset,
            "task_family": self.task_family,
            "task_type": self.task_type,
            "focus": self.focus,
        }
        record.update({key: self.metadata[key] for key in locator_keys if key in self.metadata})
        return record
