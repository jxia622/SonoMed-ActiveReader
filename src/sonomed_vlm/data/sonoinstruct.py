"""SonoInstruct Parquet discovery, adaptation, and manifest-backed loading."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from sonomed_vlm.data.schema import ImageAsset, NormalizedExample, image_group_id
from sonomed_vlm.utils.io import read_jsonl, sha256_json

_IMAGE_TOKEN = re.compile(r"\s*<image>\s*", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RawRow:
    shard: Path
    row_index: int
    value: dict[str, Any]


def discover_parquet_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root).expanduser()
    if root.is_file():
        if root.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a Parquet file, got: {root}")
        return [root.resolve()]
    if not root.exists():
        raise FileNotFoundError(f"SonoInstruct root does not exist: {root}")
    preferred = sorted((root / "parquet_full_train").glob("*.parquet"))
    files = preferred or sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet shards found under {root}")
    return [path.resolve() for path in files]


def iter_raw_rows(
    data_root: str | Path,
    *,
    columns: list[str] | None = None,
    batch_size: int = 64,
) -> Iterator[RawRow]:
    for shard in discover_parquet_files(data_root):
        parquet = pq.ParquetFile(shard)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for value in batch.to_pylist():
                yield RawRow(shard=shard, row_index=row_index, value=value)
                row_index += 1


def _json_value(value: Any, *, field_name: str) -> Any:
    if value is None or value == "":
        return [] if field_name == "QA" else {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
    return value


def _resolve_image_path(value: str, data_root: Path, shard: Path) -> Path:
    normalized = value.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    candidates = [data_root / candidate, shard.parent / candidate, data_root / "images" / candidate.name]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def extract_images(row: Mapping[str, Any], data_root: Path, shard: Path) -> list[ImageAsset]:
    assets: list[ImageAsset] = []
    raw_images = row.get("images")
    if raw_images:
        if not isinstance(raw_images, list):
            raise ValueError("images must be a list")
        for item in raw_images:
            if isinstance(item, (bytes, bytearray, memoryview)):
                assets.append(ImageAsset(image_bytes=bytes(item)))
            elif isinstance(item, Mapping):
                raw_bytes = item.get("bytes")
                raw_path = item.get("path")
                assets.append(
                    ImageAsset(
                        image_bytes=bytes(raw_bytes) if raw_bytes is not None else None,
                        image_path=(
                            _resolve_image_path(str(raw_path), data_root, shard) if raw_path else None
                        ),
                    )
                )
            else:
                raise ValueError(f"Unsupported images entry: {type(item).__name__}")
    raw_paths = row.get("image_path")
    if raw_paths and not assets:
        paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
        assets.extend(
            ImageAsset(image_path=_resolve_image_path(str(path), data_root, shard)) for path in paths
        )
    return assets


def normalize_row(raw: RawRow, data_root: str | Path) -> list[NormalizedExample]:
    """Flatten one source row into instruction examples with shared image identity."""
    root = Path(data_root).expanduser().resolve()
    row = raw.value
    images = extract_images(row, root, raw.shard)
    if not images:
        raise ValueError("row has no image data")
    image_ids = [image.compute_id() for image in images]
    grouped_image_id = image_group_id(image_ids)
    qa_items = _json_value(row.get("QA"), field_name="QA")
    info = _json_value(row.get("info"), field_name="info")
    if not isinstance(qa_items, list):
        raise ValueError("QA must decode to a list")
    if not isinstance(info, dict):
        info = {"raw_info": info}

    examples: list[NormalizedExample] = []
    for qa_index, qa in enumerate(qa_items):
        if not isinstance(qa, Mapping):
            raise ValueError(f"QA[{qa_index}] must be an object")
        question = _IMAGE_TOKEN.sub("\n", str(qa.get("question") or "")).strip()
        answer = str(qa.get("answer") or "").strip()
        system_prompt = str(qa.get("system_prompt") or "").strip() or None
        task_type = qa.get("qa_type") or qa.get("type")
        identity = {
            "shard": raw.shard.name,
            "row": raw.row_index,
            "qa": qa_index,
            "image_id": grouped_image_id,
        }
        metadata = {
            "shard": str(raw.shard),
            "row_index": raw.row_index,
            "qa_index": qa_index,
            "image_paths": [str(asset.image_path) for asset in images if asset.image_path],
            "answer_label": qa.get("answer_label"),
            "options": qa.get("options") or [],
            "choice": qa.get("choice"),
            "source_info": info,
        }
        examples.append(
            NormalizedExample(
                example_id=sha256_json(identity),
                image_id=grouped_image_id,
                images=images,
                source_dataset=str(row.get("dataset")) if row.get("dataset") is not None else None,
                task_family=str(row.get("type")) if row.get("type") is not None else None,
                task_type=str(task_type) if task_type is not None else None,
                focus=str(row.get("focus")) if row.get("focus") is not None else None,
                system_prompt=system_prompt,
                user_prompt=question,
                assistant_response=answer,
                metadata=metadata,
            )
        )
    return examples


class _ParquetRowCache:
    """Small row-group cache used by manifest-backed datasets."""

    def __init__(self, max_groups: int = 1) -> None:
        self.max_groups = max_groups
        self._cache: OrderedDict[tuple[Path, int], tuple[int, list[dict[str, Any]]]] = OrderedDict()

    def read(self, shard: Path, row_index: int) -> dict[str, Any]:
        parquet = pq.ParquetFile(shard)
        start = 0
        for group_index in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group_index).num_rows
            if start <= row_index < start + count:
                key = (shard, group_index)
                if key not in self._cache:
                    rows = parquet.read_row_group(group_index).to_pylist()
                    self._cache[key] = (start, rows)
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.max_groups:
                        self._cache.popitem(last=False)
                group_start, rows = self._cache[key]
                return rows[row_index - group_start]
            start += count
        raise IndexError(f"Row {row_index} outside {shard} ({start} rows)")


class ManifestDataset:
    """PyTorch-compatible dataset resolving clinical text/images from source Parquet."""

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        *,
        task_filters: list[str] | None = None,
        source_filters: list[str] | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        tasks = set(task_filters or [])
        sources = set(source_filters or [])
        self.records = [
            record
            for record in read_jsonl(manifest)
            if (not tasks or record.get("task_family") in tasks or record.get("task_type") in tasks)
            and (not sources or record.get("source_dataset") in sources)
        ]
        if not self.records:
            raise ValueError(f"Manifest contains no matching examples: {manifest}")
        self._cache = _ParquetRowCache()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> NormalizedExample:
        record = self.records[index]
        shard = Path(record["shard"])
        if not shard.is_absolute():
            shard = self.data_root / shard
        row_index = int(record["row_index"])
        qa_index = int(record["qa_index"])
        raw = RawRow(shard=shard.resolve(), row_index=row_index, value=self._cache.read(shard, row_index))
        examples = normalize_row(raw, self.data_root)
        try:
            example = examples[qa_index]
        except IndexError as exc:
            raise ValueError(
                f"Manifest QA index {qa_index} no longer exists for {shard}:{row_index}"
            ) from exc
        if example.example_id != record["example_id"] or example.image_id != record["image_id"]:
            raise ValueError(
                f"Manifest/source mismatch for {shard}:{row_index}:{qa_index}; rebuild manifests"
            )
        return example
