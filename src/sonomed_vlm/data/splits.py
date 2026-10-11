"""Deterministic image-overlap-aware splits and nested scale subsets."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            self.parent[merge] = keep


def attach_leakage_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union examples sharing any image, including overlapping multi-image rows."""
    union_find = _UnionFind()
    for record in records:
        image_ids = list(record.get("image_ids") or [record["image_id"]])
        for image_id in image_ids:
            union_find.find(image_id)
        for image_id in image_ids[1:]:
            union_find.union(image_ids[0], image_id)
    output = []
    for record in records:
        image_ids = list(record.get("image_ids") or [record["image_id"]])
        group = min(union_find.find(image_id) for image_id in image_ids)
        output.append({**record, "leakage_group_id": group})
    return output


def _stable_score(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def split_records(
    records: list[dict[str, Any]], val_fraction: float = 0.05, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    grouped_records = attach_leakage_groups(records)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in grouped_records:
        components[record["leakage_group_id"]].append(record)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, members in components.items():
        sources = sorted({str(item.get("source_dataset") or "") for item in members})
        tasks = sorted({str(item.get("task_family") or item.get("task_type") or "") for item in members})
        strata[(sources[0], tasks[0])].append(group_id)

    val_groups: set[str] = set()
    for group_ids in strata.values():
        ordered = sorted(group_ids, key=lambda item: _stable_score(item, seed))
        count = int(round(len(ordered) * val_fraction))
        if len(ordered) >= max(2, math.ceil(1 / val_fraction)):
            count = max(1, count)
        val_groups.update(ordered[:count])
    if not val_groups and len(components) > 1:
        val_groups.add(min(components, key=lambda item: _stable_score(item, seed)))

    train, val = [], []
    for record in grouped_records:
        (val if record["leakage_group_id"] in val_groups else train).append(record)
    return train, val


def nested_subsets(
    train_records: list[dict[str, Any]],
    fractions: tuple[int, ...] = (1, 5, 10, 25, 50, 100),
    seed: int = 42,
) -> dict[int, list[dict[str, Any]]]:
    if not fractions or any(value <= 0 or value > 100 for value in fractions):
        raise ValueError("fractions must be percentages in [1, 100]")
    records = attach_leakage_groups(train_records)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        components[record["leakage_group_id"]].append(record)
    ordered = sorted(components, key=lambda item: _stable_score(item, seed))
    output: dict[int, list[dict[str, Any]]] = {}
    for fraction in sorted(set(fractions)):
        group_count = len(ordered) if fraction == 100 else max(1, math.ceil(len(ordered) * fraction / 100))
        selected = set(ordered[:group_count])
        output[fraction] = [
            record for record in records if record["leakage_group_id"] in selected
        ]
    return output


def assert_no_image_leakage(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> None:
    left_ids = {image for record in left for image in record.get("image_ids", [record["image_id"]])}
    right_ids = {image for record in right for image in record.get("image_ids", [record["image_id"]])}
    overlap = left_ids & right_ids
    if overlap:
        raise ValueError(f"Detected {len(overlap)} image hashes in both splits")
