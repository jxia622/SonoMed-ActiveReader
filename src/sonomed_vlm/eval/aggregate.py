"""Aggregate numeric metrics overall and by task/source."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any


def _numeric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.get("score", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key].append(float(value))
    return {key: fmean(items) for key, items in sorted(values.items()) if items}


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"examples": len(rows), "overall": _numeric_means(rows)}
    for field in ("task_family", "task_type", "source_dataset", "focus"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field) or "<missing>")].append(row)
        output[f"per_{field}"] = {
            name: {"examples": len(group), **_numeric_means(group)}
            for name, group in sorted(groups.items())
        }
    return output
