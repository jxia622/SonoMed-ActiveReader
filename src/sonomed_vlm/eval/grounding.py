"""Grounding parsing and IoU with explicit invalid-output handling."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ParsedBox:
    coordinates: tuple[float, float, float, float] | None
    valid: bool
    reason: str


def parse_box(value: str, coordinate_max: float = 1.0) -> ParsedBox:
    numbers = [float(match.group()) for match in _NUMBER.finditer(value)]
    if len(numbers) != 4:
        return ParsedBox(None, False, "expected_exactly_four_numbers")
    x1, y1, x2, y2 = numbers
    if not all(math.isfinite(number) for number in numbers):
        return ParsedBox(None, False, "non_finite")
    if not all(0 <= number <= coordinate_max for number in numbers):
        return ParsedBox(None, False, "out_of_range")
    if x2 <= x1 or y2 <= y1:
        return ParsedBox(None, False, "non_positive_area")
    return ParsedBox((x1, y1, x2, y2), True, "ok")


def box_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0
