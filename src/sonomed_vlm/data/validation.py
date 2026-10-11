"""Dataset integrity validation with explicit exclusion accounting."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sonomed_vlm.data.schema import NormalizedExample

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass
class ValidationReport:
    examined: int = 0
    valid: int = 0
    exclusions: Counter[str] = field(default_factory=Counter)
    warnings: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def exclude(self, reason: str, example_id: str) -> None:
        self.exclusions[reason] += 1
        self.examples.setdefault(reason, [])
        if len(self.examples[reason]) < 20:
            self.examples[reason].append(example_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "examined": self.examined,
            "valid": self.valid,
            "excluded": sum(self.exclusions.values()),
            "exclusions": dict(self.exclusions),
            "warnings": dict(self.warnings),
            "example_ids": self.examples,
        }


def validate_grounding_coordinates(value: str) -> bool:
    numbers = [float(match.group()) for match in _NUMBER.finditer(value)]
    return len(numbers) >= 4 and all(math.isfinite(number) for number in numbers)


def validate_example(
    example: NormalizedExample,
    report: ValidationReport,
    *,
    max_text_chars: int = 100_000,
    max_image_dimension: int = 20_000,
) -> bool:
    report.examined += 1
    reasons: list[str] = []
    if not example.user_prompt.strip():
        reasons.append("empty_question")
    if not example.assistant_response.strip():
        reasons.append("empty_response")
    if not example.images:
        reasons.append("missing_image")
    if len(example.user_prompt) > max_text_chars or len(example.assistant_response) > max_text_chars:
        reasons.append("extreme_text_length")
    for asset in example.images:
        try:
            image = asset.open()
            if image.width <= 0 or image.height <= 0:
                reasons.append("zero_image_dimension")
            if image.width > max_image_dimension or image.height > max_image_dimension:
                reasons.append("extreme_image_dimension")
        except ValueError:
            reasons.append("unreadable_image")
    if example.task_type and "ground" in example.task_type.lower():
        if not validate_grounding_coordinates(example.assistant_response):
            reasons.append("malformed_grounding_coordinates")
    for reason in sorted(set(reasons)):
        report.exclude(reason, example.example_id)
    if not reasons:
        report.valid += 1
        return True
    return False


def duplicate_qa_key(example: NormalizedExample) -> tuple[str, str, str]:
    return (
        example.image_id,
        " ".join(example.user_prompt.casefold().split()),
        " ".join(example.assistant_response.casefold().split()),
    )
