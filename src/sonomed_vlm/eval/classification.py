"""Robust but auditable multiple-choice parsing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_LEADING = re.compile(r"^\s*(?:the\s+answer\s+is\s+)?\(?([A-Z])\b\)?(?:\s*[:.\-)])?\s*", re.I)
_EXPLICIT_LABEL = re.compile(
    r"\b(?:the\s+)?(?:correct\s+)?(?:answer|option|choice|label)\s*"
    r"(?:is|:)\s*(?:\*+\s*)?\(?([A-Z])\b",
    re.I,
)


@dataclass(frozen=True)
class ParsedChoice:
    label: str | None
    leading_label: str | None
    text_label: str | None
    valid: bool
    method: str


def _options_map(options: list[str]) -> dict[str, str]:
    output = {}
    for index, option in enumerate(options):
        match = re.match(r"\s*([A-Z])\s*[:.\-)]+\s*(.+)", option, re.I)
        if match:
            output[match.group(1).upper()] = match.group(2).strip()
        elif index < 26:
            output[chr(ord("A") + index)] = option.strip()
    return output


def _normalize_option_text(value: str) -> str:
    value = re.sub(r"^\s*(?:the\s+answer\s+is\s+)", "", value, flags=re.I)
    characters = [
        " " if unicodedata.category(character).startswith("P") else character for character in value
    ]
    return " ".join("".join(characters).casefold().split())


def _semantic_text_label(output: str, option_map: dict[str, str]) -> str | None:
    normalized_output = _normalize_option_text(output)
    matches = {
        label: normalized
        for label, text in option_map.items()
        if (normalized := _normalize_option_text(text))
        and f" {normalized} " in f" {normalized_output} "
    }
    maximal_matches = {
        label
        for label, normalized in matches.items()
        if not any(
            label != other_label and f" {normalized} " in f" {other_normalized} "
            for other_label, other_normalized in matches.items()
        )
    }
    return next(iter(maximal_matches)) if len(maximal_matches) == 1 else None


def _explicit_label(output: str, option_map: dict[str, str]) -> str | None:
    match = _EXPLICIT_LABEL.search(output)
    label = match.group(1).upper() if match else None
    return label if label in option_map else None


def parse_multiple_choice(output: str, options: list[str]) -> ParsedChoice:
    option_map = _options_map(options)
    match = _LEADING.match(output)
    leading_label = match.group(1).upper() if match else None
    if leading_label is not None and option_map and leading_label not in option_map:
        leading_label = None
    answer_text = output[match.end() :] if match and leading_label is not None else output
    normalized = _normalize_option_text(answer_text)
    text_matches = [
        label
        for label, text in option_map.items()
        if normalized and normalized == _normalize_option_text(text)
    ]
    text_label = text_matches[0] if len(text_matches) == 1 else None
    label = leading_label or text_label
    if leading_label is not None and text_label is not None:
        method = "leading_label_and_option_text"
    elif leading_label is not None:
        method = "leading_label"
    elif text_label is not None:
        method = "exact_option_text"
    else:
        method = "unparsed"
    return ParsedChoice(label, leading_label, text_label, label is not None, method)


def mcq_score(
    output: str, answer_label: str | None, options: list[str]
) -> dict[str, float | str | None]:
    parsed = parse_multiple_choice(output, options)
    option_map = _options_map(options)
    gold = answer_label.strip().upper() if answer_label else None
    strict_label_accuracy = float(
        parsed.leading_label is not None and gold is not None and parsed.leading_label == gold
    )
    option_text_accuracy = float(
        parsed.text_label is not None and gold is not None and parsed.text_label == gold
    )
    has_both = parsed.leading_label is not None and parsed.text_label is not None
    contradiction = has_both and parsed.leading_label != parsed.text_label
    consistency = has_both and parsed.leading_label == parsed.text_label
    semantic_text_label = parsed.text_label or _semantic_text_label(output, option_map)
    semantic_fallback_label = parsed.leading_label or _explicit_label(output, option_map)
    semantic_label = semantic_text_label or semantic_fallback_label
    if semantic_text_label is not None:
        semantic_method = "option_text"
    elif semantic_fallback_label is not None:
        semantic_method = "option_label"
    else:
        semantic_method = "unresolved"
    semantic_choice_accuracy = float(
        semantic_label is not None and gold is not None and semantic_label == gold
    )
    return {
        "parsed_prediction": parsed.label,
        "parsed_label": parsed.leading_label,
        "parsed_option_text_label": parsed.text_label,
        "parsed_semantic_option_text_label": semantic_text_label,
        "parsed_semantic_label": semantic_fallback_label,
        "parsed_semantic_choice": semantic_label,
        "semantic_resolution_method": semantic_method,
        "parse_method": parsed.method,
        "strict_label_accuracy": strict_label_accuracy,
        "option_text_accuracy": option_text_accuracy,
        "semantic_choice_accuracy": semantic_choice_accuracy,
        "label_text_consistency": float(consistency),
        "contradiction_rate": float(contradiction),
        "valid_response_rate": float(parsed.valid),
        # Retain the original keys for existing reports while corrected artifacts
        # expose the complementary metrics above.
        "valid": float(parsed.valid),
        "invalid_response_rate": float(not parsed.valid),
        "accuracy": strict_label_accuracy,
    }
