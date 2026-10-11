"""Dependency-light lexical generation metrics."""

from __future__ import annotations

import re
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize_text(value: str) -> str:
    return " ".join(_TOKEN.findall(value.casefold()))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_text(prediction).split()
    gold = normalize_text(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = normalize_text(prediction).split()
    gold = normalize_text(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    lcs = _lcs_length(predicted, gold)
    precision, recall = lcs / len(predicted), lcs / len(gold)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0
