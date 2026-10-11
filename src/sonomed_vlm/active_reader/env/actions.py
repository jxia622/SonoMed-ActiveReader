"""Strict inspect/stop protocol and annotation-independent sampling."""

import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Inspect:
    center_bin: int
    scale: str
    action: str = "inspect"


@dataclass(frozen=True)
class Stop:
    prediction: str
    evidence_frames: tuple[int, ...]
    action: str = "stop"


def parse_action(raw, max_evidence=3):
    try:
        value = (
            json.loads(raw)
            if isinstance(raw, str)
            else asdict(raw)
            if isinstance(raw, (Inspect, Stop))
            else raw
        )
        if not isinstance(value, dict):
            raise ValueError
        if value.get("action") == "inspect" and set(value) == {"action", "center_bin", "scale"}:
            if (
                type(value["center_bin"]) is int
                and 0 <= value["center_bin"] < 16
                and value["scale"] in ("local", "coarse")
            ):
                return Inspect(value["center_bin"], value["scale"])
        if value.get("action") == "stop" and set(value) == {
            "action",
            "prediction",
            "evidence_frames",
        }:
            refs = value["evidence_frames"]
            if (
                value["prediction"] in ("benign", "malignant")
                and isinstance(refs, (list, tuple))
                and 1 <= len(refs) <= max_evidence
                and all(type(x) is int and x >= 0 for x in refs)
                and len(set(refs)) == len(refs)
            ):
                return Stop(value["prediction"], tuple(refs))
    except (ValueError, TypeError):
        pass
    raise ValueError("Invalid action. Return exactly one legal inspect or stop JSON object.")


def evenly_spaced(start, end, count):
    count = min(count, end - start + 1)
    if count <= 0:
        return []
    if count == 1:
        return [start]
    return sorted({round(start + i * (end - start) / (count - 1)) for i in range(count)})


def window_indices(n, center_bin, scale):
    if (
        n < 1
        or type(center_bin) is not int
        or not 0 <= center_bin < 16
        or scale not in ("local", "coarse")
    ):
        raise ValueError("Invalid inspection coordinates")
    center = round(center_bin * (n - 1) / 15)
    width = min(n, 4 if scale == "local" else max(4, round(0.2 * n)))
    start = min(max(center - math.floor((width - 1) / 2), 0), n - width)
    return evenly_spaced(start, start + width - 1, 4)
