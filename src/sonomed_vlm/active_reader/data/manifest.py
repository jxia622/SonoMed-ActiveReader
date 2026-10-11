"""Evaluator-private manifests. Never serialize these objects into a prompt."""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from sonomed_vlm.active_reader.io import digest


@dataclass
class Frame:
    frame_index: int
    source_frame_id: str
    relative_path: str
    width: int
    height: int
    annotation_status: Literal["present", "empty", "unannotated"]
    boxes_xyxy: list[list[float]] = field(default_factory=list)
    sha256: str | None = None
    pixel_sha256: str | None = None
    dhash: str | None = None
    annotation_labels: list[str] = field(default_factory=list)


@dataclass
class Case:
    video_id: str
    source_relative_path: str
    label: str
    frames: list[Frame]
    patient_id: str | None = None
    group_id: str = ""
    group_basis: str = "sequence_only"
    original_split: list[str] = field(default_factory=list)
    split: str | None = None
    label_provenance: str = "source_path_and_annotations"
    applied_corrections: list[str] = field(default_factory=list)

    @property
    def num_frames(self):
        return len(self.frames)

    def validate(self):
        if self.label not in ("benign", "malignant") or not self.frames or not self.group_id:
            raise ValueError("Case has unresolved label, frames, or group")
        for i, frame in enumerate(self.frames):
            if frame.frame_index != i or frame.width < 1 or frame.height < 1:
                raise ValueError("Invalid frame ordering or geometry")
            if frame.annotation_status not in ("present", "empty", "unannotated"):
                raise ValueError("Unknown annotation status")
            if bool(frame.boxes_xyxy) != (frame.annotation_status == "present"):
                raise ValueError("Annotation status and boxes disagree")
            for box in frame.boxes_xyxy:
                validate_box(box, frame.width, frame.height)


def validate_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError("Non-finite or malformed box")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box outside source geometry or non-positive area")


def xywh_to_xyxy(box, width, height):
    if len(box) != 4:
        raise ValueError("Expected COCO xywh box")
    x, y, w, h = map(float, box)
    converted = [x, y, x + w, y + h]
    validate_box(converted, width, height)
    return converted


def resize_box(box, source_size, destination_size):
    validate_box(box, *source_size)
    sx, sy = destination_size[0] / source_size[0], destination_size[1] / source_size[1]
    result = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    validate_box(result, *destination_size)
    return result


def source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source frame escapes dataset root")
    return path


def fingerprint(cases):
    return digest([asdict(c) for c in cases])


def save_manifest(path: Path, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        for case in cases:
            case.validate()
            stream.write(json.dumps(asdict(case), sort_keys=True) + "\n")
    temporary.replace(path)


def load_manifest(path: Path):
    cases = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row["frames"] = [Frame(**f) for f in row["frames"]]
        case = Case(**row)
        case.validate()
        cases.append(case)
    if len({c.video_id for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs in manifest")
    return cases
