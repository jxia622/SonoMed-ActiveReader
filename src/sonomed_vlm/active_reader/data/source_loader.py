"""Independent CVA-Net COCO reader, including unlisted frames on disk."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from sonomed_vlm.active_reader.data.corrections import COUNTERPART, EXCLUDED, SHARED
from sonomed_vlm.active_reader.data.manifest import Case, Frame, source_path, xywh_to_xyxy
from sonomed_vlm.active_reader.io import file_hash

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(value):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def perceptual_hash(image):
    small = image.convert("L").resize((9, 8))
    pixels = list(small.tobytes())
    bits = sum(
        int(pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8)
        for x in range(8)
    )
    return f"{bits:016x}"


def discover_annotations(root, files):
    candidates = [root / p for p in files] if files else sorted(root.glob("*.json"))
    found = []
    for path in candidates:
        value = json.loads(path.read_text())
        if (
            isinstance(value, dict)
            and isinstance(value.get("images"), list)
            and isinstance(value.get("annotations"), list)
        ):
            found.append((path, value))
        elif files:
            raise ValueError(f"Annotation file is not COCO: {path.name}")
    if not found:
        raise ValueError(
            "No COCO annotations found. Set data.root to miccai_buv or data.annotation_files explicitly."
        )
    return found


def load_source(root: Path, annotation_files=()):
    if not root.is_dir():
        raise ValueError("BREASTVID_ROOT must point to an existing extracted dataset directory")
    root = root.resolve()
    raw = root / "rawframes" if (root / "rawframes").is_dir() else root
    records, splits, issues = {}, defaultdict(set), []
    sources = discover_annotations(root, annotation_files)
    for path, document in sources:
        videos = {v["id"]: v for v in document.get("videos", [])}
        categories = {c["id"]: str(c["name"]).lower() for c in document.get("categories", [])}
        annotations = defaultdict(list)
        image_ids = [row["id"] for row in document["images"]]
        image_id_set = set(image_ids)
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("Duplicate COCO image IDs; select unambiguous annotation files")
        for annotation in document["annotations"]:
            if annotation["image_id"] not in image_id_set:
                issues.append({"kind": "orphan_annotation", "file": path.name, "severity": "error"})
            annotations[annotation["image_id"]].append(annotation)
        for row in document["images"]:
            name = row["file_name"].replace("\\", "/")
            location = source_path(root, name)
            if not location.is_file():
                location = source_path(raw, name)
            if not location.is_file():
                issues.append({"kind": "missing_image", "frame": name, "severity": "error"})
                continue
            key = str(location.relative_to(root))
            ann = annotations[row["id"]]
            value = {
                "width": row.get("width"),
                "height": row.get("height"),
                "patient_id": row.get(
                    "patient_id", videos.get(row.get("video_id"), {}).get("patient_id")
                ),
                "empty": row.get("annotation_status") == "empty"
                or row.get("annotated_empty") is True,
                "boxes": sorted([a.get("bbox", []) for a in ann]),
                "labels": sorted({categories.get(a.get("category_id"), "unknown") for a in ann}),
            }
            if key in records and records[key] != value:
                raise ValueError(
                    "Conflicting annotation variants for the same frame; select data.annotation_files explicitly"
                )
            records[key] = value
            splits[key].add(path.stem)
    grouped = defaultdict(list)
    for location in sorted(raw.rglob("*")):
        if location.is_file() and location.suffix.lower() in IMAGE_SUFFIXES:
            source_path(root, str(location.relative_to(root)))
            grouped[str(location.parent.relative_to(raw))].append(location)
    cases = []
    for source, paths in sorted(grouped.items()):
        components = source.lower().split("/")
        labels = {x for x in components if x in ("benign", "malignant")}
        source_id = source.split("/")[-1]
        known_correction = source_id in SHARED | {COUNTERPART, EXCLUDED}
        all_annotation_labels = {
            label
            for p in paths
            for label in records.get(str(p.relative_to(root)), {}).get("labels", [])
            if label in ("benign", "malignant")
        }
        if len(labels) != 1:
            labels = all_annotation_labels
        if len(labels) != 1:
            issues.append({"kind": "unresolved_case_label", "source": source, "severity": "error"})
            label = "unresolved"
        else:
            label = next(iter(labels))
        if all_annotation_labels - {label} and not known_correction:
            issues.append({"kind": "label_disagreement", "source": source, "severity": "error"})
        frames, patients, original_splits = [], set(), set()
        ordered = sorted(paths, key=lambda p: natural_key(p.name))
        if len({p.stem for p in ordered}) != len(ordered):
            issues.append({"kind": "ambiguous_frame_ids", "source": source, "severity": "error"})
        for location in ordered:
            key = str(location.relative_to(root))
            meta = records.get(key, {})
            if meta.get("patient_id") is not None:
                patients.add(str(meta["patient_id"]))
            original_splits.update(splits[key])
            try:
                with Image.open(location) as image:
                    image.load()
                    clean = image.convert("RGB")
                    width, height = image.size
                    pixel_hash = hashlib.sha256(
                        f"{width}x{height}:".encode() + clean.tobytes()
                    ).hexdigest()
                    dhash = perceptual_hash(clean)
                if meta.get("width") not in (None, width) or meta.get("height") not in (
                    None,
                    height,
                ):
                    issues.append({"kind": "dimension_mismatch", "frame": key, "severity": "error"})
                boxes, invalid_boxes = [], []
                for box in meta.get("boxes", []):
                    try:
                        boxes.append(xywh_to_xyxy(box, width, height))
                    except (ValueError, TypeError):
                        invalid_boxes.append(box)
                if invalid_boxes:
                    # Preserve every source frame and its index. Do not infer intended
                    # geometry from negative widths or coordinates outside the image.
                    issues.append(
                        {
                            "kind": "invalid_box_geometry",
                            "frame": key,
                            "source_boxes_xywh": invalid_boxes,
                            "severity": "warning",
                            "resolution": "retain_pixels_mark_frame_unannotated_disable_case_relevance",
                        }
                    )
                    boxes = []
                if boxes and meta.get("empty"):
                    raise ValueError("Contradictory explicit empty annotation")
                status = (
                    "unannotated"
                    if invalid_boxes
                    else "present"
                    if boxes
                    else "empty"
                    if meta.get("empty")
                    else "unannotated"
                )
                frames.append(
                    Frame(
                        len(frames),
                        location.stem,
                        key,
                        width,
                        height,
                        status,
                        boxes,
                        file_hash(location),
                        pixel_hash,
                        dhash,
                        meta.get("labels", []),
                    )
                )
            except (OSError, ValueError) as exc:
                issues.append(
                    {
                        "kind": "corrupt_image_or_invalid_box",
                        "frame": key,
                        "detail": str(exc),
                        "severity": "error",
                    }
                )
        if len(patients) > 1:
            issues.append(
                {"kind": "conflicting_patient_ids", "source": source, "severity": "error"}
            )
        case_id = "case_" + hashlib.sha256(source.encode()).hexdigest()[:16]
        patient = next(iter(patients)) if len(patients) == 1 else None
        cases.append(
            Case(
                case_id,
                source,
                label,
                frames,
                patient,
                "patient_" + hashlib.sha256(patient.encode()).hexdigest()[:16]
                if patient
                else case_id,
                "patient_id" if patient else "sequence_only",
                sorted(original_splits),
            )
        )
    if not cases:
        raise ValueError("No source images found under dataset root/rawframes")
    return cases, {
        "sources": [{"file": str(p.relative_to(root)), "sha256": file_hash(p)} for p, _ in sources],
        "issues": issues,
        "source_sequences": len(cases),
        "source_frames": sum(len(p) for p in grouped.values()),
    }
