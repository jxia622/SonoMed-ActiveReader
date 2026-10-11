"""Private audit, contamination checks, and human-only annotation sheets."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sonomed_vlm.active_reader.data.corrections import COUNTERPART, EXCLUDED, apply_corrections
from sonomed_vlm.active_reader.data.manifest import fingerprint, source_path
from sonomed_vlm.active_reader.data.source_loader import load_source
from sonomed_vlm.active_reader.data.splits import connect_groups
from sonomed_vlm.active_reader.io import file_hash, write_json


def quarantine_conflicts(cases, report):
    """Do not guess labels for sequences with contradictory exact visual evidence."""
    if report.get("quarantine") and report.get("manifest_fingerprint") == fingerprint(cases):
        return cases, report
    by_id = {c.video_id: c for c in cases}
    conflicts = []
    excluded_groups = set()
    for a, b in report["exact_overlap_pairs"]:
        left, right = by_id[a], by_id[b]
        if left.label != right.label:
            overlap = {f.pixel_sha256 for f in left.frames} & {f.pixel_sha256 for f in right.frames}
            conflicts.append(
                {
                    "cases": [a, b],
                    "sources": [left.source_relative_path, right.source_relative_path],
                    "labels": [left.label, right.label],
                    "shared_pixel_hashes": len(overlap),
                    "resolution": "quarantine_entire_connected_group_no_label_inference",
                }
            )
            excluded_groups.update([left.group_id, right.group_id])
    quarantined = [c for c in cases if c.group_id in excluded_groups]
    retained = [c for c in cases if c.group_id not in excluded_groups]
    report["author_corrected_sequences"] = len(cases)
    report["author_corrected_frames"] = sum(c.num_frames for c in cases)
    report["label_conflicts"] = conflicts
    report["quarantine"] = {
        "version": "exact-cross-label-overlap-v1",
        "cases": [c.video_id for c in quarantined],
        "sequences": len(quarantined),
        "frames": sum(c.num_frames for c in quarantined),
    }
    report["corrected_sequences"] = len(retained)
    report["corrected_frames"] = sum(c.num_frames for c in retained)
    report["class_balance"] = dict(Counter(c.label for c in retained))
    report["annotation_status"] = dict(
        Counter(f.annotation_status for c in retained for f in c.frames)
    )
    report["dimensions"] = dict(
        Counter(f"{f.width}x{f.height}" for c in retained for f in c.frames)
    )
    excluded_ids = {c.video_id for c in quarantined}
    if "per_case" in report:
        report["quarantined_case_statistics"] = [
            r for r in report["per_case"] if r["video_id"] in excluded_ids
        ]
        report["per_case"] = [r for r in report["per_case"] if r["video_id"] not in excluded_ids]
    report["sequence_length_quantiles"] = np.quantile(
        [c.num_frames for c in retained], [0, 0.25, 0.5, 0.75, 1]
    ).tolist()
    report["manifest_fingerprint"] = fingerprint(retained)
    return retained, report


def contamination(cases, sft_manifests):
    hashes = {f.sha256 for c in cases for f in c.frames} | {
        f.pixel_sha256 for c in cases for f in c.frames
    }
    sources, matches, provenance_mentions = [], [], []
    for path in sft_manifests:
        if not path.is_file():
            raise ValueError("Configured SFT manifest is missing")
        text = path.read_text()
        sources.append({"path": str(path), "sha256": file_hash(path)})
        for line_index, line in enumerate(text.splitlines()):
            if any(word in line.lower() for word in ("breastvid", "miccai_buv", "cva-net")):
                provenance_mentions.append({"manifest": str(path), "line": line_index + 1})
            # Existing SonoInstruct manifests may contain source byte hashes nested in records.
            for value in set(__import__("re").findall(r"\b[a-f0-9]{64}\b", line)):
                if value in hashes:
                    matches.append({"manifest": str(path), "line": line_index + 1, "hash": value})
    return {
        "status": "potentially_contaminated" if matches or provenance_mentions else "unknown",
        "sources": sources,
        "exact_matches": matches,
        "provenance_mentions": provenance_mentions,
        "limitations": "Absence of hash matches does not exclude resized images, derived crops, or undocumented sources. Perceptual SFT-image/crop audit still requires source images.",
    }


def audit_data(root, annotation_files=(), sft_manifests=()):
    source, metadata = load_source(root, annotation_files)
    cases, corrections = apply_corrections(source)
    by_source = {c.source_relative_path.split("/")[-1]: c for c in source}
    a, b = by_source.get(EXCLUDED), by_source.get(COUNTERPART)
    corrections["duplicate_relationship"] = (
        "pixel_confirmed"
        if a and b and [f.pixel_sha256 for f in a.frames] == [f.pixel_sha256 for f in b.frames]
        else "author_confirmed; local pair unavailable or unequal"
    )
    exact = connect_groups(cases)
    near = []
    for i, left in enumerate(cases):
        lh = [int(f.dhash, 16) for f in left.frames[:: max(1, len(left.frames) // 8)] if f.dhash]
        for right in cases[i + 1 :]:
            if left.group_id == right.group_id:
                continue
            rh = [
                int(f.dhash, 16) for f in right.frames[:: max(1, len(right.frames) // 8)] if f.dhash
            ]
            hits = sum(any((x ^ y).bit_count() <= 4 for y in rh) for x in lh)
            if hits >= 2:
                near.append(
                    {
                        "cases": [left.video_id, right.video_id],
                        "sampled_dhash_matches": hits,
                        "status": "suspected_only_requires_review",
                    }
                )
    per_case = []
    for case in cases:
        areas = [
            ((b[2] - b[0]) * (b[3] - b[1])) / (f.width * f.height)
            for f in case.frames
            for b in f.boxes_xyxy
        ]
        per_case.append(
            {
                "video_id": case.video_id,
                "frames": len(case.frames),
                "annotation_status": dict(Counter(f.annotation_status for f in case.frames)),
                "area_min": min(areas) if areas else None,
                "area_max": max(areas) if areas else None,
                "area_std": float(np.std(areas)) if areas else None,
                "multiple_boxes": any(len(f.boxes_xyxy) > 1 for f in case.frames),
                "non_numeric_frame_ids": any(not f.source_frame_id.isdigit() for f in case.frames),
            }
        )
    errors = [i for i in metadata["issues"] if i["severity"] == "error"]
    report = {
        "schema_version": 1,
        **metadata,
        "manifest_fingerprint": fingerprint(cases),
        "invalid_box_policy": "retain all source images; invalid geometry makes frame unannotated and disables relevance for its case; no clipping or coordinate inversion",
        "corrections": corrections,
        "corrected_sequences": len(cases),
        "corrected_frames": sum(c.num_frames for c in cases),
        "class_balance": dict(Counter(c.label for c in cases)),
        "sequence_length_quantiles": np.quantile(
            [c.num_frames for c in cases], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "dimensions": dict(Counter(f"{f.width}x{f.height}" for c in cases for f in c.frames)),
        "annotation_status": dict(Counter(f.annotation_status for c in cases for f in c.frames)),
        "exact_overlap_pairs": exact,
        "near_duplicate_candidates": near,
        "original_cross_split_overlap": [
            pair
            for pair in exact
            if len({s for c in cases if c.video_id in pair for s in c.original_split}) > 1
        ],
        "patient_ids_available": sum(c.patient_id is not None for c in cases),
        "per_case": per_case,
        "contamination": contamination(cases, sft_manifests),
        "automatic_checks_passed": not errors,
        "visual_shortcut_review": "pending: inspect contact sheets for identifiers, class text, measurements; no masks applied",
        "frame_selection_headroom": "unmeasured; requires frozen Qwen3 baselines",
        "research_scope": "prerecorded short-sequence benchmark, no clinical validation",
    }
    return quarantine_conflicts(cases, report)


def write_audit(path, report):
    write_json(path, report)
    markdown = path.with_suffix(".md")
    markdown.write_text(
        "# BreastVid data audit\n\nPrivate audit; annotated sheets must never enter model inputs.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    markdown.chmod(0o600)


def contact_sheet(case, root: Path, output: Path, indices=None):
    indices = (
        indices
        if indices is not None
        else sorted(
            set(np.linspace(0, case.num_frames - 1, min(12, case.num_frames)).round().astype(int))
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (4 * 256, ((len(indices) + 3) // 4) * 220), "white")
    for slot, index in enumerate(indices):
        frame = case.frames[index]
        with Image.open(source_path(root, frame.relative_path)) as image:
            tile = image.convert("RGB")
        draw = ImageDraw.Draw(tile)
        for box in frame.boxes_xyxy:
            draw.rectangle(box, outline="red", width=3)
        tile.thumbnail((256, 190))
        x, y = (slot % 4) * 256, (slot // 4) * 220
        sheet.paste(tile, (x, y))
        ImageDraw.Draw(sheet).text(
            (x + 4, y + 194),
            f"AUDIT ONLY | frame {index} | {frame.annotation_status}",
            fill="black",
        )
    sheet.save(output)
    output.chmod(0o600)
