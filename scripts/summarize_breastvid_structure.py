#!/usr/bin/env python3
"""Supplement coverage bounds and regenerate distinct representative trajectories."""

import math
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sonomed_vlm.active_reader.data.manifest import xywh_to_xyxy
from sonomed_vlm.active_reader.io import file_hash, read_json, write_json
from sonomed_vlm.active_reader.suitability import allowed_cases, uniform_indices


def main():
    base = Path("artifacts/active_reader/suitability_20261008/structure")
    cases, _ = allowed_cases(
        Path("artifacts/active_reader/manifest.jsonl"), Path("artifacts/active_reader/splits.json")
    )
    rows = read_json(base / "cases.json")
    allowed = {f.relative_path: f for c in cases for f in c.frames}
    source = {}
    for path in Path("data/raw/breastvid/miccai_buv").glob("*.json"):
        doc = read_json(path)
        by = defaultdict(list)
        for a in doc["annotations"]:
            by[a["image_id"]].append(a)
        for im in doc["images"]:
            name = im["file_name"].replace("\\", "/")
            if name not in allowed:
                name = "rawframes/" + name
            if name in allowed:
                source[name] = by[im["id"]]
    reasons = Counter()
    for name, f in allowed.items():
        if f.annotation_status != "unannotated":
            continue
        if name not in source:
            reasons["source_image_record_missing"] += 1
            continue
        anns = source[name]
        if not anns:
            reasons["source_record_without_boxes"] += 1
            continue
        invalid = 0
        for ann in anns:
            try:
                xywh_to_xyxy(ann["bbox"], f.width, f.height)
            except (ValueError, TypeError):
                invalid += 1
        reasons["invalid_geometry" if invalid else "other_unknown"] += 1
    counts = Counter(f.annotation_status for c in cases for f in c.frames)
    capture = {}
    for k in (1, 4, 8, 12):
        uniform = []
        random = []
        for c in cases:
            present = sum(f.annotation_status == "present" for f in c.frames)
            n = c.num_frames
            uniform.append(
                any(c.frames[i].annotation_status == "present" for i in uniform_indices(n, k))
            )
            random.append(
                1 - math.comb(n - present, k) / math.comb(n, k) if n - present >= k else 1.0
            )
        capture[str(k)] = {
            "uniform_cases_with_a_valid_box": sum(uniform),
            "cases": len(cases),
            "random_probability_any_valid_box_case_mean": float(np.mean(random)),
        }
    distant = {
        metric: float(
            np.median(
                [
                    r["temporal"][str(max(1, round(0.5 * (r["num_frames"] - 1))))][metric]["p50"]
                    for r in rows
                ]
            )
        )
        for metric in ("full_ssim", "central_ssim", "embedding_cosine")
    }
    all_area = [a for r in rows for a in r["normalized_union_area"] if a is not None]
    result = {
        "test_opened": False,
        "unknown_annotation_reasons": dict(reasons),
        "explicitly_annotated_empty_frames": counts["empty"],
        "cases_no_valid_box": sum(r["annotation_present_fraction"] == 0 for r in rows),
        "case_coverage_fraction_quantiles": dict(
            zip(
                ("min", "p10", "p25", "median", "p75", "p90", "max"),
                np.quantile(
                    [r["annotation_present_fraction"] for r in rows],
                    [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
                ).tolist(),
                strict=True,
            )
        ),
        "normalized_union_area_known_frames_quantiles": dict(
            zip(
                ("min", "p10", "p25", "median", "p75", "p90", "max"),
                np.quantile(all_area, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist(),
                strict=True,
            )
        ),
        "median_per_video_area_cv": float(
            np.median([r["area_cv"] for r in rows if r["area_cv"] is not None])
        ),
        "within_video_exact_duplicate_frames": sum(
            r["num_frames"] - r["exact_unique_pixel_frames"] for r in rows
        ),
        "half_video_lag_median_of_case_medians": distant,
        "annotated_lesion_capture": capture,
        "source_script_sha256": file_hash(Path(__file__)),
    }
    # Select six distinct examples without using diagnoses or reader outcomes.
    used = set()
    selections = []
    rules = [
        ("Shortest", lambda r: r["num_frames"], False),
        (
            "Median length",
            lambda r: abs(r["num_frames"] - np.median([x["num_frames"] for x in rows])),
            False,
        ),
        ("Longest", lambda r: r["num_frames"], True),
        ("Most partial annotation gaps", lambda r: r["annotation_present_fraction"], False),
        ("Largest relative area variation", lambda r: r["area_cv"] or 0, True),
        ("Lowest adjacent-frame SSIM", lambda r: r["temporal"]["1"]["central_ssim"]["p50"], False),
    ]
    for title, key, reverse in rules:
        row = next(
            r
            for r in sorted(rows, key=key, reverse=reverse)
            if r["video_id"] not in used
            and (
                title != "Most partial annotation gaps" or 0 < r["annotation_present_fraction"] < 1
            )
        )
        selections.append((title, row))
        used.add(row["video_id"])
    result["representatives"] = [
        {"rule": title, "alias": r["alias"], "video_id": r["video_id"]} for title, r in selections
    ]
    write_json(base / "supplemental_summary.json", result)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, (title, r) in zip(axes.flat, selections, strict=True):
        a = np.array([np.nan if v is None else 100 * v for v in r["normalized_union_area"]])
        x = np.linspace(0, 1, len(a))
        if np.isnan(a).all():
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "No valid boxes\nVisibility unknown",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.plot(x, a, color="#2678a6", lw=1.3)
            ax.set_ylim(0, np.nanmax(a) * 1.1)
        for i in np.flatnonzero(np.isnan(a)):
            ax.axvspan(
                max(0, (i - 0.5) / (len(a) - 1)),
                min(1, (i + 0.5) / (len(a) - 1)),
                color="#bdbdbd",
                alpha=0.45,
                lw=0,
            )
        ax.set(xlabel="Relative frame position", ylabel="Box union / image area (%)", xlim=(0, 1))
        ax.set_title(
            f"{title}: {r['alias']} / {r['num_frames']} frames\nValid box on {r['annotation_present_fraction']:.1%} of frames",
            fontsize=10,
        )
    fig.suptitle("Lesion annotation trajectories; gray = unknown, not lesion-free")
    fig.tight_layout()
    fig.savefig("reports/active_reader/suitability_figures/area_trajectories.png", dpi=180)
    plt.close(fig)
    print(result)


if __name__ == "__main__":
    main()
