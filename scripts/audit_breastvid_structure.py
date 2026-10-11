#!/usr/bin/env python3
"""Train/dev length, annotation coverage, SSIM, and cached ResNet18 feature audit."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from sonomed_vlm.active_reader.data.manifest import source_path
from sonomed_vlm.active_reader.io import digest, file_hash, read_json, write_json
from sonomed_vlm.active_reader.suitability import (
    allowed_cases,
    require_allowed,
    uniform_indices,
    union_area,
)


def stats(values):
    x = np.asarray(values, dtype=float)
    return {
        "n": len(x),
        "min": float(x.min()),
        "mean": float(x.mean()),
        **{f"p{p}": float(np.percentile(x, p)) for p in (5, 10, 25, 50, 75, 90, 95)},
        "max": float(x.max()),
    }


def ssim_pairs(images, lag):
    """Mean 7x7 uniform-window SSIM, unit range, population moments, no padding."""
    a = torch.from_numpy(images[:-lag, None].astype(np.float32) / 255)
    b = torch.from_numpy(images[lag:, None].astype(np.float32) / 255)

    def pool(x):
        return F.avg_pool2d(x, 7, stride=1)

    ma, mb = pool(a), pool(b)
    va, vb = (pool(a * a) - ma * ma).clamp_min(0), (pool(b * b) - mb * mb).clamp_min(0)
    cov = pool(a * b) - ma * mb
    value = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return value.mean((1, 2, 3)).numpy()


def greedy_cover(sim, threshold):
    reps = []
    for i in range(len(sim)):
        if not reps or max(sim[i, reps]) < threshold:
            reps.append(i)
    return reps


def source_metadata(root, cases):
    allowed = {f.relative_path for c in cases for f in c.frames}
    counts = Counter()
    schemas = defaultdict(set)
    timing = []
    for path in sorted(root.glob("*.json")):
        document = read_json(path)
        image_ids, video_ids = set(), set()
        for im in document.get("images", []):
            name = im["file_name"].replace("\\", "/")
            if name not in allowed and "rawframes/" + name not in allowed:
                continue
            image_ids.add(im["id"])
            video_ids.add(im["video_id"])
            schemas["images"].update(im)
            for k, v in im.items():
                if any(t in k.lower() for t in ("fps", "timestamp", "duration", "frame_rate")):
                    timing.append({"key": k, "value": v})
        for v in document.get("videos", []):
            if v["id"] in video_ids:
                schemas["videos"].update(v)
                for k, value in v.items():
                    if any(t in k.lower() for t in ("fps", "timestamp", "duration", "frame_rate")):
                        timing.append({"key": k, "value": value})
        for a in document.get("annotations", []):
            if a["image_id"] in image_ids:
                counts["raw_annotations"] += 1
                counts[f"occluded={a.get('occluded', 'missing')}"] += 1
                counts[f"generated={a.get('generated', 'missing')}"] += 1
    return {
        "schemas": {k: sorted(v) for k, v in schemas.items()},
        "timing_fields": timing,
        "annotation_flags": dict(counts),
        "duration_seconds": None,
        "duration_note": "No assumed FPS; extracted frame count is not a measured duration.",
    }


def run(args):
    torch.set_num_threads(4)
    cases, inventory = allowed_cases(args.manifest, args.splits)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval().to(device)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "embeddings").mkdir(exist_ok=True)
    (args.output / "cases").mkdir(exist_ok=True)
    provenance = {
        "manifest_sha256": file_hash(args.manifest),
        "splits_sha256": file_hash(args.splits),
        "script_sha256": file_hash(Path(__file__)),
        "inventory": inventory,
        "audited": len(cases),
        "test_opened": False,
        "embedding": "torchvision ResNet18 IMAGENET1K_V1 512d avgpool; standard resize256/center-crop224",
        "weights_sha256": file_hash(
            Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        ),
        "similarity": "128x128 grayscale, 7x7 uniform SSIM; whole frame and central 80% crop; normalized avgpool cosine",
    }
    meta = args.output / "provenance.json"
    if meta.exists() and read_json(meta) != provenance:
        raise ValueError("Structure audit changed; choose new output")
    write_json(meta, provenance)
    rows = []
    for number, case in enumerate(cases, 1):
        require_allowed(case)
        key = digest(case.video_id)
        path = args.output / "cases" / f"{key}.json"
        if path.exists():
            rows.append(read_json(path))
            continue
        gray, central, features, tensor_batch, info_keys = [], [], [], [], set()
        with torch.inference_mode():
            for i, frame in enumerate(case.frames):
                with Image.open(source_path(args.data_root, frame.relative_path)) as src:
                    im = src.convert("RGB")
                    info_keys.update(src.info)
                gray.append(
                    np.asarray(im.convert("L").resize((128, 128), Image.Resampling.BILINEAR))
                )
                w, h = im.size
                crop = im.crop((round(w * 0.1), round(h * 0.1), round(w * 0.9), round(h * 0.9)))
                central.append(
                    np.asarray(crop.convert("L").resize((128, 128), Image.Resampling.BILINEAR))
                )
                tensor_batch.append(transform(im))
                if len(tensor_batch) == 32 or i == len(case.frames) - 1:
                    features.append(
                        model(torch.stack(tensor_batch).to(device)).float().cpu().numpy()
                    )
                    tensor_batch.clear()
        features = np.concatenate(features)
        np.savez_compressed(args.output / "embeddings" / f"{key}.npz", features=features)
        normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        similarity = np.clip(normalized @ normalized.T, -1, 1)
        n = len(features)
        lags = sorted(
            {
                1,
                4,
                8,
                16,
                min(32, n - 1),
                max(1, round(0.25 * (n - 1))),
                max(1, round(0.5 * (n - 1))),
            }
        )
        temporal = {}
        for lag in lags:
            temporal[str(lag)] = {
                "full_ssim": stats(ssim_pairs(np.stack(gray), lag)),
                "central_ssim": stats(ssim_pairs(np.stack(central), lag)),
                "embedding_cosine": stats(similarity[np.arange(n - lag), np.arange(lag, n)]),
            }
        coverage = {}
        for budget in (1, 4, 8, 12):
            nearest = similarity[:, uniform_indices(n, budget)].max(1)
            coverage[str(budget)] = {
                "mean_nearest_cosine": float(nearest.mean()),
                "fraction_cosine_ge_098": float((nearest >= 0.98).mean()),
                "fraction_cosine_ge_0995": float((nearest >= 0.995).mean()),
            }
        statuses = Counter(f.annotation_status for f in case.frames)
        areas = [
            union_area(f.boxes_xyxy) / (f.width * f.height)
            if f.annotation_status == "present"
            else None
            for f in case.frames
        ]
        known = [x for x in areas if x is not None]
        dhashes = [int(f.dhash, 16) for f in case.frames]
        centered = features - features.mean(0)
        eig = np.maximum(np.linalg.eigvalsh(centered @ centered.T), 0)
        effective_rank = float(eig.sum() ** 2 / max(1e-12, (eig * eig).sum()))
        row = {
            "video_id": case.video_id,
            "group_id": case.group_id,
            "split": case.split,
            "alias": f"V{number:03}",
            "num_frames": n,
            "annotation_status": dict(statuses),
            "annotation_present_fraction": statuses["present"] / n,
            "visibility_upper_fraction": (statuses["present"] + statuses["unannotated"]) / n,
            "normalized_union_area": areas,
            "area_stats": stats(known) if known else None,
            "area_cv": float(np.std(known) / np.mean(known))
            if known and np.mean(known) > 0
            else None,
            "exact_unique_pixel_frames": len({f.pixel_sha256 for f in case.frames}),
            "adjacent_dhash": stats([(a ^ b).bit_count() for a, b in itertools.pairwise(dhashes)]),
            "temporal": temporal,
            "uniform_embedding_coverage": coverage,
            "greedy_cosine_representatives": {
                str(t): len(greedy_cover(similarity, t)) for t in (0.98, 0.99, 0.995)
            },
            "within_video_centered_feature_effective_rank": effective_rank,
            "image_metadata_keys": sorted(info_keys),
            "source_frame_ids_contiguous": all(
                int(b.source_frame_id) - int(a.source_frame_id) == 1
                for a, b in zip(case.frames[:-1], case.frames[1:], strict=True)
            ),
        }
        write_json(path, row)
        rows.append(row)
        print(f"Structure and embeddings {number}/{len(cases)} ({n} frames)", flush=True)
    summary = {
        "inventory": inventory,
        "audited_sequences": len(rows),
        "audited_frames": sum(r["num_frames"] for r in rows),
        "test_opened": False,
    }
    for split in ("train", "dev", "all"):
        selected = [r for r in rows if split == "all" or r["split"] == split]
        counts = Counter()
        for r in selected:
            counts.update(r["annotation_status"])
        total = sum(r["num_frames"] for r in selected)
        summary[split] = {
            "length": stats([r["num_frames"] for r in selected]),
            "annotation_status": dict(counts),
            "frame_weighted_present_fraction": counts["present"] / total,
            "case_mean_present_fraction": float(
                np.mean([r["annotation_present_fraction"] for r in selected])
            ),
            "cases_all_frames_annotated_present": sum(
                r["annotation_present_fraction"] == 1 for r in selected
            ),
            "adjacent_ssim_case_medians": stats(
                [r["temporal"]["1"]["central_ssim"]["p50"] for r in selected]
            ),
            "adjacent_embedding_case_medians": stats(
                [r["temporal"]["1"]["embedding_cosine"]["p50"] for r in selected]
            ),
            "greedy_cosine_representatives": {
                str(t): stats([r["greedy_cosine_representatives"][str(t)] for r in selected])
                for t in (0.98, 0.99, 0.995)
            },
            "feature_effective_rank": stats(
                [r["within_video_centered_feature_effective_rank"] for r in selected]
            ),
            "uniform_coverage": {
                str(k): {
                    metric: stats(
                        [r["uniform_embedding_coverage"][str(k)][metric] for r in selected]
                    )
                    for metric in (
                        "mean_nearest_cosine",
                        "fraction_cosine_ge_098",
                        "fraction_cosine_ge_0995",
                    )
                }
                for k in (1, 4, 8, 12)
            },
        }
    summary["source_metadata"] = source_metadata(args.data_root, cases)
    summary["image_metadata_keys"] = sorted({k for r in rows for k in r["image_metadata_keys"]})
    summary["all_source_frame_ids_contiguous"] = all(r["source_frame_ids_contiguous"] for r in rows)
    write_json(args.output / "cases.json", rows)
    write_json(args.output / "summary.json", summary)
    plots(rows, args.figures)


def plots(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.arange(0, 451, 25)
    for split, color in (("train", "#3479a8"), ("dev", "#dd8b38")):
        lens = [r["num_frames"] for r in rows if r["split"] == split]
        ax[0].hist(lens, bins=bins, alpha=0.7, label=f"{split} (n={len(lens)})", color=color)
    ax[0].set(xlabel="Extracted frames per sequence", ylabel="Sequences")
    ax[0].legend()
    x = np.sort([r["num_frames"] for r in rows])
    ax[1].plot(x, np.arange(1, len(x) + 1) / len(x), color="#3479a8")
    ax[1].set(xlabel="Extracted frames per sequence", ylabel="Cumulative fraction", ylim=(0, 1))
    fig.suptitle("BreastVid length — 145 train/development sequences; test excluded")
    fig.tight_layout()
    fig.savefig(output / "lengths.png", dpi=180)
    plt.close(fig)
    selected = sorted(rows, key=lambda r: (r["annotation_present_fraction"], r["num_frames"]))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = np.arange(len(selected))
    present = np.array([r["annotation_present_fraction"] for r in selected])
    ax.bar(x, present, width=1, color="#3479a8", label="Valid lesion annotation")
    ax.bar(x, 1 - present, bottom=present, width=1, color="#c9c9c9", label="Visibility unknown")
    ax.set(
        xlabel="Sequences sorted by annotation coverage",
        ylabel="Fraction of frames",
        ylim=(0, 1.13),
        xlim=(-1, len(x)),
    )
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(output / "visibility.png", dpi=180)
    plt.close(fig)
    rules = [
        ("Shortest", min(rows, key=lambda r: r["num_frames"])),
        ("Median length", sorted(rows, key=lambda r: r["num_frames"])[len(rows) // 2]),
        ("Longest", max(rows, key=lambda r: r["num_frames"])),
        ("Most unknown", min(rows, key=lambda r: r["annotation_present_fraction"])),
        ("Largest relative area variation", max(rows, key=lambda r: r["area_cv"] or 0)),
        (
            "Most embedding representatives",
            max(rows, key=lambda r: r["greedy_cosine_representatives"]["0.99"]),
        ),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, (rule, r) in zip(axes.flat, rules, strict=True):
        area = np.array([np.nan if a is None else a for a in r["normalized_union_area"]])
        x = np.linspace(0, 1, len(area))
        ax.plot(x, area * 100, lw=1, color="#3479a8")
        for i in np.flatnonzero(np.isnan(area)):
            ax.axvspan(
                max(0, (i - 0.5) / (len(area) - 1)),
                min(1, (i + 0.5) / (len(area) - 1)),
                color="#bbb",
                alpha=0.5,
                lw=0,
            )
        ax.set_title(
            f"{rule}: {r['alias']} / {r['num_frames']} frames\nValid box on {r['annotation_present_fraction']:.1%} of frames",
            fontsize=10,
        )
        ax.set(ylabel="Box union / image area (%)", xlabel="Relative frame position", xlim=(0, 1))
    fig.suptitle("Annotation-derived lesion extent; gray bands = visibility unknown")
    fig.tight_layout()
    fig.savefig(output / "area_trajectories.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ssim = [r["temporal"]["1"]["central_ssim"]["p50"] for r in rows]
    distant = [
        r["temporal"][str(max(1, round(0.5 * (r["num_frames"] - 1))))]["central_ssim"]["p50"]
        for r in rows
    ]
    axes[0].hist(ssim, bins=20, alpha=0.7, label="Adjacent")
    axes[0].hist(distant, bins=20, alpha=0.6, label="Half-video lag")
    axes[0].legend()
    axes[0].set(xlabel="Median central-crop SSIM per video", ylabel="Sequences")
    axes[1].scatter(
        [r["num_frames"] for r in rows],
        [r["greedy_cosine_representatives"]["0.99"] for r in rows],
        s=13,
        alpha=0.7,
    )
    axes[1].set(xlabel="Frames", ylabel="Greedy representatives at cosine 0.99")
    axes[2].boxplot(
        [
            [r["uniform_embedding_coverage"][str(k)]["fraction_cosine_ge_0995"] for r in rows]
            for k in (1, 4, 8, 12)
        ],
        tick_labels=["1", "4", "8", "12"],
    )
    axes[2].set(
        xlabel="Uniform frame budget",
        ylabel="Fraction near a selected feature (≥0.995)",
        ylim=(0, 1.05),
    )
    fig.suptitle("Visual redundancy proxies — not counts of independent diagnostic evidence")
    fig.tight_layout()
    fig.savefig(output / "redundancy.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/active_reader/manifest.jsonl"))
    p.add_argument("--splits", type=Path, default=Path("artifacts/active_reader/splits.json"))
    p.add_argument("--data-root", type=Path, default=Path("data/raw/breastvid/miccai_buv"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/active_reader/suitability_20261008/structure"),
    )
    p.add_argument(
        "--figures", type=Path, default=Path("reports/active_reader/suitability_figures")
    )
    run(p.parse_args())
