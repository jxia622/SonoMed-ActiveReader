#!/usr/bin/env python3
"""Plot annotation coverage on training subjects only; never load held-out data."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from sonomed_vlm.active_reader.acouslic.data import Cases
from sonomed_vlm.active_reader.io import write_json


def plot(data_root, output):
    data = Cases(data_root)
    rows = []
    for case in data.subset("train"):
        quality, labels = data.quality(case["alias"])
        rows.append((case["alias"], int((quality > 0).sum()), labels))
    assert len(rows) == 32
    ranked = sorted(rows, key=lambda x: (x[1], x[0]))
    examples = [ranked[i] for i in (0, 10, 21, 31)]
    counts = np.array([row[1] for row in rows])
    colors = ["#edf1f3", "#287b78", "#e7ad53"]
    fig, (ax, timeline) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1.5]}
    )
    ax.hist(100 * counts / 840, bins=8, color=colors[1], edgecolor="white")
    ax.set(xlabel="Frames with supported-plane annotation (%)", ylabel="Training subjects")
    ax.set_title("Supported measurement planes are sparsely annotated", loc="left", fontsize=12)
    timeline.imshow(
        np.stack([row[2] for row in examples]),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
    )
    timeline.set(
        yticks=range(4),
        yticklabels=[f"{row[0]} ({row[1]} frames)" for row in examples],
        xticks=[0, 139, 279, 419, 559, 699, 839],
        xlabel="Global frame index across six sweeps",
    )
    for boundary in (139.5, 279.5, 419.5, 559.5, 699.5):
        timeline.axvline(boundary, color="#82919a", linewidth=0.7)
    timeline.set_title(
        "Example training scans selected by annotation-count rank", loc="left", fontsize=12
    )
    timeline.legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in zip(colors, ["Unannotated", "Optimal", "Suboptimal"], strict=True)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        frameon=False,
    )
    total = len(rows) * 840
    fig.suptitle(f"ACOUSLIC development data: {len(rows)} subjects, {total:,} frames", fontsize=15)
    fig.text(
        0.08,
        0.02,
        f"{counts.sum()} frames ({counts.sum() / total:.2%}) have a supported-plane annotation. "
        "Unannotated frames are not verified anatomical negatives.\n"
        "Examples are ranks 1, 11, 22 and 32 by annotated-frame count. Held-out subjects are excluded.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "training_annotation_coverage.png", dpi=170)
    plt.close(fig)
    write_json(
        output / "training_annotation_coverage.json",
        {
            "subjects": 32,
            "frames": total,
            "annotated_frames": int(counts.sum()),
            "annotated_frames_per_subject": {
                "min": int(counts.min()),
                "median": float(np.median(counts)),
                "max": int(counts.max()),
            },
            "annotated_frames_by_sweep": np.stack([row[2] > 0 for row in rows])
            .reshape(32, 6, 140)
            .sum(axis=(0, 2))
            .tolist(),
            "example_selection": "ranks 1, 11, 22, 32 by annotated-frame count, alias tie break",
            "heldout_opened": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.data, args.output)
