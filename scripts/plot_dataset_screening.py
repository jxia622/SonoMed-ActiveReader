#!/usr/bin/env python3
"""Plot anonymous aggregates from the frozen ACOUSLIC feasibility screen."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, output):
    data = json.loads(summary.read_text())
    by_method = {r["method"]: r for r in data["methods"]}
    keys = [
        "uniform",
        "random",
        "coverage",
        "qwen_current",
        "qwen_history",
        "broader_uniform24",
        "privileged12",
    ]
    names = [
        "Uniform 12",
        "Random 12",
        "Coverage 12",
        "Qwen current 12",
        "Qwen history 12",
        "Uniform 24*",
        "Privileged 12†",
    ]
    y = np.arange(len(keys))
    rows = [by_method[k] for k in keys]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.7, 5.8), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    capture = [r["annotated_capture_count"] / r["case_start_pairs"] for r in rows]
    colors = ["#50687c"] * 3 + ["#287b78"] * 2 + ["#949caa", "#b88636"]
    left.barh(y, capture, color=colors, height=0.57)
    for j, v in enumerate(capture):
        left.text(v + 0.02, j, f"{round(v * 10)}/10", va="center", fontsize=10)
    left.set(
        yticks=y,
        yticklabels=names,
        xlim=(0, 1.2),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        xlabel="Fraction of case/start pairs",
        title="Acquired any annotated plane",
    )
    left.invert_yaxis()
    best = [r["mean_best_available_wfss"] for r in rows]
    final = [r["mean_final_wfss"] for r in rows]
    right.barh(y - 0.15, best, height=0.29, color="#9eafb9", label="Best among acquired frames")
    right.barh(y + 0.15, final, height=0.29, color="#d56a46", label="Frozen reader's choice")
    for j, (b, f) in enumerate(zip(best, final, strict=True)):
        right.text(b + 0.02, j - 0.15, f"{b:.2f}", va="center", fontsize=9)
        right.text(f + 0.02, j + 0.15, f"{f:.2f}", va="center", fontsize=9)
    right.set(
        yticks=y,
        yticklabels=[],
        xlim=(0, 1.2),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        xlabel="Mean annotation WFSS (0 to 1)",
        title="Reader failed on ordinary sampled sets",
    )
    right.invert_yaxis()
    handles, labels = right.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.65, 0.14),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    for ax in (left, right):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.16)
    fig.suptitle(
        "ACOUSLIC feasibility screen with the original frozen ultrasound Qwen",
        fontsize=15,
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.015,
        "Five public training subjects, two starts each; exploratory, no RL training.\n*Twice the observation budget.  †Uses private labels to choose the frame set; not a deployable method.\nWFSS follows annotations; unannotated frames are not verified anatomically unusable.",
        fontsize=9,
        va="bottom",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    fig.savefig(output, dpi=170, facecolor="white")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plot(args.summary, args.output)
