#!/usr/bin/env python3
"""Plot aggregate fixed-reader development outcomes from a completed campaign."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary, output):
    if not summary["complete"] or len(summary["arms"]) != 3:
        raise ValueError("Plot only a completed three-arm campaign")

    def metric(rows, method):
        return next(r for r in rows if r["method"] == method and r["budget"] == 12)

    full = summary["arms"]["full"]
    current = summary["arms"]["current_frame"]
    auxiliary = summary["arms"]["full_box_auxiliary"]
    bars = [
        ("Uniform", metric(summary["before"], "uniform"), "#8C969D"),
        ("Random", metric(summary["before"], "random"), "#8C969D"),
        ("Full history · before RL", metric(summary["before"], "full"), "#9CBED6"),
        ("Full history · after RL", metric(full["after"], "full"), "#176BA0"),
        ("Current frame · after RL", metric(current["after"], "current_frame"), "#D8842D"),
        ("Full + box bonus · after RL", metric(auxiliary["after"], "full"), "#8B697D"),
    ]
    contrasts = [
        ("RL learning\nfull after - full before", full["learning"]),
        ("Selection vs baseline\nfull after - uniform", full["versus_uniform"]),
        (
            "Older visual history\nfull - current batch, same weights",
            full["full_versus_current_batch"],
        ),
        ("Visual context\nfull - current frame, same weights", full["full_versus_current_frame"]),
        (
            "Matched training arms\ntrained full - trained current frame",
            summary["trained_full_versus_trained_current_frame"],
        ),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    values = [r["balanced_accuracy"] * 100 for _, r, _ in bars]
    ax.barh(range(len(bars)), values, color=[c for _, _, c in bars], height=0.62)
    ax.set_yticks(range(len(bars)), [name for name, _, _ in bars])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Balanced accuracy (%)")
    ax.set_title("Frozen reader at 12 selected frames", loc="left", weight="bold", pad=18)
    ax.axvline(50, color="#CBD0D4", linestyle=":", linewidth=1, zorder=0)
    for i, value in enumerate(values):
        ax.text(value + 1.4, i, f"{value:.1f}", va="center", fontsize=10)

    ax = axes[1]
    for i, (_, comparison) in enumerate(contrasts):
        value = comparison["delta"]["balanced_accuracy"] * 100
        lo, hi = [v * 100 for v in comparison["paired_group_bootstrap_95ci"]["balanced_accuracy"]]
        ax.hlines(i, lo, hi, color="#176BA0", linewidth=2)
        ax.scatter([value], [i], color="#176BA0", s=38, zorder=3)
    ax.set_yticks(range(len(contrasts)), [name for name, _ in contrasts])
    ax.invert_yaxis()
    ax.axvline(0, color="#89949B", linestyle="--", linewidth=1)
    ax.set_xlabel("Difference in balanced accuracy (percentage points)")
    ax.set_title("Paired differences and 95% intervals", loc="left", weight="bold", pad=18)
    ax.grid(axis="x", alpha=0.15)
    for ax in axes:
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#CBD0D4")
        ax.tick_params(axis="y", length=0, pad=9)
    n = metric(summary["before"], "uniform")["episodes"]
    figure.suptitle(
        f"Did the selector learn better evidence?  |  {n} development cases",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.02,
        0.02,
        "Fixed ultrasound Qwen reader · one training seed · test sealed. "
        "Connected-group bootstrap intervals are exploratory; no multiple-comparison correction.",
        fontsize=9,
        color="#52616B",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.91), w_pad=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output)
