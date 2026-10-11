#!/usr/bin/env python3
"""Plot aggregate, audited BreastVid reader and history-probe results."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sonomed_vlm.active_reader.io import read_json


def interval_points(ax, labels, means, intervals, color, scale=1):
    means = np.asarray(means) * scale
    intervals = np.asarray(intervals) * scale
    ax.errorbar(
        means,
        np.arange(len(labels)),
        xerr=np.maximum(0, np.stack([means - intervals[:, 0], intervals[:, 1] - means])),
        fmt="o",
        color=color,
        capsize=3,
        markersize=6,
    )
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)


def main(args):
    for phase in ("development", "history"):
        assert read_json(args.directory / phase / "input_audit.json")["status"] == "passed"
    dev = read_json(args.directory / "development/summary.json")
    hist = read_json(args.directory / "history/history_predictability.json")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    budgets = [4, 8, 12]
    for prefix, name, color, offset in (
        ("uniform", "Uniform", "#247ba0", -0.18),
        ("random_three_seeds", "Random (3 seeds)", "#dc7b22", 0),
        ("privileged_best_of_eight_sets", "Label-informed best of 8", "#8955a8", 0.18),
    ):
        results = [dev["methods"][f"{prefix}_{k}"] for k in budgets]
        means = np.array([r["balanced_accuracy"] for r in results]) * 100
        ci = np.array([r["group_bootstrap_95ci"]["balanced_accuracy"] for r in results]) * 100
        axes[0].errorbar(
            np.array(budgets) + offset,
            means,
            yerr=np.maximum(0, np.stack([means - ci[:, 0], ci[:, 1] - means])),
            fmt="o-",
            color=color,
            capsize=3,
            label=name,
        )
    axes[0].set(
        title="Frame choices change reader outcomes",
        xlabel="Selected frame budget",
        ylabel="Development balanced accuracy (%)",
        xticks=budgets,
        ylim=(25, 102),
    )
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(loc="lower left", fontsize=9)
    names = [
        ("first_frame", "First frame"),
        ("center_frame", "Middle frame"),
        ("random_single_expectation_all_frames", "Random single (exact expectation)"),
        ("dense_single_mean_probability", "All frames, single-frame pooling"),
        ("dense_chunk12_weighted_mean_probability", "All frames, 12-frame batch pooling"),
        ("privileged_best_single_all_frames", "Label-informed best single"),
    ]
    results = [dev["methods"][k] for k, _ in names]
    interval_points(
        axes[1],
        [name for _, name in names],
        [r["balanced_accuracy"] for r in results],
        [r["group_bootstrap_95ci"]["balanced_accuracy"] for r in results],
        "#4c6675",
        100,
    )
    axes[1].set(xlabel="Development balanced accuracy (%)", xlim=(25, 102))
    axes[1].set_title("More frames alone do not resolve errors")
    fig.suptitle(f"Frozen ultrasound Qwen reader · {dev['cases']} development sequences", y=0.98)
    fig.text(
        0.5,
        0.015,
        "95% connected-group bootstrap intervals. Label-informed choices are retrospective diagnostics, not deployable selectors.\n"
        "Dense controls pool separate calls over all frames; they do not use joint whole-video attention.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=2.5)
    fig.savefig(args.output / "reader_headroom.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    names = [
        ("bookkeeping", "Bookkeeping only"),
        ("current_frame", "+ Current frame"),
        ("current_batch", "+ Latest batch"),
        ("history", "+ Earlier visual history"),
        ("uniform_random_menu", "Random menu action"),
        ("coverage", "Coverage action"),
        ("privileged_menu_best", "Label-informed best action"),
    ]
    results = [hist["validation"][k] for k, _ in names]
    interval_points(
        axes[0],
        [name for _, name in names],
        [r["mean"]["selected_log_score"] for r in results],
        [r["group_bootstrap_95ci"]["selected_log_score"] for r in results],
        "#247ba0",
    )
    axes[0].set(
        xlabel="Selected evidence log score (higher is better)", title="Held-out action selection"
    )
    names = [
        ("current_frame_minus_bookkeeping", "Current frame minus bookkeeping"),
        ("current_batch_minus_current_frame", "Latest batch minus current frame"),
        ("history_minus_current_batch", "History minus latest batch"),
        ("history_minus_bookkeeping", "History minus bookkeeping"),
        ("history_minus_uniform_random_menu", "History minus random action"),
        ("history_minus_coverage", "History minus coverage"),
    ]
    results = [hist["paired_contrasts"][k] for k, _ in names]
    interval_points(
        axes[1],
        [name for _, name in names],
        [r["mean"]["selected_log_score"] for r in results],
        [r["group_bootstrap_95ci"]["selected_log_score"] for r in results],
        "#8955a8",
    )
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Paired log-score difference", title="Incremental value of visual history")
    fig.suptitle(
        f"History probe · {hist['validation_groups']} held-out training groups · 12 final frames",
        y=0.98,
    )
    fig.text(
        0.5,
        0.018,
        "95% paired/group bootstrap intervals. Fixed ResNet18 features + ridge ranking; no Qwen policy training.\n"
        "Two states per sequence remain one independent group. Test split untouched.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94), w_pad=2.5)
    fig.savefig(args.output / "history_value.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/active_reader/suitability_figures"))
    main(p.parse_args())
