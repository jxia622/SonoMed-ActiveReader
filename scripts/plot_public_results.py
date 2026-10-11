#!/usr/bin/env python3
"""Rebuild publication figures and CSV using only reviewed aggregate JSON."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    first = json.loads((ROOT / "results/acouslic_v1.json").read_text())
    refined = json.loads((ROOT / "results/acouslic_v2b.json").read_text())
    breast = json.loads((ROOT / "results/breastvid_reader.json").read_text())
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "axes.edgecolor": "#cbd5e1",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    colors = ["#2563eb", "#0d9488", "#64748b"]
    modes = ["full", "current", "geometry"]
    labels = ["Full history", "Current observations", "Blank images"]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), sharey=True)
    for ax, campaign in zip(axes, ["v1", "v2b"], strict=True):
        for position, (mode, color) in enumerate(zip(modes, colors, strict=True)):
            if campaign == "v1":
                before = first["methods"]["pre_rl_" + mode]["mean"]
                after = first["methods"]["rl_" + mode]["mean"]
            else:
                before = refined["arms"][mode]["before_tune_wfss"]
                after = refined["arms"][mode]["after_tune_wfss"]
            x = [position - 0.15, position + 0.15]
            ax.plot(x, [before, after], color=color, lw=2.5)
            ax.scatter(x[0], before, s=85, facecolors="white", edgecolors=color, lw=2, zorder=3)
            ax.scatter(x[1], after, s=85, color=color, zorder=3)
            ax.text(x[0], before - 0.035, f"{before:.3f}", ha="center", fontsize=10, color=color)
            ax.text(x[1], after + 0.018, f"{after:.3f}", ha="center", fontsize=10, color=color)
            rows.append(
                {
                    "campaign": campaign,
                    "method": mode,
                    "before_rl": before,
                    "after_rl": after,
                    "updates": 32 if campaign == "v1" else 64,
                    "tune_subjects": 8,
                    "budget": 12,
                }
            )
        ax.axhline(0.45, color="#334155", ls="--", lw=1.5, label="Uniform · 0.450")
        ax.axhline(0.425, color="#94a3b8", ls=":", lw=1.5, label="Location prior · 0.425")
        ax.axhline(
            0.23925, color="#d97706", ls="--", lw=1.1, alpha=0.7, label="Random frames · 0.239"
        )
        ax.set_xticks(range(3), labels, fontsize=10)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0, 0.55)
        ax.grid(axis="y", alpha=0.18)
        ax.set_title(
            "Initial pilot · 32 updates" if campaign == "v1" else "Refinement · 64 updates",
            loc="left",
            pad=12,
            weight="bold",
        )
    axes[0].set_ylabel("Mean best acquired WFSS (0-1)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "RL changes some means; uniform sampling remains ahead",
        x=0.07,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.07,
        0.865,
        "8 reused tuning subjects · 12 frames · open circle: pre-RL / filled circle: post-RL",
        fontsize=11,
        color="#475569",
    )
    fig.text(
        0.07,
        0.04,
        "Descriptive point estimates, not confirmed superiority. v1 paired intervals include zero; no held-out outcomes opened.",
        fontsize=10,
        color="#475569",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.25, wspace=0.17)
    fig.savefig(assets / "acouslic_results.png", dpi=180)
    plt.close(fig)
    for method in [
        "uniform",
        "geometry_prior",
        "coverage",
        "random_actions",
        "random",
        "privileged",
    ]:
        rows.append(
            {
                "campaign": "shared_reference",
                "method": method,
                "before_rl": "",
                "after_rl": first["methods"][method]["mean"],
                "updates": 0,
                "tune_subjects": 8,
                "budget": 12,
            }
        )
    with (ROOT / "results/acouslic_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(4)
    for key, label, color, style in [
        ("uniform", "Uniform", "#2563eb", "-"),
        ("random", "Random", "#0d9488", "-"),
        ("privileged_best", "Privileged best candidate (label-informed)", "#94a3b8", "--"),
    ]:
        ax.plot(
            x,
            100 * np.array(breast["balanced_accuracy"][key]),
            marker="o",
            color=color,
            ls=style,
            lw=2,
            label=label,
        )
    ax.set_xticks(x, ["1", "4", "8", "12"])
    ax.set_xlabel("Selected frames (oracle information/compute is not matched)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.suptitle(
        "BreastVid: oracle headroom is not a learned selection gain",
        x=0.09,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09, 0.88, "26 development sequences · identical frozen reader and prompt", color="#475569"
    )
    fig.text(
        0.09,
        0.04,
        "Oracle: best individual frame at budget 1; best of eight sets at budgets 4/8/12. Test split excluded.",
        fontsize=10,
        color="#475569",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.19)
    fig.savefig(assets / "breastvid_headroom.png", dpi=180)
    plt.close(fig)
    print("Rebuilt two aggregate-only figures and results/acouslic_summary.csv")


if __name__ == "__main__":
    main()
