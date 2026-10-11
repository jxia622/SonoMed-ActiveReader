#!/usr/bin/env python3
"""Report all v1 tuning results without admitting held-out subjects."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summarize_acouslic_pilot import paired_subject_contrasts

from sonomed_vlm.active_reader.io import read_json, write_json


def summarize(run, output):
    audit = read_json(run / "development_audit.json")
    assert audit["status"] == "passed" and not audit["heldout_opened"]
    assert not (run / "evaluation").exists()
    rows = [
        r
        for r in read_json(run / "development_baselines/acquisition.json")
        if r["split"] == "tune" and r["primary"]
    ]
    for mode in ("full", "current", "geometry"):
        for phase, prefix in (("before_tune", "pre_rl"), ("after_tune", "rl")):
            records = read_json(run / mode / phase / "acquisition.json")
            rows.extend([{**r, "method": f"{prefix}_{mode}"} for r in records])
    cases = sorted({r["alias"] for r in rows})
    assert len(cases) == 8 and all(r["split"] == "tune" for r in rows)
    methods = sorted({r["method"] for r in rows})
    matrix = {}
    for method in methods:
        values = []
        for case in cases:
            subset = [r for r in rows if r["alias"] == case and r["method"] == method]
            assert len(subset) == (100 if method in {"random", "random_actions"} else 2)
            values.append(np.mean([r["best_wfss"] for r in subset]))
        matrix[method] = np.array(values)
    sample = np.random.default_rng(20261011).integers(0, 8, size=(10000, 8))
    comparisons = [("rl_full", m) for m in methods if m not in {"rl_full", "privileged"}]
    contrasts, radius = paired_subject_contrasts(matrix, comparisons, sample)
    stats = {
        m: {
            "mean": float(v.mean()),
            "subject_values": v.tolist(),
            "subject_bootstrap_95": np.quantile(v[sample].mean(1), [0.025, 0.975]).tolist(),
        }
        for m, v in matrix.items()
    }
    result = {
        "scope": "Exploratory tuning results; eight subjects, not held-out confirmation",
        "subjects": 8,
        "methods": stats,
        "contrasts": contrasts,
        "simultaneous_radius": radius,
        "training_signal": audit["training_signal"],
        "history_probe": audit["tuning_older_image_probe"],
        "heldout_opened": False,
        "breastvid_test_opened": False,
        "development_gate_passed": stats["rl_full"]["mean"]
        > max(
            stats[m]["mean"]
            for m in (
                "uniform",
                "coverage",
                "geometry_prior",
                "random",
                "random_actions",
                "pre_rl_full",
            )
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "development_results.json", result)
    order = [
        "pre_rl_full",
        "rl_full",
        "rl_current",
        "rl_geometry",
        "uniform",
        "geometry_prior",
        "random",
        "random_actions",
    ]
    names = [
        "Qwen before RL",
        "Qwen RL: full history",
        "Qwen RL: latest images",
        "Qwen RL: blank images",
        "Uniform sampling",
        "Training location prior",
        "Random frames",
        "Random inspection actions",
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 5.7), layout="constrained")
    colors = [
        "#abb5bd",
        "#187fa6",
        "#6fa6ba",
        "#a5c2cd",
        "#e79b45",
        "#e9b675",
        "#c2c5c8",
        "#c2c5c8",
    ]
    y = np.arange(len(order))
    means = np.array([stats[m]["mean"] for m in order])
    ax.barh(y, means, color=colors, height=0.65)
    for i, m in enumerate(order):
        lo, hi = stats[m]["subject_bootstrap_95"]
        ax.plot([lo, hi], [i, i], color="#263542", lw=1.4)
        ax.plot([lo, hi], [i, i], "|", color="#263542")
        ax.text(0.91, i, f"{means[i]:.3f}", va="center", ha="right", weight="bold")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("Best supported-plane quality among 12 acquired frames (WFSS)")
    ax.set_title(
        "RL improves from its starting policy, but trails uniform sampling",
        loc="left",
        weight="bold",
        pad=30,
    )
    ax.text(
        0,
        1.025,
        "Development only • 8 subjects • whiskers: subject-bootstrap 95% intervals",
        transform=ax.transAxes,
        color="#59636d",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.savefig(output / "development_results.png", dpi=180)
    plt.close(fig)
    print({m: stats[m]["mean"] for m in order})
    print("Development gate passed:", result["development_gate_passed"])
    print("Paired contrasts:", contrasts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summarize(a.run, a.output)
