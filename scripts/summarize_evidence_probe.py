#!/usr/bin/env python3
"""Aggregate corrected baselines and paired fixed-state learning/history outcomes."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import _bootstrap  # noqa: F401
import numpy as np

from sonomed_vlm.active_reader.io import read_json, write_json
from sonomed_vlm.active_reader.navigation.preference import summarize_rankings


def load_rankings(path):
    return [r for p in sorted(path.glob("*.json")) if "selected_candidate" in (r := read_json(p))]


def paired(left, right):
    by = {r["state_id"]: r for r in right}
    if {r["state_id"] for r in left} != set(by):
        raise ValueError("Fixed-state comparison has unmatched states")
    output = {}
    for role in ("fit", "validation"):
        for depth in (4, 8, None):
            groups = defaultdict(list)
            for a in left:
                if a["role"] != role or (depth is not None and a["observed_frames"] != depth):
                    continue
                b = by[a["state_id"]]
                if a["group_id"] != b["group_id"] or a["role"] != b["role"]:
                    raise ValueError("Comparison group/partition differs")
                groups[a["group_id"]].append(
                    [
                        a["selected_log_score"] - b["selected_log_score"],
                        a["expected_log_score"] - b["expected_log_score"],
                        int(a["selected_correct"]) - int(b["selected_correct"]),
                    ]
                )
            if not groups:
                continue
            values = np.array([np.mean(v, axis=0) for _, v in sorted(groups.items())])
            rng = np.random.default_rng(42)
            samples = values[rng.integers(len(values), size=(1000, len(values)))].mean(axis=1)
            output[f"{role}:{depth or 'all'}"] = {
                "groups": len(values),
                "delta": dict(
                    zip(
                        ("selected_log_score", "expected_log_score", "correct_fraction"),
                        values.mean(axis=0).tolist(),
                        strict=True,
                    )
                ),
                "paired_group_bootstrap_95ci": dict(
                    zip(
                        ("selected_log_score", "expected_log_score", "correct_fraction"),
                        np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
                        strict=True,
                    )
                ),
            }
    return output


def run(directory):
    result = {"complete": (directory / "report.json").exists(), "test_opened": False}
    if result["complete"]:
        result["status"] = read_json(directory / "report.json")["status"]
    if (directory / "corrected_baselines/report.json").exists():
        baseline = read_json(directory / "corrected_baselines/report.json")
        rows = read_json(directory / "corrected_baselines/trajectories.json")
        result["corrected_dev_baselines"] = [
            {
                "method": r["method"],
                "budget": r["budget"],
                "cases": r["videos"],
                "episodes": r["episodes"],
                "balanced_accuracy": r["balanced_accuracy"],
                "group_bootstrap_95ci": r["group_bootstrap_95ci"]["balanced_accuracy"],
                "mean_log_score": mean(
                    x["reader_true_label_log_score"]
                    for x in rows
                    if x["method"] == r["method"] and x["budget"] == r["budget"]
                ),
            }
            for r in baseline["curves"]
        ]
    if (directory / "selection_signal.json").exists():
        result["selection_signal"] = read_json(directory / "selection_signal.json")
    result["arms"] = {}
    for arm in ("full", "current_frame"):
        before = load_rankings(directory / arm / "before")
        after = load_rankings(directory / arm / "after")
        if not before or len(before) != len(after):
            continue
        result["arms"][arm] = {
            "before": summarize_rankings(before),
            "after": summarize_rankings(after),
            "learning": paired(after, before),
            "evaluation_cost": {
                "before_and_after_logical_selector_passes": sum(
                    len(r["log_probabilities"]) for r in before + after
                ),
                "before_and_after_selector_image_presentations": sum(
                    len(r["log_probabilities"]) * len(r["native_inputs"]["image_grid_thw"])
                    for r in before + after
                ),
                "scope": "One teacher-forced selector pass per complete candidate action; excludes reader labeling, training, and other context ablations",
            },
        }
        training_path = directory / arm / "training/report.json"
        if training_path.exists():
            result["arms"][arm]["training"] = read_json(training_path)
    full = load_rankings(directory / "full/after")
    for mode in ("current_frame", "shuffled_history", "blank_history", "no_visual"):
        ablated = load_rankings(directory / "full/ablations" / mode)
        if full and len(full) == len(ablated):
            result[f"full_minus_{mode}_same_weights"] = paired(full, ablated)
    current = load_rankings(directory / "current_frame/after")
    if full and len(full) == len(current):
        result["trained_full_minus_trained_current_frame"] = paired(full, current)
    result["interpretation"] = (
        "Probe states are clustered by connected training group. Candidate rankings and learning-test accuracy are not end-to-end development results. No multiple-comparison correction."
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    write_json(args.output, run(args.directory))
