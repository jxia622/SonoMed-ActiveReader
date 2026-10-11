#!/usr/bin/env python3
"""Aggregate a development navigation campaign without exposing case-level records."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import _bootstrap  # noqa: F401

from sonomed_vlm.active_reader.evaluation.metrics import summarize
from sonomed_vlm.active_reader.io import read_json, write_json
from sonomed_vlm.active_reader.navigation.evaluation import paired


def compare(left, right, bootstrap=1000):
    rows = [{**r, "method": "candidate"} for r in left]
    rows += [{**r, "method": "reference"} for r in right]
    result = paired(rows, "reference", bootstrap)
    if len(result) != 1:
        raise ValueError("Comparison must contain exactly one shared budget")
    reference = {(r["video_id"], r["seed"], r["budget"]): r for r in right}
    pairs = [(r, reference[(r["video_id"], r["seed"], r["budget"])]) for r in left]
    result[0]["selected_sets_differ"] = sum(
        a["selected_indices"] != b["selected_indices"] for a, b in pairs
    )
    result[0]["mean_selected_set_jaccard"] = mean(
        len(set(a["selected_indices"]) & set(b["selected_indices"]))
        / len(set(a["selected_indices"]) | set(b["selected_indices"]))
        for a, b in pairs
    )
    return result[0]


def subset(rows, method, budget=12):
    return [r for r in rows if r["method"] == method and r["budget"] == budget]


def table(rows):
    result = []
    for method, budget in sorted({(r["method"], r["budget"]) for r in rows}):
        selected = subset(rows, method, budget)
        metrics = summarize(selected)
        result.append(
            {
                "method": method,
                "budget": budget,
                **{
                    k: metrics[k]
                    for k in (
                        "accuracy",
                        "balanced_accuracy",
                        "malignant_sensitivity",
                        "benign_specificity",
                        "confusion_matrix",
                    )
                },
                "mean_true_label_log_score": mean(
                    r["reader_true_label_log_score"] for r in selected
                ),
                "mean_selector_calls": mean(r["selector_model_calls"] for r in selected),
                "mean_selector_image_presentations": mean(
                    r["selector_image_presentations"] for r in selected
                ),
                "mean_reader_logical_calls": mean(r["reader_model_calls"] for r in selected),
                "mean_reader_executed_calls": mean(
                    r["executed_reader_model_calls"] for r in selected
                ),
                "episodes": len(selected),
            }
        )
    return result


def report(directory):
    before = read_json(directory / "before" / "trajectories.json")
    if any(r["split"] != "dev" for r in before):
        raise ValueError("Development records required")
    result = {
        "complete": (directory / "report.json").exists(),
        "before": table(before),
        "before_paired_vs_uniform": paired(before),
        "arms": {},
        "test_opened": False,
    }
    after_rows = {}
    for arm, history in (
        ("full", "full"),
        ("current_frame", "current_frame"),
        ("full_box_auxiliary", "full"),
    ):
        after_path = directory / arm / "after" / "trajectories.json"
        if not after_path.exists():
            continue
        after = read_json(after_path)
        if any(r["split"] != "dev" for r in after):
            raise ValueError("Development records required")
        after_rows[arm] = after
        training = read_json(directory / arm / "training" / "training_report.json")
        steps = [
            read_json(p)
            for p in sorted((directory / arm / "training" / "steps").glob("*.json"))
        ]
        traces = [s["summary"] for s in steps]
        train_rows = [r for s in steps for r in s["trajectories"]]
        result["arms"][arm] = {
            "after": table(after),
            "learning": compare(subset(after, history), subset(before, history)),
            "versus_uniform": compare(subset(after, history), subset(before, "uniform")),
            "versus_random": compare(subset(after, history), subset(before, "random")),
            "full_versus_current_batch": compare(
                subset(after, "full"), subset(after, "current_batch")
            ),
            "full_versus_current_frame": compare(
                subset(after, "full"), subset(after, "current_frame")
            ),
            "training": {
                "updates": len(traces),
                "class_counts": dict(Counter(s["trajectories"][0]["label"] for s in steps)),
                "zero_variance_groups": sum(s["zero_variance_group"] for s in traces),
                "maximum_likelihood_drift": max(s["max_behavior_logprob_drift"] for s in traces),
                "mean_gradient_norm_before_clipping": mean(s["gradient_norm"] for s in traces),
                "min_gradient_norm_before_clipping": min(s["gradient_norm"] for s in traces),
                "mean_reader_gain": mean(r["private_reader_gain"] for r in train_rows),
                "mean_box_bonus": mean(r["private_box_bonus"] for r in train_rows),
                "mean_distinct_final_sets_per_group": mean(
                    len({tuple(r["selected_indices"]) for r in s["trajectories"]})
                    for s in steps
                ),
                "groups_with_nonzero_box_bonus": sum(
                    any(r["private_box_bonus"] != 0 for r in s["trajectories"])
                    for s in steps
                ),
                "refinement_actions": sum(
                    t["action"]["action"] == "refine"
                    for r in train_rows
                    for t in r["turns"]
                ),
                "inspection_actions": sum(len(r["turns"]) for r in train_rows),
                **{
                    k: training[k]
                    for k in (
                        "language_adapter_changed",
                        "frozen_reader_unchanged",
                        "frozen_reference_and_vision_unchanged",
                        "save_reload_identical",
                    )
                },
            },
        }
    if {"full", "current_frame"} <= after_rows.keys():
        result["trained_full_versus_trained_current_frame"] = compare(
            subset(after_rows["full"], "full"),
            subset(after_rows["current_frame"], "current_frame"),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.directory)
    write_json(args.output, result)
    print(f"Campaign complete={result['complete']}; summarized {len(result['arms'])} training arms")
