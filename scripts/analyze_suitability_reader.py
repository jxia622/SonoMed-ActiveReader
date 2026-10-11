#!/usr/bin/env python3
"""Paired development headroom estimates from a completed frozen-reader audit."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from sonomed_vlm.active_reader.io import read_json, write_json


def outcome(label, probability):
    prediction = "malignant" if probability > 0.5 else "benign"
    return [
        float(prediction == label),
        math.log(max(1e-8, probability if label == "malignant" else 1 - probability)),
    ]


def summarize(rows, seed=142):
    methods = defaultdict(list)
    case_diagnostics = []
    for row in rows:
        label, records = row["label"], row["evaluations"]
        singles = sorted(
            [r for r in records if r["method"] == "single"], key=lambda r: r["candidate"]
        )
        assert len(singles) == row["num_frames"]
        single_scores = np.array([r["probabilities"][label] for r in singles])
        current = {}
        for name, record in [
            ("first_frame", singles[0]),
            ("center_frame", singles[len(singles) // 2]),
            ("privileged_best_single_all_frames", singles[int(single_scores.argmax())]),
        ]:
            current[name] = outcome(label, record["probabilities"]["malignant"])
        current["random_single_expectation_all_frames"] = np.mean(
            [outcome(label, r["probabilities"]["malignant"]) for r in singles], axis=0
        ).tolist()
        current["dense_single_mean_probability"] = outcome(
            label, float(np.mean([r["probabilities"]["malignant"] for r in singles]))
        )
        chunks = [r for r in records if r["method"] == "dense_chunk"]
        assert [i for r in chunks for i in r["indices"]] == list(range(row["num_frames"]))
        dense_p = (
            sum(len(r["indices"]) * r["probabilities"]["malignant"] for r in chunks)
            / row["num_frames"]
        )
        current["dense_chunk12_weighted_mean_probability"] = outcome(label, dense_p)
        diag = {
            "video_id": row["video_id"],
            "group_id": row["group_id"],
            "label": label,
            "single_correct_fraction": float(np.mean([r["prediction"] == label for r in singles])),
            "single_true_probability_range": float(np.ptp(single_scores)),
            "single_logscore_range": float(np.ptp(np.log(np.maximum(1e-8, single_scores)))),
            "single_prediction_varies": len({r["prediction"] for r in singles}) > 1,
            "all_single_frames_wrong": all(r["prediction"] != label for r in singles),
            "budgets": {},
        }
        for budget in (4, 8, 12):
            subset = [r for r in records if r.get("budget") == budget]
            for method in ("uniform", "privileged_area"):
                r = next(r for r in subset if r["method"] == method)
                current[f"{method}_{budget}"] = outcome(label, r["probabilities"]["malignant"])
            random = [r for r in subset if r["method"] == "random"]
            current[f"random_three_seeds_{budget}"] = np.mean(
                [outcome(label, r["probabilities"]["malignant"]) for r in random], axis=0
            ).tolist()
            menu = [r for r in subset if r["method"] != "privileged_area"]
            assert len(menu) == 8
            best = max(menu, key=lambda r: r["probabilities"][label])
            current[f"privileged_best_of_eight_sets_{budget}"] = outcome(
                label, best["probabilities"]["malignant"]
            )
            diag["budgets"][str(budget)] = {
                "distinct_sets": len({tuple(r["indices"]) for r in menu}),
                "prediction_varies": len({r["prediction"] for r in menu}) > 1,
                "all_wrong": all(r["prediction"] != label for r in menu),
                "logscore_range": float(
                    np.ptp([math.log(max(1e-8, r["probabilities"][label])) for r in menu])
                ),
            }
        for name, values in current.items():
            methods[name].append(
                {
                    "video_id": row["video_id"],
                    "group_id": row["group_id"],
                    "label": label,
                    "correct_fraction": values[0],
                    "log_score": values[1],
                }
            )
        case_diagnostics.append(diag)
    bygroup = defaultdict(list)
    for i, r in enumerate(rows):
        bygroup[r["group_id"]].append(i)
    groups = sorted(bygroup)
    rng = np.random.default_rng(seed)
    sample_indices = [
        np.concatenate([bygroup[groups[i]] for i in rng.integers(len(groups), size=len(groups))])
        for _ in range(2000)
    ]
    labels = np.array([r["label"] == "malignant" for r in rows])

    def metrics(correct, logscore, indices):
        y = labels[indices]
        if not y.any() or y.all():
            return [np.nan, np.nan, np.nan]
        return [
            0.5 * (correct[indices][y].mean() + correct[indices][~y].mean()),
            correct[indices].mean(),
            logscore[indices].mean(),
        ]

    estimates = {}
    samples = {}
    for name, values in methods.items():
        c = np.array([r["correct_fraction"] for r in values])
        s = np.array([r["log_score"] for r in values])
        estimate = metrics(c, s, np.arange(len(rows)))
        draws = np.array([metrics(c, s, idx) for idx in sample_indices])
        samples[name] = draws
        estimates[name] = {
            "balanced_accuracy": estimate[0],
            "accuracy": estimate[1],
            "mean_log_score": estimate[2],
            "group_bootstrap_95ci": dict(
                zip(
                    ("balanced_accuracy", "accuracy", "mean_log_score"),
                    np.nanquantile(draws, [0.025, 0.975], axis=0).T.tolist(),
                    strict=True,
                )
            ),
        }
    contrasts = {}
    for budget in (4, 8, 12):
        for left in (
            "dense_single_mean_probability",
            "dense_chunk12_weighted_mean_probability",
            f"privileged_best_of_eight_sets_{budget}",
            f"random_three_seeds_{budget}",
            "center_frame",
        ):
            right = f"uniform_{budget}"
            contrasts[f"{left}_minus_{right}"] = {
                "delta": {
                    metric: estimates[left][metric] - estimates[right][metric]
                    for metric in ("balanced_accuracy", "accuracy", "mean_log_score")
                },
                "paired_group_bootstrap_95ci": dict(
                    zip(
                        ("balanced_accuracy", "accuracy", "mean_log_score"),
                        np.nanquantile(
                            samples[left] - samples[right], [0.025, 0.975], axis=0
                        ).T.tolist(),
                        strict=True,
                    )
                ),
            }
    return {
        "cases": len(rows),
        "groups": len(groups),
        "test_opened": False,
        "methods": estimates,
        "paired_contrasts": contrasts,
        "selection_sensitivity": {
            "single_prediction_varies_cases": sum(
                r["single_prediction_varies"] for r in case_diagnostics
            ),
            "all_single_frames_wrong_cases": sum(
                r["all_single_frames_wrong"] for r in case_diagnostics
            ),
            "mean_single_true_probability_range": float(
                np.mean([r["single_true_probability_range"] for r in case_diagnostics])
            ),
            "budgets": {
                str(k): {
                    "prediction_varies_cases": sum(
                        r["budgets"][str(k)]["prediction_varies"] for r in case_diagnostics
                    ),
                    "all_menu_wrong_cases": sum(
                        r["budgets"][str(k)]["all_wrong"] for r in case_diagnostics
                    ),
                    "mean_logscore_range": float(
                        np.mean([r["budgets"][str(k)]["logscore_range"] for r in case_diagnostics])
                    ),
                }
                for k in (4, 8, 12)
            },
        },
        "costs": {
            "single_frame": {"unique_frames": 1, "reader_forwards": 2},
            "uniform": {str(k): {"unique_frames": k, "reader_forwards": 2} for k in (4, 8, 12)},
            "dense_single": {
                "mean_unique_frames": float(np.mean([r["num_frames"] for r in rows])),
                "mean_reader_forwards": float(np.mean([2 * r["num_frames"] for r in rows])),
            },
            "dense_chunk12": {
                "mean_unique_frames": float(np.mean([r["num_frames"] for r in rows])),
                "mean_reader_forwards": float(
                    np.mean([2 * math.ceil(r["num_frames"] / 12) for r in rows])
                ),
            },
        },
        "private_case_diagnostics": case_diagnostics,
        "private_method_rows": dict(methods),
        "limits": [
            "Dense results pool separate fixed-reader calls, not joint all-frame attention or an upper bound.",
            "Privileged best uses ground truth and pays for all candidates; neither deployable nor a global bound.",
            "Eight candidate sets are uniform + three random + four local windows; no boxes enter this menu.",
            "Uniform samples the entire sequence without the older four-frame seeded start; do not mix protocols.",
            "Repeated seeds/frames do not increase independent case count; intervals cluster connected groups.",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    report = read_json(args.directory / "report.json")
    assert report["status"] == "complete" and report["frozen_reader_unchanged"]
    write_json(
        args.directory / "summary.json", summarize(read_json(args.directory / "development.json"))
    )
