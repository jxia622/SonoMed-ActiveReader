"""Development-only matched-budget comparisons, paired at connected-group level."""

from collections import defaultdict

import numpy as np

from sonomed_vlm.active_reader.evaluation.metrics import evaluate, summarize
from sonomed_vlm.active_reader.io import digest, read_json, write_json
from sonomed_vlm.active_reader.navigation.rollout import HISTORIES, NONADAPTIVE, select_episode


def paired(rows, reference="uniform", bootstrap=1000):
    by = {(r["video_id"], r["seed"], r["budget"], r["method"]): r for r in rows}
    results = []
    for method, budget in sorted({(r["method"], r["budget"]) for r in rows}):
        if method == reference:
            continue
        pairs = [
            (r, by[(r["video_id"], r["seed"], budget, reference)])
            for r in rows
            if r["method"] == method
            and r["budget"] == budget
            and (r["video_id"], r["seed"], budget, reference) in by
        ]
        if not pairs:
            continue
        groups = defaultdict(list)
        for a, b in pairs:
            if (
                a["initial_indices"] != b["initial_indices"]
                or a["unique_frames"] != b["unique_frames"]
            ):
                raise ValueError("Unmatched initial observations or frame budgets")
            groups[a["group_id"]].append((a, b))

        def differences(items):
            left, right = zip(*items, strict=True)
            la, lb = summarize(left), summarize(right)
            return {
                "balanced_accuracy": (
                    la["balanced_accuracy"] - lb["balanced_accuracy"]
                    if la["balanced_accuracy"] is not None and lb["balanced_accuracy"] is not None
                    else None
                ),
                "accuracy": la["accuracy"] - lb["accuracy"],
                "true_label_log_score": float(
                    np.mean(
                        [
                            a["reader_true_label_log_score"] - b["reader_true_label_log_score"]
                            for a, b in items
                        ]
                    )
                ),
            }

        estimates = defaultdict(list)
        keys = sorted(groups)
        rng = np.random.default_rng(42)
        for _ in range(bootstrap):
            sample = [
                pair for key in rng.choice(keys, len(keys), replace=True) for pair in groups[key]
            ]
            for metric, value in differences(sample).items():
                if value is not None:
                    estimates[metric].append(value)
        results.append(
            {
                "method": method,
                "reference": reference,
                "budget": budget,
                "episodes": len(pairs),
                "groups": len(groups),
                "delta": differences(pairs),
                "paired_group_bootstrap_95ci": {
                    k: np.quantile(v, [0.025, 0.975]).tolist()
                    for k, v in estimates.items()
                    if len(groups) > 1
                },
                "prediction_disagreements": sum(
                    a["prediction"] != b["prediction"] for a, b in pairs
                ),
            }
        )
    return results


def run_evaluation(selector, reader, cases, root, config, output, provenance, methods=None):
    if not cases or any(c.split != "dev" for c in cases):
        raise ValueError("Navigation milestone permits development evaluation only; test is sealed")
    methods = methods or [
        "uniform",
        "random",
        "full",
        "current_batch",
        "current_frame",
        "privileged_area",
    ]
    if set(methods) - set(HISTORIES + NONADAPTIVE):
        raise ValueError("Unknown navigation method")
    output.mkdir(parents=True, exist_ok=True)
    signature = digest(
        {
            "provenance": provenance,
            "methods": methods,
            "budgets": config.evaluation.budgets,
            "seeds": config.evaluation.seeds,
            "cases": [c.video_id for c in cases],
        }
    )
    meta = output / "provenance.json"
    saved = {"signature": signature, "provenance": provenance, "methods": methods}
    if meta.exists() and read_json(meta) != saved:
        raise ValueError("Navigation output belongs to a different experiment")
    write_json(meta, saved)
    rows = []
    for case in cases:
        for budget in config.evaluation.budgets:
            for seed in config.evaluation.seeds:
                for method in methods:
                    path = (
                        output
                        / "episodes"
                        / (digest([signature, case.video_id, budget, seed, method]) + ".json")
                    )
                    if path.exists():
                        record = read_json(path)
                    else:
                        record = select_episode(
                            case, root, config, selector, reader, method, seed, budget
                        ).record
                        write_json(path, record)
                        print(
                            f"navigation {method} budget={budget} completed={len(rows) + 1}",
                            flush=True,
                        )
                    rows.append(record)
    report = {
        **evaluate(rows, config.evaluation.bootstrap_samples),
        "split": "dev",
        "test_opened": False,
        "provenance": provenance,
        "paired_vs_uniform": paired(rows, bootstrap=config.evaluation.bootstrap_samples),
        "paired_vs_current_frame": paired(
            rows, "current_frame", config.evaluation.bootstrap_samples
        ),
        "paired_vs_current_batch": paired(
            rows, "current_batch", config.evaluation.bootstrap_samples
        ),
        "score_interpretation": "fixed-reader forced-choice log scores; not calibrated probabilities",
    }
    write_json(output / "trajectories.json", rows)
    write_json(output / "report.json", report)
    return report
