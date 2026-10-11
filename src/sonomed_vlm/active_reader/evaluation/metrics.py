"""Classification/cost curves with group bootstrap; invalid outputs are failures."""

from collections import defaultdict

import numpy as np


def summarize(rows):
    if not rows:
        raise ValueError("Cannot evaluate empty trajectories")
    confusion = {
        truth: {pred: 0 for pred in ("benign", "malignant", "invalid")}
        for truth in ("benign", "malignant")
    }
    for row in rows:
        pred = row.get("prediction")
        confusion[row["label"]][pred if pred in ("benign", "malignant") else "invalid"] += 1
    b, m = confusion["benign"], confusion["malignant"]
    specificity = b["benign"] / sum(b.values()) if sum(b.values()) else None
    sensitivity = m["malignant"] / sum(m.values()) if sum(m.values()) else None

    def mean_available(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "episodes": len(rows),
        "videos": len({r["video_id"] for r in rows}),
        "groups": len({r["group_id"] for r in rows}),
        "confusion_matrix": confusion,
        "accuracy": (b["benign"] + m["malignant"]) / len(rows),
        "balanced_accuracy": (specificity + sensitivity) / 2
        if specificity is not None and sensitivity is not None
        else None,
        "malignant_sensitivity": sensitivity,
        "benign_specificity": specificity,
        "invalid_terminal_rate": sum(
            r.get("prediction") not in ("benign", "malignant") for r in rows
        )
        / len(rows),
        **{
            f"mean_{key}": mean_available(key)
            for key in (
                "unique_frames",
                "fraction_viewed",
                "inspection_calls",
                "decision_turns",
                "model_calls",
                "image_presentations",
                "visual_input_tokens",
                "generated_tokens",
                "latency_seconds",
                "generation_seconds",
                "peak_memory_bytes",
                "revisits",
                "max_relevance",
                "mean_relevance",
            )
        },
    }


def evaluate(rows, bootstrap_samples=1000, seed=42):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["method"], row["budget"])].append(row)
    output = []
    for (method, budget), bucket in sorted(buckets.items()):
        result = {"method": method, "budget": budget, **summarize(bucket)}
        groups = defaultdict(list)
        for row in bucket:
            groups[row["group_id"]].append(row)
        keys = sorted(groups)
        rng = np.random.default_rng(seed)
        estimates = defaultdict(list)
        for _ in range(bootstrap_samples):
            resampled = [
                row for key in rng.choice(keys, len(keys), replace=True) for row in groups[key]
            ]
            values = summarize(resampled)
            for key in (
                "accuracy",
                "balanced_accuracy",
                "malignant_sensitivity",
                "benign_specificity",
            ):
                if values[key] is not None:
                    estimates[key].append(values[key])
        result["group_bootstrap_95ci"] = {
            key: np.quantile(values, [0.025, 0.975]).tolist()
            for key, values in estimates.items()
            if len(keys) > 1
        }
        output.append(result)
    return {
        "curves": output,
        "uncertainty_unit": "connected group, all videos and repeated seeds resampled together",
        "score": "categorical prediction; ROC-AUC and calibrated probabilities not claimed",
    }
