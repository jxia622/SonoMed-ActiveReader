#!/usr/bin/env python3
"""Group-held-out action-value probes: geometry, current pixels, and earlier pixels.

Only small ridge regressions are fitted; the Qwen reader and image encoder stay fixed.
Labels, boxes, unseen frame embeddings, and reader scores never enter predictors.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from sonomed_vlm.active_reader.io import digest, read_json, write_json
from sonomed_vlm.active_reader.navigation.browser import action_key

MODES = ("bookkeeping", "current_frame", "current_batch", "history")
ALPHAS = (0.01, 0.1, 1.0, 10.0)


def load_features(rows, directory):
    return {
        video: np.load(directory / f"{digest(video)}.npz")["features"].astype(float)
        for video in sorted({r["video_id"] for r in rows})
    }


def projection(fit, features):
    keys = sorted({(r["video_id"], i) for r in fit for i in r["context"]["observed_indices"]})
    x = np.stack([features[v][i] for v, i in keys])
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    mean = x.mean(0)
    _, _, v = np.linalg.svd(x - mean, full_matrices=False)
    return mean, v[:8].T


def visual_state(row, features, proj):
    c = row["context"]
    f = features[row["video_id"]]
    observed = c["observed_indices"]
    returned = c["returned_indices"]
    old = sorted(set(observed) - set(returned))
    assert len(old) == len(returned) == 4

    def get(indices):
        x = f[indices]
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        return (x - proj[0]) @ proj[1]

    def pooled(indices):
        x = get(indices)
        times = np.asarray(indices, dtype=float)
        times = (times - times.mean()) / max(1.0, np.ptp(times))
        return np.concatenate([x.mean(0), x.std(0), (x * times[:, None]).mean(0)])

    return {
        "current": get([returned[len(returned) // 2]])[0],
        "batch": pooled(returned),
        "older": pooled(old),
    }


def row_features(row, visual, mode):
    c = row["context"]
    n = c["total_frames"]
    obs = np.array(c["observed_indices"]) / (n - 1)
    ret = np.array(c["returned_indices"]) / (n - 1)
    current = ret[len(ret) // 2]
    state = np.concatenate([obs, ret, [np.log(n) / 6, current, c["remaining_frames"] / 12]])
    values = []
    for o in row["outcomes"]:
        new = np.array(o["new_indices"]) / (n - 1)
        action = o["action"]
        family = action.get("scale", action.get("direction"))
        a = np.concatenate(
            [
                new,
                [
                    new.mean(),
                    new.std(),
                    np.ptp(new),
                    new.mean() - current,
                    abs(new - current).min(),
                    abs(new[:, None] - obs[None, :]).min(1).mean(),
                ],
                [float(family == k) for k in ("coarse", "local", "left", "right")],
            ]
        )
        geometry = np.concatenate([a, a * a, np.outer(state, a).ravel()])
        if mode == "bookkeeping":
            values.append(geometry)
            continue
        pixels = [visual["current"]]
        if mode in ("current_batch", "history"):
            pixels.append(visual["batch"])
        if mode == "history":
            pixels.append(visual["older"])
        # Interactions let visual context change the ranking of candidate locations.
        pixels = np.concatenate(pixels)
        values.append(
            np.concatenate([geometry, np.outer(pixels, np.concatenate([[1.0], a])).ravel()])
        )
    x = np.stack(values)
    return x - x.mean(0, keepdims=True)


def design(rows, features, proj, mode, older_override=None):
    chunks = []
    for r in rows:
        v = visual_state(r, features, proj)
        if older_override is not None:
            v["older"] = older_override[r["state_id"]]
        chunks.append(row_features(r, v, mode))
    return np.concatenate(chunks)


def targets(rows):
    return np.concatenate(
        [
            np.array([o["log_score"] for o in r["outcomes"]])
            - np.mean([o["log_score"] for o in r["outcomes"]])
            for r in rows
        ]
    )


def fit_predict(x, y, z, alpha):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-6] = 1
    a = (x - mean) / scale
    b = (z - mean) / scale
    penalty = alpha * len(x)
    if a.shape[1] > len(x):
        weights = a.T @ np.linalg.solve(a @ a.T + penalty * np.eye(len(x)), y)
    else:
        weights = np.linalg.solve(a.T @ a + penalty * np.eye(a.shape[1]), a.T @ y)
    return b @ weights


def evaluate(rows, prediction):
    out = []
    offset = 0
    for r in rows:
        count = len(r["outcomes"])
        p = prediction[offset : offset + count]
        offset += count
        i = int(np.argmax(p))
        scores = np.array([o["log_score"] for o in r["outcomes"]])
        best = scores.max()
        out.append(
            {
                "state_id": r["state_id"],
                "video_id": r["video_id"],
                "group_id": r["group_id"],
                "selected": i,
                "selected_log_score": float(scores[i]),
                "regret": float(best - scores[i]),
                "best_action_match": float(scores[i] >= best - 1e-8),
                "selected_correct": float(r["outcomes"][i]["correct"]),
                "menu_gap": float(np.ptp(scores)),
            }
        )
    return out


def estimate(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r["group_id"]].append(
            [
                r[k]
                for k in ("selected_log_score", "regret", "best_action_match", "selected_correct")
            ]
        )
    x = np.stack([np.mean(v, axis=0) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(142)
    draws = x[rng.integers(len(x), size=(2000, len(x)))].mean(1)
    names = ("selected_log_score", "regret", "best_action_match", "selected_correct")
    return {
        "groups": len(x),
        "states": len(records),
        "mean": dict(zip(names, x.mean(0).tolist(), strict=True)),
        "group_bootstrap_95ci": dict(
            zip(names, np.quantile(draws, [0.025, 0.975], axis=0).T.tolist(), strict=True)
        ),
    }


def contrast(a, b):
    lookup = {r["state_id"]: r for r in b}
    rows = []
    for r in a:
        row = dict(r)
        for k in ("selected_log_score", "regret", "best_action_match", "selected_correct"):
            row[k] -= lookup[r["state_id"]][k]
        rows.append(row)
    return estimate(rows)


def run(args):
    report = read_json(args.directory / "report.json")
    assert report["status"] == "complete" and report["frozen_reader_unchanged"]
    rows = read_json(args.directory / "history.json")
    assert all(r["split"] == "train" for r in rows)
    fit = [r for r in rows if r["role"] == "fit"]
    val = [r for r in rows if r["role"] == "validation"]
    assert {r["group_id"] for r in fit}.isdisjoint(r["group_id"] for r in val)
    features = load_features(rows, args.embeddings)
    groups = sorted({r["group_id"] for r in fit}, key=lambda g: digest([142, g]))
    folds = {g: i % 4 for i, g in enumerate(groups)}
    cv = {mode: {str(a): [] for a in ALPHAS} for mode in MODES}
    for fold in range(4):
        train = [r for r in fit if folds[r["group_id"]] != fold]
        hold = [r for r in fit if folds[r["group_id"]] == fold]
        proj = projection(train, features)
        y = targets(train)
        for mode in MODES:
            x = design(train, features, proj, mode)
            z = design(hold, features, proj, mode)
            for alpha in ALPHAS:
                predictions = fit_predict(x, y, z, alpha)
                cv[mode][str(alpha)].extend(evaluate(hold, predictions))
        print(f"History probe inner fold {fold + 1}/4 complete", flush=True)
    results = {}
    predictions = {}
    chosen = {}
    projection_full = projection(fit, features)
    for mode in MODES:
        scores = {
            a: float(np.mean([r["selected_log_score"] for r in v])) for a, v in cv[mode].items()
        }
        alpha = float(max(scores, key=scores.get))
        chosen[mode] = {"alpha": alpha, "inner_cv_selected_log_scores": scores}
        x = design(fit, features, projection_full, mode)
        z = design(val, features, projection_full, mode)
        p = fit_predict(x, targets(fit), z, alpha)
        predictions[mode] = evaluate(val, p)
        results[mode] = estimate(predictions[mode])
    # Known geometric coverage and uniform-random-menu reference values.
    coverage = []
    uniform = []
    oracle = []
    for r in val:
        seen = set(r["context"]["observed_indices"])

        def score(o, seen=seen):
            return (
                sum(min(abs(i - j) for j in seen) for i in o["new_indices"]) / 4,
                max(o["new_indices"]) - min(o["new_indices"]),
                action_key(o["action"]),
            )

        index = max(range(len(r["outcomes"])), key=lambda i: score(r["outcomes"][i]))
        utility = np.array([o["log_score"] for o in r["outcomes"]])
        correct = np.array([o["correct"] for o in r["outcomes"]], dtype=float)
        for arr, s, c, m in (
            (
                coverage,
                utility[index],
                correct[index],
                float(utility[index] >= utility.max() - 1e-8),
            ),
            (
                uniform,
                utility.mean(),
                correct.mean(),
                float(np.mean(utility >= utility.max() - 1e-8)),
            ),
            (oracle, utility.max(), correct[int(utility.argmax())], 1.0),
        ):
            arr.append(
                {
                    "state_id": r["state_id"],
                    "video_id": r["video_id"],
                    "group_id": r["group_id"],
                    "selected_log_score": float(s),
                    "regret": float(utility.max() - s),
                    "best_action_match": m,
                    "selected_correct": float(c),
                    "menu_gap": float(np.ptp(utility)),
                }
            )
    for name, records in (
        ("coverage", coverage),
        ("uniform_random_menu", uniform),
        ("privileged_menu_best", oracle),
    ):
        predictions[name] = records
        results[name] = estimate(records)
    pairs = {
        f"{a}_minus_{b}": contrast(predictions[a], predictions[b])
        for a, b in (
            ("current_frame", "bookkeeping"),
            ("current_batch", "current_frame"),
            ("history", "current_batch"),
            ("history", "bookkeeping"),
            ("history", "uniform_random_menu"),
            ("history", "coverage"),
        )
    }
    # Destructive intervention changes only older visual features, leaving geometry/current pixels intact.
    base_visual = {r["state_id"]: visual_state(r, features, projection_full)["older"] for r in val}
    order = sorted({r["video_id"] for r in val})
    by = {(r["video_id"], r["seed"]): r for r in val}
    interventions = []
    x = design(fit, features, projection_full, "history")
    for shift in (1, 3, 7, 11, 17):
        donor = {v: order[(i + shift) % len(order)] for i, v in enumerate(order)}
        older = {
            r["state_id"]: base_visual[by[(donor[r["video_id"]], r["seed"])]["state_id"]]
            for r in val
        }
        z = design(val, features, projection_full, "history", older)
        p = fit_predict(x, targets(fit), z, chosen["history"]["alpha"])
        evaluated = evaluate(val, p)
        interventions.append(
            {
                "shift": shift,
                "changed_actions": sum(
                    a["selected"] != b["selected"]
                    for a, b in zip(evaluated, predictions["history"], strict=True)
                ),
                "intact_minus_shuffled_older": contrast(predictions["history"], evaluated),
            }
        )
    output = {
        "test_opened": False,
        "fit_groups": len(groups),
        "validation_groups": len(order),
        "states": len(rows),
        "encoder": "frozen generic ResNet18; PCA8 fitted only on actually observed fit-group frames",
        "features": "public past/current indices, frame count, remaining budget, proposed index geometry; optional observed pixel embeddings with action interactions",
        "target": "within-state centered frozen-reader true-label log score; labels/rewards excluded from features",
        "selection": "four inner folds by connected group choose ridge penalty on fit only; untouched validation evaluates once",
        "chosen_penalties": chosen,
        "validation": results,
        "paired_contrasts": pairs,
        "older_feature_interventions": interventions,
        "validation_states_with_menu_gap_ge_01": sum(r["menu_gap"] >= 0.1 for r in uniform),
        "private_predictions": predictions,
        "limitations": [
            "Small fixed-state probe, not causal proof or a deployed policy benchmark.",
            "Failure of this generic embedding/ridge probe does not prove absence of all useful visual history.",
            "Only one later decision per start and eight restricted candidate actions; 32 independent validation groups.",
            "No multiple-comparison correction; destructive history shuffles are out of distribution.",
        ],
    }
    write_json(args.directory / "history_predictability.json", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    run(p.parse_args())
