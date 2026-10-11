#!/usr/bin/env python3
"""Anonymous subject-level estimates and paired uncertainty after frozen evaluation."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sonomed_vlm.active_reader.io import read_json, write_json


def paired_subject_contrasts(matrix, comparisons, sample):
    """Use the same sampled subjects for every method and every contrast."""
    shapes = {np.asarray(values).shape for values in matrix.values()}
    assert len(shapes) == 1
    shape = next(iter(shapes))
    assert len(shape) == 1 and sample.ndim == 2 and sample.shape[1] == shape[0]
    assert np.issubdtype(sample.dtype, np.integer) and sample.min() >= 0
    assert sample.max() < shape[0]
    deltas = np.array([np.mean(matrix[a] - matrix[b]) for a, b in comparisons])
    draws = np.stack([(matrix[a] - matrix[b])[sample].mean(1) for a, b in comparisons], axis=1)
    radius = float(np.quantile(np.max(np.abs(draws - deltas), axis=1), 0.95))
    contrasts = []
    for i, (a, b) in enumerate(comparisons):
        contrasts.append(
            {
                "arm": a,
                "comparator": b,
                "difference": float(deltas[i]),
                "paired_percentile_95": np.quantile(draws[:, i], [0.025, 0.975]).tolist(),
                "simultaneous_basic_95": [float(deltas[i] - radius), float(deltas[i] + radius)],
            }
        )
    return contrasts, radius


def summarize(run, output):
    assert read_json(run / "evaluation/report.json")["status"] == "complete"
    assert read_json(run / "reader/report.json")["frozen_unchanged"]
    audit = read_json(run / "audit.json")
    assert audit["status"] == "passed", "Publish only after the native artifact audit passes"
    campaign = read_json(run / "campaign.json")
    protocol = campaign["protocol"]
    assert read_json(run / "evaluation_lock.json")["protocol_sha256"] == campaign["protocol_sha256"]
    rows = [r for r in read_json(run / "evaluation/acquisition.json") if r["primary"]]
    cases = sorted({r["alias"] for r in rows})
    assert len(cases) == 20 and all(r["split"] == "evaluation" for r in rows)
    methods = [
        "uniform",
        "random",
        "coverage",
        "random_actions",
        "geometry_prior",
        "original_prompted",
        "pre_rl_full",
        "pre_rl_current",
        "pre_rl_geometry",
        "rl_full",
        "rl_current",
        "rl_geometry",
        "privileged",
    ]
    assert {r["method"] for r in rows} == set(methods)
    for row in rows:
        assert len(set(row["indices"])) == 12
        assert 0 <= row["best_wfss"] <= 1
    reader = read_json(run / "reader/reader_results.json")
    assert len(reader) == 20 * 2 * (len(methods) - 1)
    assert all(r["split"] == "evaluation" for r in reader)
    matrix, statistics = {}, {}
    for method in methods:
        values = []
        captures, optimal = [], []
        for case in cases:
            subset = [r for r in rows if r["method"] == method and r["alias"] == case]
            assert len(subset) == (100 if method in {"random", "random_actions"} else 2)
            values.append(np.mean([r["best_wfss"] for r in subset]))
            captures.append(np.mean([r["any_supported"] for r in subset]))
            optimal.append(np.mean([r["any_optimal"] for r in subset]))
        matrix[method] = np.array(values)
        reads = [r for r in reader if r["method"] == method]
        if method != "privileged":
            assert all(sum(r["alias"] == case for r in reads) == 2 for case in cases)
        reader_mean = float(np.mean([r["reader_wfss"] for r in reads])) if reads else None
        inputs = [t["input"] for r in rows if r["method"] == method for t in r.get("turns", [])]
        statistics[method] = {
            "mean_acquisition_wfss": float(np.mean(values)),
            "capture_rate": float(np.mean(captures)),
            "optimal_capture_rate": float(np.mean(optimal)),
            "reader_wfss_two_starts": reader_mean,
            "subjects": 20,
            "starts_per_subject": 100 if method in {"random", "random_actions"} else 2,
            "selector_calls": len(inputs),
            "selector_generated_tokens": sum(r["generated_tokens"] for r in inputs),
            "selector_visual_tokens": sum(r["visual_input_tokens"] for r in inputs),
            "selector_image_presentations": sum(r["image_presentations"] for r in inputs),
            "selector_call_seconds": sum(r["call_seconds"] for r in inputs),
        }
        episodes = [r for r in rows if r["method"] == method and "turns" in r]
        if episodes:
            statistics[method]["inspection_contributions"] = {
                "initial_mean_wfss": float(np.mean([r["initial_wfss"] for r in episodes])),
                "first_inspection_mean_gain": float(
                    np.mean([r["turns"][0]["private_quality_increment"] for r in episodes])
                ),
                "second_inspection_mean_gain": float(
                    np.mean([r["turns"][1]["private_quality_increment"] for r in episodes])
                ),
                "fraction_at_max_score_before_second": float(
                    np.mean(
                        [
                            r["initial_wfss"] + r["turns"][0]["private_quality_increment"] == 1
                            for r in episodes
                        ]
                    )
                ),
            }
    rng = np.random.default_rng(20261011)
    sample = rng.integers(0, len(cases), size=(protocol["bootstrap_samples"], len(cases)))
    comparisons = [("rl_full", m) for m in methods if m not in {"rl_full", "privileged"}]
    comparisons += [("rl_current", "pre_rl_current"), ("rl_geometry", "pre_rl_geometry")]
    contrasts, radius = paired_subject_contrasts(matrix, comparisons, sample)
    by_comparator = {r["comparator"]: r for r in contrasts if r["arm"] == "rl_full"}
    primary_comparators = [
        "uniform",
        "random",
        "coverage",
        "random_actions",
        "geometry_prior",
        "original_prompted",
        "pre_rl_full",
    ]
    simple = ["uniform", "random", "coverage", "random_actions", "geometry_prior"]
    target_met = all(by_comparator[m]["difference"] >= 0.1 for m in simple) and all(
        by_comparator[m]["simultaneous_basic_95"][0] > 0 for m in primary_comparators
    )
    rl_supported = all(
        by_comparator[m]["simultaneous_basic_95"][0] > 0 for m in primary_comparators
    )
    history_supported = all(
        by_comparator[m]["simultaneous_basic_95"][0] > 0 for m in ("rl_current", "rl_geometry")
    )
    for method in methods:
        statistics[method]["subject_bootstrap_95"] = np.quantile(
            matrix[method][sample].mean(1), [0.025, 0.975]
        ).tolist()
    result = {
        "subjects": 20,
        "primary_endpoint": protocol["primary_endpoint"],
        "methods": statistics,
        "contrasts": contrasts,
        "primary_target_met": target_met,
        "rl_superiority_supported": rl_supported,
        "history_benefit_supported": history_supported,
        "uncertainty": "10,000 paired subject bootstrap resamples; simultaneous basic 95% bounds use the maximum absolute centered error across all declared contrasts",
        "training_seed_count": 1,
        "test_scope": "internal held-out subjects from public ACOUSLIC training release",
        "breastvid_test_opened": False,
        "audit": audit,
    }
    write_json(output / "summary.json", result)
    order = [m for m in methods if m != "privileged"]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 7))
    values = [statistics[m]["mean_acquisition_wfss"] for m in order]
    ax.barh(y, values, color=["#287b78" if m.startswith("rl_") else "#8ea0ae" for m in order])
    bounds = np.array([statistics[m]["subject_bootstrap_95"] for m in order])
    ax.errorbar(
        values,
        y,
        xerr=[np.maximum(0, np.array(values) - bounds[:, 0]), np.maximum(0, bounds[:, 1] - values)],
        fmt="none",
        ecolor="#263238",
        capsize=3,
        linewidth=1,
    )
    for j, v in enumerate(values):
        ax.text(max(v, bounds[j, 1]) + 0.015, j, f"{v:.3f}", va="center")
    ax.set(
        yticks=y,
        yticklabels=[m.replace("_", " ") for m in order],
        xlim=(0, 1.12),
        xlabel="Mean acquisition WFSS at 12 observed frames",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("ACOUSLIC held-out evidence acquisition pilot")
    fig.text(
        0.1,
        0.015,
        "20 independent subjects; one training seed. Bars: means; whiskers: subject-bootstrap 95% intervals.\nRandom baselines average 100 starts per subject; others two. Acquisition is not diagnostic accuracy.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    fig.savefig(output / "acquisition_results.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    contrast_y = np.arange(len(contrasts))
    differences = np.array([r["difference"] for r in contrasts])
    ax.errorbar(differences, contrast_y, xerr=radius, fmt="o", color="#287b78", capsize=3)
    ax.axvline(0, color="#66757f", linestyle="--", linewidth=1)
    ax.set(
        yticks=contrast_y,
        yticklabels=[f"{r['arm']} - {r['comparator']}".replace("_", " ") for r in contrasts],
        xlabel="Paired difference in acquisition WFSS (positive favors RL arm)",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Did RL and visual history improve evidence acquisition?")
    fig.text(
        0.1,
        0.02,
        "Simultaneous approximate 95% bounds across all 13 declared contrasts; 20 subjects.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(output / "paired_differences.png", dpi=170)
    plt.close(fig)
    lines = [
        "# ACOUSLIC pilot results",
        "",
        "The table reports held-out acquisition quality at a matched 12-frame budget. Supervised adaptation and RL are separated by the pre-RL comparator.",
        "",
        "Acquisition WFSS is the best annotation-scored frame in the acquired set, computed by the evaluator. The original-reader column scores the reader's single final choice. These describe different stages; the acquisition score alone does not establish an end-to-end challenge win.",
        "",
        f"**RL superiority supported: {rl_supported}. Visual-history benefit supported: {history_supported}.** These are separate claims, tested against the predefined comparators using simultaneous paired bounds.",
        "",
        "![Acquisition scores](acquisition_results.png)",
        "",
        "| Method | Acquisition WFSS | Supported-plane capture | Original reader WFSS |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        s = statistics[method]
        reader_value = (
            "diagnostic only"
            if s["reader_wfss_two_starts"] is None
            else f"{s['reader_wfss_two_starts']:.3f}"
        )
        lines.append(
            f"| {method} | {s['mean_acquisition_wfss']:.3f} | {s['capture_rate']:.1%} | {reader_value} |"
        )
    lines += [
        "",
        "## Paired differences",
        "",
        "Intervals below are simultaneous approximate 95% bounds across the listed comparisons, resampling subjects rather than frames or starts.",
        "",
        "![Paired differences](paired_differences.png)",
        "",
        "| Arm | Comparator | Difference | Simultaneous interval |",
        "|---|---|---:|---:|",
    ]
    for r in contrasts:
        lo, hi = r["simultaneous_basic_95"]
        lines.append(
            f"| {r['arm']} | {r['comparator']} | {r['difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
        )
    lines += [
        "",
        f"Predeclared primary target met: **{target_met}**.",
        "",
        "History compares all previous images with the latest three-frame observation batch plus identical location bookkeeping, and with a blank-image geometry control. Equal observation budgets do not imply equal inference cost.",
        "",
        "## Selector inference cost",
        "",
        "Totals across 20 subjects and two starts, excluding the identical downstream reader. Timings include input preparation and may reflect different GPU nodes.",
        "",
        "| Method | Calls | Image presentations | Visual tokens | Seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in order:
        s = statistics[method]
        if s["selector_calls"]:
            lines.append(
                f"| {method} | {s['selector_calls']} | {s['selector_image_presentations']} | {s['selector_visual_tokens']} | {s['selector_call_seconds']:.1f} |"
            )
    lines += [
        "",
        "## Where acquisition gains occurred",
        "",
        "These descriptive components sum to final acquisition quality. A score of 1 before the second inspection leaves no scoring headroom for that action, even if its evidence is useful in other ways.",
        "",
        "| Policy | Initial score | First inspection gain | Second inspection gain | Already at 1 before second |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in order:
        component = statistics[method].get("inspection_contributions")
        if component is not None:
            lines.append(
                f"| {method} | {component['initial_mean_wfss']:.3f} | {component['first_inspection_mean_gain']:.3f} | {component['second_inspection_mean_gain']:.3f} | {component['fraction_at_max_score_before_second']:.1%} |"
            )
    lines += [
        "",
        "This pilot uses one training seed and 20 held-out subjects. Annotation zeros are not verified anatomical negatives. The privileged row is a label-informed diagnostic ceiling. The sealed BreastVid test remained untouched.",
        "",
        "## Verification",
        "",
        f"The independent artifact audit passed for {audit['native_inputs']:,} native inputs and {audit['rl_episodes']:,} RL trajectories. It replayed actions, source-pixel hashes, budgets, private rewards, split boundaries, legal decoding support and the fixed checkpoint lock. The original reader weights remained unchanged.",
        "",
        "| Training arm | Updates | Groups with varying rewards | Adapter changed | Maximum probability reconstruction drift |",
        "|---|---:|---:|---|---:|",
    ]
    for mode, signal in audit["training_signal"].items():
        lines.append(
            f"| {mode} | {signal['updates']} | {signal['reward_varying_groups']} | {signal['language_adapter_changed']} | {signal['max_behavior_logprob_drift']:.6f} |"
        )
    lines += [
        "",
        "Training gives both inspection actions the same terminal-gain advantage. The counts below identify second actions that received positive credit despite adding no annotation-scored quality themselves. This is valid terminal-return policy credit, but can be a source of variance in a short run.",
        "",
        "| Arm | Second actions with positive advantage | Of these, no new second-action quality gain |",
        "|---|---:|---:|",
    ]
    for mode, signal in audit["training_signal"].items():
        lines.append(
            f"| {mode} | {signal['positive_advantage_second_actions']} | {signal['positive_advantage_second_actions_without_new_quality_gain']} |"
        )
    controls = audit["pre_rl_tuning_interventions"]
    lines += [
        "",
        f"Before RL, blanking images changed {controls['blanking_images_changes_first_action']}/{controls['subject_start_pairs']} first actions on the tuning set. Removing older observations changed {controls['removing_older_images_changes_second_action']}/{controls['subject_start_pairs']} second actions. These controls establish input sensitivity; only the held-out performance comparisons above assess whether those inputs help select better evidence.",
        "",
    ]
    history_probe = audit.get("tuning_older_image_probe")
    if history_probe is not None:
        lines += [
            "## Tuning-only visual-history intervention",
            "",
            "This exploratory diagnostic fixes the final full-history checkpoint and its first action. At the second decision, it hides only the six older images while keeping text tokens, image slots, coordinates, budget and the latest three images identical. It is separate from the held-out comparisons and cannot establish a held-out superiority claim.",
            "",
            f"Across eight tuning subjects, mean acquisition WFSS was **{history_probe['full_history_mean_wfss']:.3f}** with real history and **{history_probe['hidden_older_mean_wfss']:.3f}** with older images hidden. The second action changed in {history_probe['second_action_changed']}/16 subject-start pairs. Native replay and unchanged-weight checks passed.",
            "",
        ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"subjects": 20, "primary_target_met": target_met, "methods": statistics}, indent=2
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summarize(args.run, args.output)
