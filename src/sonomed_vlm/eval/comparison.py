"""Compare saved generations only after verifying matched evaluation records."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from sonomed_vlm.eval.aggregate import aggregate_scores
from sonomed_vlm.eval.parsing import score_prediction

METRICS = {
    "strict_label_accuracy": "MCQ strict label accuracy",
    "option_text_accuracy": "MCQ option-text accuracy",
    "semantic_choice_accuracy": "MCQ semantic-choice accuracy",
    "contradiction_rate": "MCQ label/text contradiction rate (lower is better)",
    "invalid_response_rate": "MCQ invalid response rate (lower is better)",
    "token_f1": "QA + open-response token F1",
    "rouge_l_f1": "QA + open-response ROUGE-L",
    "valid_box_rate": "Grounding valid-box rate",
    "iou": "Grounding mean IoU",
    "localization_at_0_5": "Grounding Localization@0.5",
}
PAIRS = [
    ("qwen_finetuned", "qwen_original"),
    ("qwen_finetuned", "medgemma_finetuned"),
    ("qwen_original", "medgemma_original"),
]
MATCH_FIELDS = (
    "image_id", "prompt", "ground_truth", "answer_label", "options",
    "task_family", "task_type", "source_dataset", "focus", "generation_settings",
)


def compare_predictions(
    predictions: dict[str, list[dict[str, Any]]], manifest_ids: list[str]
) -> dict[str, Any]:
    expected = set(manifest_ids)
    if not expected or len(expected) != len(manifest_ids):
        raise ValueError("Manifest must contain unique, nonempty example IDs")
    scored = {}
    reference = None
    for name, rows in predictions.items():
        by_id = {row["example_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != expected:
            raise ValueError(f"{name}: duplicated, missing, or extra evaluation examples")
        ordered = [by_id[example_id] for example_id in manifest_ids]
        for index, row in enumerate(ordered):
            if reference is not None:
                for field in MATCH_FIELDS:
                    if row.get(field) != reference[index].get(field):
                        raise ValueError(f"{name}: unmatched {field} for {row['example_id']}")
            settings = row.get("generation_settings", {})
            if settings.get("do_sample") is not False or settings.get("max_new_tokens") != 256:
                raise ValueError(f"{name}: generation settings do not match the protocol")
            if "raw_model_output" not in row:
                raise ValueError(f"{name}: missing raw generation")
        reference = reference or ordered
        scored[name] = [{**row, "score": score_prediction(row)} for row in ordered]
    models = {}
    for name, rows in scored.items():
        aggregate = aggregate_scores(rows)
        counts = Counter(key for row in rows for key in METRICS if key in row["score"])
        if any(not math.isfinite(value) for value in aggregate["overall"].values()):
            raise ValueError(f"{name}: non-finite metric")
        models[name] = {**aggregate, "metric_counts": dict(counts)}
    comparisons = []
    for left, right in PAIRS:
        if left not in models or right not in models:
            continue
        differences = {}
        for metric in METRICS:
            values = [
                a["score"][metric] - b["score"][metric]
                for a, b in zip(scored[left], scored[right], strict=True)
                if metric in a["score"] and metric in b["score"]
            ]
            if values:
                differences[metric] = {
                    "difference": sum(values) / len(values),
                    "examples": len(values),
                    "left_higher": sum(value > 0 for value in values),
                    "equal": sum(value == 0 for value in values),
                    "right_higher": sum(value < 0 for value in values),
                }
        comparisons.append({"left": left, "right": right, "metrics": differences})
    return {"examples": len(expected), "models": models, "comparisons": comparisons}


def markdown_report(report: dict[str, Any]) -> str:
    names = list(report["models"])
    lines = [
        "# Qwen-MedGemma comparison", "",
        f"Matched held-out examples: {report['examples']:,}. All values below use the same scorer.",
        "Scores are on a 0-1 scale; higher is better except contradiction/invalid rates.", "",
        "| Metric | N | " + " | ".join(names) + " |",
        "|---|---:|" + "---:|" * len(names),
    ]
    first = report["models"][names[0]]
    for key, label in METRICS.items():
        values = [report["models"][name]["overall"].get(key) for name in names]
        cells = [f"{value:.4f}" if value is not None else "N/A" for value in values]
        lines.append(f"| {label} | {first['metric_counts'].get(key, 0)} | " + " | ".join(cells) + " |")
    for pair in report["comparisons"]:
        lines.extend(["", f"## {pair['left']} versus {pair['right']}", "",
                      "Differences are first model minus second model.", "",
                      "| Metric | Difference |", "|---|---:|"])
        for key, value in pair["metrics"].items():
            lines.append(f"| {METRICS[key]} | {value['difference']:+.4f} |")
    lines.extend([
        "", "## Interpretation limits", "",
        "Both adapted models use one epoch and matched training settings. This measures performance",
        "under that recipe, not each model's best achievable result. Architectures, native image",
        "processors, tokenizers, and adapter parameter counts differ. Qwen's image area is capped",
        "at 1,048,576 pixels. A shared 256-token output cap is not identical text length across tokenizers.",
        "The validation split was also used for training validation; this is not an external test set.",
        "Semantic-choice scores normalize formatting and must be read alongside strict-label accuracy",
        "and contradiction rate. Text overlap does not establish clinical correctness. These are",
        "single-seed results without confidence intervals or significance claims. Original MedGemma",
        "generations are reused and rescored; the JSON records source paths and SHA-256 checksums.", "",
        "Task routing follows explicit dataset labels: 4,905 MCQ, 4,628 QA/open, and 565 grounding",
        "examples. The 167 QA examples that carry option metadata remain QA. Historical summaries",
        "used differing denominators/parsers; this report recomputes every score from raw outputs.", "",
    ])
    return "\n".join(lines)
