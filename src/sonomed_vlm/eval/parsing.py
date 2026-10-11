"""Task-aware prediction parsing."""

from __future__ import annotations

from typing import Any

from sonomed_vlm.eval.classification import mcq_score
from sonomed_vlm.eval.generation import exact_match, rouge_l_f1, token_f1
from sonomed_vlm.eval.grounding import box_iou, parse_box


def score_prediction(record: dict[str, Any]) -> dict[str, Any]:
    raw = str(record.get("raw_model_output") or record.get("prediction") or "")
    reference = str(record.get("ground_truth") or "")
    task_type = str(record.get("task_type") or "").casefold()
    task_family = str(record.get("task_family") or "").casefold()
    options = list(record.get("options") or [])
    # Some QA records retain choice metadata. An explicit task label takes
    # precedence; only infer MCQ from options when no task type was supplied.
    if task_type in {"mcq", "multiple_choice", "multiple-choice"} or (not task_type and options):
        return mcq_score(raw, record.get("answer_label"), options)
    if "ground" in task_type or task_family in {"vg", "sonogrounding"}:
        prediction_box = parse_box(raw, coordinate_max=1000.0)
        reference_box = parse_box(reference, coordinate_max=1000.0)
        if not prediction_box.valid or not reference_box.valid:
            return {
                "parsed_prediction": prediction_box.coordinates,
                "valid_box_rate": 0.0,
                "invalid_box_rate": float(not prediction_box.valid),
                "iou": 0.0,
                "localization_at_0_5": 0.0,
            }
        iou = box_iou(prediction_box.coordinates, reference_box.coordinates)
        return {
            "parsed_prediction": prediction_box.coordinates,
            "valid_box_rate": 1.0,
            "invalid_box_rate": 0.0,
            "iou": iou,
            "localization_at_0_5": float(iou >= 0.5),
        }
    return {
        "parsed_prediction": raw.strip(),
        "normalized_exact_match": exact_match(raw, reference),
        "token_f1": token_f1(raw, reference),
        "rouge_l_f1": rouge_l_f1(raw, reference),
    }
