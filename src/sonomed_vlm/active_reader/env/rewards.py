"""Private geometric proxy; incomplete, multi-lesion, constant cases are disabled."""

from collections import Counter

import numpy as np


def relevance(case):
    if any(f.annotation_status == "unannotated" or len(f.boxes_xyxy) > 1 for f in case.frames):
        return None, "incomplete_or_multiple_lesions"
    areas = [
        (
            (f.boxes_xyxy[0][2] - f.boxes_xyxy[0][0])
            * (f.boxes_xyxy[0][3] - f.boxes_xyxy[0][1])
            / (f.width * f.height)
        )
        if f.boxes_xyxy
        else 0.0
        for f in case.frames
    ]
    positive = [x for x in areas if x > 0]
    if not positive:
        return None, "no_positive_annotations"
    if max(areas) - min(areas) <= max(1e-8, np.mean(positive) * 0.01):
        return None, "near_constant_proxy"
    denominator = float(np.percentile(positive, 90)) + 1e-12
    return [min(1.0, x / denominator) for x in areas], "enabled"


def class_weights(training_cases, mode):
    if not training_cases or any(c.split != "train" for c in training_cases):
        raise ValueError("Reward frequencies require corrected training cases only")
    counts = Counter(c.label for c in training_cases)
    if set(counts) != {"benign", "malignant"}:
        raise ValueError("Training split needs both classes for balanced reward")
    return {
        k: len(training_cases) / (2 * count) if mode == "inverse_frequency_reward" else 1.0
        for k, count in counts.items()
    }
