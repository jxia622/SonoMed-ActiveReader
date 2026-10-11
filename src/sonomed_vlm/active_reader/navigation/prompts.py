"""Versioned selector and reader interfaces with distinct information boundaries."""

import json

from sonomed_vlm.active_reader.io import digest

SELECTOR_SYSTEM = """Select useful visual evidence from one prerecorded breast ultrasound sequence.
Another fixed reader will classify the selected frames. Your job is only to choose the next inspection.
Return exactly one JSON object from legal_actions. Never output a diagnosis or a stop action.
Every inspection reveals up to four NEW frames and the session ends automatically at the fixed frame budget.
An inspect action addresses one of 16 temporal center bins: coarse spans 20% of the sequence;
local covers four adjacent frames at that bin. A refine action explores the nearest unseen frames
within eight frame indices to the left or right of an already observed anchor_frame.
Use visible ultrasound evidence and previous inspection bookkeeping to decide where to inspect next.
Only images actually supplied in this call are visible. No unseen image content is available.
Images are chronological and labeled with their browser indices."""

READER_SYSTEM = """Classify the lesion in the supplied clean breast ultrasound frames as benign or malignant.
The images are selected frames from one prerecorded sequence, shown in chronological order.
Use the supplied images. Return exactly one JSON object: prediction is benign or malignant."""

SELECTOR_HASH = digest({"system": SELECTOR_SYSTEM, "version": 1})
READER_HASH = digest(
    {
        "system": READER_SYSTEM,
        "version": 2,
        "frame_labels": "ordinal only; original browser indices are private",
        "scoring": "two-complete-constrained-label-paths",
    }
)


def messages(observation):
    if observation.protocol == "reader":
        # Exact same text at matched budgets. Absolute browser positions could
        # otherwise convey selection strategy or sequence-length correlations.
        content = []
        for ordinal, image in enumerate(observation.images, 1):
            content += [
                {"type": "text", "text": f"Frame {ordinal}"},
                {"type": "image", "image": image},
            ]
        return [
            {"role": "system", "content": [{"type": "text", "text": READER_SYSTEM}]},
            {"role": "user", "content": content},
        ]
    elif observation.protocol == "navigation":
        metadata = {
            k: observation.public_dict()[k]
            for k in (
                "observed_indices",
                "returned_indices",
                "visible_indices",
                "total_frames",
                "remaining_frames",
                "legal_actions",
                "history",
            )
        }
        system = SELECTOR_SYSTEM
    else:
        raise ValueError("Unknown navigation protocol")
    content = [{"type": "text", "text": json.dumps(metadata, sort_keys=True)}]
    for index, image in zip(observation.image_indices, observation.images, strict=True):
        content += [
            {"type": "text", "text": f"Observed frame {index}"},
            {"type": "image", "image": image},
        ]
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]
