"""No hidden diagnosis, filename, case ID, annotation, or reward in this template."""

import json

from sonomed_vlm.active_reader.io import digest

SYSTEM = """You read selected frames from one prerecorded breast ultrasound sequence for a research benchmark.
Classify the lesion as benign or malignant using only the visible images.
Return exactly one JSON object, with no markdown or additional prose.
An inspect object has action=inspect, an integer center_bin, and scale=coarse or local.
A stop object has action=stop, prediction=benign or malignant, and an evidence_frames list.
Only choose an inspection listed in legal_inspections. Scales are coarse and local.
Cite one to three distinct observed frame indices when stopping. If no inspections are legal, stop.
Unseen frames are unknown. Prior action history is supplied with the current budget.
Images are in chronological order and labeled by their true browser indices."""
FINAL = "Read the provided frames and return only the stop JSON object. No further inspections are available."
PROMPT_HASH = digest({"system": SYSTEM, "final": FINAL, "version": 1})


def build_messages(observation, final_only=False):
    if observation.protocol != "joint":
        from sonomed_vlm.active_reader.navigation.prompts import messages

        return messages(observation)
    metadata = observation.public_dict()
    if final_only:
        metadata["legal_inspections"] = []
    content = [
        {
            "type": "text",
            "text": (FINAL + "\n" if final_only else "") + json.dumps(metadata, sort_keys=True),
        }
    ]
    for index, image in zip(observation.indices, observation.images, strict=True):
        content.extend(
            [{"type": "text", "text": f"Observed frame {index}"}, {"type": "image", "image": image}]
        )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": content},
    ]


def prompt_hash(observation):
    if observation.protocol == "joint":
        return PROMPT_HASH
    from sonomed_vlm.active_reader.navigation.prompts import READER_HASH, SELECTOR_HASH

    return READER_HASH if observation.protocol == "reader" else SELECTOR_HASH
