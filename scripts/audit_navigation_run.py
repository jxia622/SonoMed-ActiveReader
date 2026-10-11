#!/usr/bin/env python3
"""Independently inspect saved navigation calls against source pixels and native tokens."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from PIL import Image
from tokenizers import Tokenizer

from sonomed_vlm.active_reader.data.manifest import load_manifest, source_path
from sonomed_vlm.active_reader.data.splits import attach_splits
from sonomed_vlm.active_reader.io import read_json, write_json


def audit(directory, manifest, root, tokenizer_path, splits):
    cases = {c.video_id: c for c in attach_splits(load_manifest(manifest), read_json(splits))}
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checks, errors, pixel_cache = Counter(), [], {}
    reader_prompts, reader_inputs = set(), {}
    reader_text_by_count = defaultdict(set)

    def require(condition, message):
        if not condition:
            errors.append(message)

    def pixels(case, index):
        key = (case.video_id, index)
        if key not in pixel_cache:
            with Image.open(source_path(root, case.frames[index].relative_path)) as image:
                pixel_cache[key] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        return pixel_cache[key]

    records = []
    for path in sorted(directory.glob("**/trajectories.json")):
        records.extend(read_json(path))
    for path in sorted(directory.glob("**/steps/*.json")):
        records.extend(read_json(path)["trajectories"])
    for row in records:
        case = cases[row["video_id"]]
        if case.split not in ("train", "dev") or row["split"] != case.split:
            errors.append("Recorded split differs from the sealed split manifest")
            # Never decode any pixels from a mislabeled or sealed-test episode.
            continue
        require(
            len(set(row["selected_indices"])) == min(row["budget"], case.num_frames),
            "Unequal final frame budget",
        )
        checks["episodes"] += 1
        for turn in row["turns"]:
            ctx = turn["context"]
            visible, observed = ctx["visible_indices"], ctx["observed_indices"]
            latest = ctx["returned_indices"]
            mode = ctx["history_mode"]
            expected = (
                observed
                if mode == "full"
                else latest
                if mode == "current_batch"
                else [latest[len(latest) // 2]]
            )
            require(visible == expected, "Visual history mask differs from its declared condition")
            require(
                turn["visible_image_indices"] == visible,
                "Logged processor images differ from observation",
            )
            require(
                len(turn["image_grid_thw"]) == len(visible) == turn["image_presentations"],
                "Actual image grid count mismatch",
            )
            require(
                turn["clean_image_sha256"] == [pixels(case, i) for i in visible],
                "Model input differs from selected clean source pixels",
            )
            text = tokenizer.decode(turn["input_ids"], skip_special_tokens=False)
            require(
                all(
                    s not in text
                    for s in (
                        case.video_id,
                        case.source_relative_path,
                        "reward_components",
                        "boxes_xyxy",
                    )
                ),
                "Private evaluator content in actual prompt tokens",
            )
            action = json.loads(turn["raw_output"])
            require(action in ctx["legal_actions"], "Sampled action was outside public support")
            require(
                action["action"] in ("inspect", "refine"),
                "Noninspection received navigation credit",
            )
            targets = turn["public_action_frame_sets"]
            require(
                len(targets) == len({tuple(t["new_indices"]) for t in targets}),
                "Duplicate frame-set action aliases",
            )
            chosen = next(t["new_indices"] for t in targets if t["action"] == action)
            require(
                set(turn["selected_after"]) - set(turn["selected_before"]) == set(chosen),
                "Inspection revealed an unexpected frame set",
            )
            if action["action"] == "refine":
                anchor = action["anchor_frame"]
                sign = -1 if action["direction"] == "left" else 1
                require(
                    anchor in observed and all(1 <= sign * (i - anchor) <= 8 for i in chosen),
                    "Refinement did not stay near a discovered anchor",
                )
                checks["refinement_actions"] += 1
            if turn.get("behavior_log_probs") is not None:
                mask = turn["loss_mask"]
                tokens = turn["generated_token_ids"]
                require(
                    mask == [0] * len(turn["input_ids"]) + [1] * len(tokens),
                    "Loss includes noninspection targets",
                )
                require(
                    all(
                        token in support
                        for token, support in zip(tokens, turn["token_constraints"], strict=True)
                    ),
                    "Generated token is outside its stored constrained support",
                )
                require(
                    len(tokens) == len(turn["behavior_log_probs"])
                    and all(
                        math.isfinite(logp) and logp <= 0 for logp in turn["behavior_log_probs"]
                    ),
                    "Invalid constrained behavior likelihoods",
                )
                require(
                    all(
                        logp == 0
                        for logp, support in zip(
                            turn["behavior_log_probs"], turn["token_constraints"], strict=True
                        )
                        if len(support) == 1
                    ),
                    "Forced tokens have nonzero constrained log probability",
                )
                require(
                    turn["private_reader_before"]["selected_indices"] == turn["selected_before"]
                    and turn["private_reader_after"]["selected_indices"] == turn["selected_after"],
                    "Reader credit is not aligned with inspection boundaries",
                )
                checks["training_inspections"] += 1
            checks["selector_calls"] += 1
            checks[f"calls_{mode}"] += 1
        final = row["reader"]
        require(
            final["selected_indices"] == row["selected_indices"],
            "Frozen reader saw a different frame set",
        )
        require(
            final["clean_image_sha256"] == [pixels(case, i) for i in row["selected_indices"]],
            "Reader clean pixels do not match selected source frames",
        )
        if row["private_reader_gain"] is not None:
            require(
                math.isclose(
                    sum(t["private_reader_logscore_increment"] for t in row["turns"]),
                    row["private_reader_gain"],
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                ),
                "Inspection score increments do not telescope to the terminal reader gain",
            )
        for read in row["reader_calls"]:
            reader_prompts.add(read["prompt_template_hash"])
            if "input_ids" in read:
                reader_text = tokenizer.decode(read["input_ids"], skip_special_tokens=True)
                reader_text_by_count[len(read["selected_indices"])].add(reader_text)
                require(
                    all(
                        forbidden not in reader_text
                        for forbidden in (
                            case.video_id,
                            case.source_relative_path,
                            "observed_indices",
                            "legal_actions",
                            "remaining_frames",
                            "boxes_xyxy",
                        )
                    ),
                    "Reader tokens include private metadata or browser bookkeeping",
                )
            require(
                read["prompt_template_hash"] == final["prompt_template_hash"],
                "Reader prompt changed across an episode",
            )
            require(
                len(read["image_grid_thw"]) == len(read["selected_indices"]),
                "Reader image grid count differs from the selected frame count",
            )
            require(
                read["clean_image_sha256"] == [pixels(case, i) for i in read["selected_indices"]],
                "Intermediate reader input differs from selected clean source pixels",
            )
            key = (tuple(read["selected_indices"]), tuple(read["clean_image_sha256"]))
            value = (read["input_ids_sha256"], read["probabilities"])
            if key in reader_inputs:
                require(
                    reader_inputs[key] == value,
                    "Identical evidence produced different reader prompts or scores",
                )
                checks["identical_evidence_reader_checks"] += 1
            else:
                reader_inputs[key] = value
    require(len(reader_prompts) == 1, "Reader prompt differs across methods or episodes")
    require(
        all(len(texts) == 1 for texts in reader_text_by_count.values()),
        "Actual reader text differs across methods at a matched frame count",
    )
    checks["reader_frame_counts_with_identical_actual_text"] = len(reader_text_by_count)
    result = {
        "status": "passed" if records and not errors else "failed",
        "checks": dict(checks),
        "errors": errors,
        "test_opened": False,
        "scope": "native token, source RGB, support, budget, history, and local-action audit; not a clinical sufficiency assessment",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.manifest, args.data_root, args.tokenizer, args.splits)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)
