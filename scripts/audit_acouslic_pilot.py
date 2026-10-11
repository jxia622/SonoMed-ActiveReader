#!/usr/bin/env python3
"""Replay native inputs, split boundaries, actions and rewards after a pilot completes."""

import argparse
import hashlib
import random
import string
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoProcessor

from sonomed_vlm.active_reader.acouslic.data import Cases
from sonomed_vlm.active_reader.acouslic.environment import (
    coverage,
    initial,
    inspect,
    legal_centers,
    outcome,
    prior_action,
    uniform,
)
from sonomed_vlm.active_reader.acouslic.policy import READER_QUESTION, inspection_question
from sonomed_vlm.active_reader.io import file_hash, read_json, write_json


def audit(data_root, run, output, development_only=False):
    if development_only:
        assert not (run / "evaluation").exists(), "Development audit must precede held-out opening"
    else:
        assert read_json(run / "evaluation/report.json")["status"] == "complete"
        assert read_json(run / "reader/report.json")["frozen_unchanged"]
    campaign = read_json(run / "campaign.json")
    protocol = campaign["protocol"]
    assert file_hash(data_root / "partition.json") == campaign["data_partition_sha256"]
    lock = read_json(
        run / ("development_checkpoint_lock.json" if development_only else "evaluation_lock.json")
    )
    assert lock["protocol_sha256"] == campaign["protocol_sha256"]
    data = Cases(data_root, allow_evaluation=not development_only)
    train = {r["alias"] for r in data.subset("train")}
    tune = {r["alias"] for r in data.subset("tune")}
    heldout = set() if development_only else {r["alias"] for r in data.subset("evaluation")}
    assert len(train) == 32 and len(tune) == 8
    assert len(heldout) == (0 if development_only else 20)
    assert not any(data.by_alias[a]["screening_exposed"] for a in heldout)
    tokenizer = Tokenizer.from_file(str(run / "warm/checkpoint/processor/tokenizer.json"))
    processor = AutoProcessor.from_pretrained(
        run / "warm/checkpoint/processor", local_files_only=True
    )
    cache = {}
    processed_roundtrips = set()
    blank_hash = hashlib.sha256(Image.new("RGB", (744, 562)).tobytes()).hexdigest()
    checked_inputs = 0

    def native(record, expected_hidden=()):
        nonlocal checked_inputs
        alias, indices = record["alias"], record["image_indices"]
        assert record.get("hidden_older_indices", []) == list(expected_hidden)
        assert alias in data.by_alias and len(indices) == len(set(indices))
        assert len(indices) == len(record["clean_images"]) == len(record["image_grid_thw"])
        labels = list(string.ascii_uppercase[: len(indices)])
        if record["context"]["protocol"] == "acouslic_anchored_actions_probe_v1":
            labels = [
                string.ascii_uppercase[record["context"]["observed"].index(i)] for i in indices
            ]
            assert record["displayed_image_labels"] == labels
        native_text = "<|im_start|>user\n"
        for j, (index, im, grid) in enumerate(
            zip(indices, record["clean_images"], record["image_grid_thw"], strict=True)
        ):
            key = (alias, index)
            if key not in cache:
                path = data_root / "cases" / alias / "frames" / f"{index:04d}.png"
                assert file_hash(path) == data.admission(alias)["frames"][str(index)]
                with Image.open(path) as image:
                    assert not image.info and image.size == (744, 562)
                    cache[key] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            assert im["index"] == index and im["source_rgb_sha256"] == cache[key]
            assert im["presented_rgb_sha256"] == (
                blank_hash if record["blank_images"] or index in expected_hidden else cache[key]
            )
            assert grid[0] == 1 and grid[1] % 2 == grid[2] % 2 == 0
            native_text += f"Image {labels[j]}:<|vision_start|>"
            native_text += "<|image_pad|>" * (int(np.prod(grid)) // 4) + "<|vision_end|>"
        native_text += record["question"] + "<|im_end|>\n<|im_start|>assistant\n"
        assert tokenizer.encode(native_text, add_special_tokens=False).ids == record["input_ids"]
        texts = record["legal_choices"]
        sequences = [tokenizer.encode(x, add_special_tokens=False).ids for x in texts]
        tokens = record["generated_token_ids"]
        assert tokens == tokenizer.encode(record["raw_output"], add_special_tokens=False).ids
        assert tokens in sequences
        for position, support in enumerate(record["token_constraints"]):
            expected = sorted({s[position] for s in sequences if s[:position] == tokens[:position]})
            assert support == expected and tokens[position] in support
        assert len(tokens) == record["generated_tokens"] and record["deterministic_termination"]
        if record["behavior_log_probs"] is not None:
            assert len(record["behavior_log_probs"]) == len(tokens)
            assert np.isfinite(record["behavior_log_probs"]).all()
        context = record["context"]
        if context["protocol"] == "acouslic_navigation":
            observed, latest, mode = context["observed"], context["latest"], context["mode"]
            assert indices == (list(latest) if mode == "current" else sorted(observed))
            assert record["blank_images"] == (mode == "geometry")
            assert context["budget_remaining"] == 12 - len(observed)
            assert record["question"] == inspection_question(indices, observed)
            assert texts == [f"{i:03d}" for i in range(840) if i not in observed]
            assert len(tokens) == 3
        elif context["protocol"] == "acouslic_anchored_actions_probe_v1":
            from probe_acouslic_anchored_actions import actions

            observed, latest, mode = context["observed"], context["latest"], context["mode"]
            assert indices == (list(latest) if mode == "current" else observed)
            assert context["budget_remaining"] == 12 - len(observed)
            assert record["blank_images"] == (mode == "geometry")
            mapping = actions(observed)
            assert record["action_mapping"] == mapping and texts == list(mapping)
            assert set(mapping.values()) == set(legal_centers(observed))
            assert record["action_center"] == mapping[record["raw_output"]]
            assert len(tokens) == 5
        else:
            assert texts == list(string.ascii_uppercase[: len(indices)])
            assert (
                record["question"]
                == READER_QUESTION + f"\nReply with exactly one letter from {', '.join(texts)}."
            )
            assert not record["blank_images"]
        # Independently rebuild representative processed pixels for every native
        # input shape/mode, in addition to checking source pixels for every call.
        roundtrip_key = (
            record["purpose"],
            context.get("mode"),
            len(indices),
            record["blank_images"],
            tuple(i in expected_hidden for i in indices),
        )
        if roundtrip_key not in processed_roundtrips:
            processor.image_processor.size["longest_edge"] = protocol[
                "reader_max_pixels"
                if record["purpose"] == "original_frozen_reader"
                else "selector_max_pixels"
            ]
            images, _ = data.images(alias, indices, record["blank_images"])
            for j, index in enumerate(indices):
                if index in expected_hidden:
                    images[j] = Image.new("RGB", images[j].size)
            content = []
            for j, image in enumerate(images):
                content.extend(
                    [
                        {"type": "text", "text": f"Image {labels[j]}:"},
                        {"type": "image", "image": image},
                    ]
                )
            content.append({"type": "text", "text": record["question"]})
            rebuilt = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            assert rebuilt["input_ids"][0].tolist() == record["input_ids"]
            assert rebuilt["image_grid_thw"].tolist() == record["image_grid_thw"]
            assert (
                hashlib.sha256(
                    rebuilt["pixel_values"].contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                == record["processed_pixels_sha256"]
            )
            processed_roundtrips.add(roundtrip_key)
        checked_inputs += 1

    def trajectory(row, allowed):
        assert row["alias"] in allowed
        q, labels = data.quality(row["alias"])
        observed = initial(row["seed"])
        latest = observed.copy()
        assert row["initial_indices"] == observed
        assert row["initial_wfss"] == float(q[observed].max())
        for turn in row["turns"]:
            record = turn["input"]
            assert record["alias"] == row["alias"]
            assert (
                record["context"]["observed"] == observed and record["context"]["latest"] == latest
            )
            assert record["context"]["mode"] == row["mode"]
            native(record)
            assert turn["center"] == int(record["raw_output"])
            previous = float(q[observed].max())
            latest = inspect(turn["center"], observed)
            assert turn["returned_indices"] == latest
            observed.extend(latest)
            assert (
                abs(turn["private_quality_increment"] - (float(q[observed].max()) - previous))
                < 1e-8
            )
        expected = outcome(q, labels, observed)
        for key in expected:
            assert row[key] == expected[key]
        assert abs(row["reward"] - (row["best_wfss"] - row["initial_wfss"])) < 1e-8

    warm = read_json(run / "warm/report.json")
    assert (
        warm["gate_passed"]
        and warm["frozen_vision_reference_unchanged"]
        and warm["save_reload_identical"]
    )
    warm_records = [read_json(p) for p in sorted((run / "warm/steps").glob("*.json"))]
    assert len(warm_records) == warm["updates"]
    for row in warm_records:
        assert row["alias"] in train and row["split"] == "train"
        native(row["input"])
        selected = row["input"]["image_indices"][
            string.ascii_uppercase.index(row["input"]["raw_output"])
        ]
        q, _ = data.quality(row["alias"])
        assert q[selected] == q.max() and q.max() > 0
    for phase in ("before", "after"):
        controls = read_json(run / "warm" / phase / "recognition_inputs.json")
        for row in controls:
            assert row["alias"] in tune
            native(row["input"])
        report = read_json(run / "warm" / phase / "report.json")
        assert sum(r["correct"] for r in controls) == report["correct"]
    train_episodes = 0
    learning = {}
    before_tune = {}
    for mode in protocol["arms"]:
        path = run / mode / "training"
        report = read_json(path / "report.json")
        assert report["updates"] == protocol["rl_updates"]
        assert report["frozen_vision_reference_unchanged"] and report["save_reload_identical"]
        checkpoint = (
            path / f"step-{protocol['rl_updates']:04d}" / "policy/adapter_model.safetensors"
        )
        if checkpoint.exists():
            assert file_hash(checkpoint) == lock["adapter_hashes"][mode]
        else:
            assert development_only and lock["hashes_verified_on_crc"]
        steps = [read_json(p) for p in sorted((path / "steps").glob("*.json"))]
        assert len(steps) == protocol["rl_updates"]
        learning[mode] = {
            "updates": len(steps),
            "reward_varying_groups": sum(not step["zero_variance"] for step in steps),
            "max_behavior_logprob_drift": max(step["max_logprob_drift"] for step in steps),
            "language_adapter_changed": report["language_adapter_changed"],
            "positive_advantage_second_actions": 0,
            "positive_advantage_second_actions_without_new_quality_gain": 0,
        }
        for step in steps:
            assert step["alias"] in train and step["split"] == "train"
            assert step["max_logprob_drift"] <= protocol["max_behavior_logprob_drift"]
            assert len(step["trajectories"]) == protocol["group_size"]
            rewards = [row["reward"] for row in step["trajectories"]]
            assert abs(step["mean_reward"] - np.mean(rewards)) < 1e-8
            assert step["zero_variance"] == (np.std(rewards) < 1e-8)
            for row in step["trajectories"]:
                assert row["alias"] == step["alias"] and row["mode"] == mode
                trajectory(row, train)
                if row["reward"] > np.mean(rewards) + 1e-7:
                    learning[mode]["positive_advantage_second_actions"] += 1
                    if row["turns"][1]["private_quality_increment"] == 0:
                        learning[mode][
                            "positive_advantage_second_actions_without_new_quality_gain"
                        ] += 1
                train_episodes += 1
        for phase in ("before_tune", "after_tune"):
            records = read_json(run / mode / phase / "acquisition.json")
            assert len(records) == len(tune) * len(protocol["evaluation_starts"])
            for row in records:
                trajectory(row, tune)
            if phase == "before_tune":
                before_tune[mode] = {(r["alias"], r["seed"]): r for r in records}
    visual_changes = history_changes = set_changes = 0
    for key, full in before_tune["full"].items():
        current, geometry = before_tune["current"][key], before_tune["geometry"][key]
        a, b, c = [row["turns"][0]["input"] for row in (full, current, geometry)]
        assert a["input_ids"] == b["input_ids"] == c["input_ids"]
        assert a["processed_pixels_sha256"] == b["processed_pixels_sha256"]
        assert a["processed_pixels_sha256"] != c["processed_pixels_sha256"]
        assert full["turns"][0]["center"] == current["turns"][0]["center"]
        visual_changes += full["turns"][0]["center"] != geometry["turns"][0]["center"]
        history_changes += full["turns"][1]["center"] != current["turns"][1]["center"]
        set_changes += full["indices"] != current["indices"]
    rows, readers = [], []
    if not development_only:
        rows = read_json(run / "evaluation/acquisition.json")
        train_quality = np.stack([data.quality(alias)[0] for alias in sorted(train)])
        prior_sets = {}
        for seed in protocol["evaluation_starts"]:
            selected = initial(seed)
            for _ in range(2):
                selected.extend(inspect(prior_action(train_quality, selected), selected))
            prior_sets[seed] = sorted(selected)
        for row in rows:
            assert row["alias"] in heldout and row["split"] == "evaluation"
            if "turns" in row:
                trajectory(row, heldout)
            else:
                q, labels = data.quality(row["alias"])
                expected = outcome(q, labels, row["indices"])
                assert all(row[k] == v for k, v in expected.items())
                method, seed = row["method"], row["seed"]
                if method == "uniform":
                    indices = uniform()
                elif method == "coverage":
                    indices = coverage(seed)
                elif method == "random":
                    indices = sorted(random.Random(seed).sample(range(840), 12))
                elif method == "random_actions":
                    rng, indices = random.Random(seed), initial(seed)
                    for _ in range(2):
                        indices.extend(inspect(rng.choice(legal_centers(indices)), indices))
                    indices.sort()
                elif method == "geometry_prior":
                    indices = prior_sets[seed]
                else:
                    assert method == "privileged"
                    indices = sorted(sorted(range(840), key=lambda i: (-q[i], i))[:12])
                assert row["indices"] == indices
        acquisition_by_key = {(r["alias"], r["method"], r["seed"]): r for r in rows}
        readers = read_json(run / "reader/reader_results.json")
        for row in readers:
            assert row["alias"] in heldout
            assert (
                row["input"]["image_indices"]
                == acquisition_by_key[(row["alias"], row["method"], row["seed"])]["indices"]
            )
            native(row["input"])
            selected = row["input"]["image_indices"][
                string.ascii_uppercase.index(row["input"]["raw_output"])
            ]
            assert row["selected_frame"] == selected
            assert row["reader_wfss"] == float(data.quality(row["alias"])[0][selected])
        assert (
            read_json(run / "reader/report.json")["parameter_sha256"]
            == "fe27583d6f9c0df41da4559e775f2d09d014f3b267b14902c54ea173a04115d1"
        )
    history_probe = None
    if (run / "history_probe/report.json").exists():
        history_probe = read_json(run / "history_probe/report.json")
        assert history_probe["status"] == "complete" and history_probe["policy_unchanged"]
        assert history_probe["checkpoint_sha256"] == lock["adapter_hashes"]["full"]
        assert not history_probe["heldout_opened"]
        probes = read_json(run / "history_probe/interventions.json")
        final_tune = {
            (r["alias"], r["seed"]): r for r in read_json(run / "full/after_tune/acquisition.json")
        }
        assert {(r["alias"], r["seed"]) for r in probes} == set(final_tune)
        assert len(probes) == len(final_tune) == 16
        for row in probes:
            reference = final_tune[(row["alias"], row["seed"])]
            assert row["alias"] in tune and row["split"] == "tune"
            assert row["fixed_first_action"] == reference["turns"][0]["center"]
            context = reference["turns"][1]["input"]["context"]
            observed, latest = context["observed"], context["latest"]
            older = sorted(set(observed) - set(latest))
            assert row["older_indices"] == older and row["latest_indices"] == latest
            full, hidden = row["full_history"], row["hidden_older_images"]
            for key in ("input_ids", "processed_pixels_sha256", "raw_output", "image_grid_thw"):
                assert full["input"][key] == reference["turns"][1]["input"][key]
            for key in ("input_ids", "image_grid_thw", "image_indices", "legal_choices", "context"):
                assert hidden["input"][key] == full["input"][key]
            q, labels = data.quality(row["alias"])
            for variant, obscured in ((full, []), (hidden, older)):
                native(variant["input"], obscured)
                selected = observed + inspect(int(variant["input"]["raw_output"]), observed)
                assert all(variant[k] == v for k, v in outcome(q, labels, selected).items())
        for variant, field in (
            ("full_history", "full_history_mean_wfss"),
            ("hidden_older_images", "hidden_older_mean_wfss"),
        ):
            assert (
                abs(history_probe[field] - np.mean([r[variant]["best_wfss"] for r in probes]))
                < 1e-8
            )
        changed = sum(
            r["full_history"]["input"]["raw_output"]
            != r["hidden_older_images"]["input"]["raw_output"]
            for r in probes
        )
        assert history_probe["second_action_changed"] == changed
    anchored_probe = None
    if (run / "anchored_probe/report.json").exists():
        anchored_probe = read_json(run / "anchored_probe/report.json")
        assert anchored_probe["status"] == "complete" and anchored_probe["policy_unchanged"]
        assert anchored_probe["optimizer_updates"] == 0 and not anchored_probe["heldout_opened"]
        assert anchored_probe["script_sha256"] == file_hash(
            Path(__file__).with_name("probe_acouslic_anchored_actions.py")
        )
        records = read_json(run / "anchored_probe/acquisition.json")
        expected_keys = {
            (a, m, s) for a in tune for m in protocol["arms"] for s in protocol["evaluation_starts"]
        }
        assert len(records) == 48
        assert {(r["alias"], r["mode"], r["seed"]) for r in records} == expected_keys
        for row in records:
            assert row["split"] == "tune"
            q, labels = data.quality(row["alias"])
            observed = initial(row["seed"])
            latest = observed.copy()
            assert row["initial_wfss"] == float(q[observed].max())
            for turn in row["turns"]:
                record = turn["input"]
                assert record["alias"] == row["alias"] and record["context"]["mode"] == row["mode"]
                assert (
                    record["context"]["observed"] == observed
                    and record["context"]["latest"] == latest
                )
                native(record)
                assert turn["center"] == record["action_center"]
                old = float(q[observed].max())
                latest = inspect(turn["center"], observed)
                assert latest == turn["returned_indices"]
                observed.extend(latest)
                assert (
                    abs(turn["private_quality_increment"] - (float(q[observed].max()) - old)) < 1e-8
                )
            assert all(row[k] == v for k, v in outcome(q, labels, observed).items())
        engineering = read_json(run / "anchored_probe/engineering_checks.json")
        assert {r["mode"] for r in engineering} == set(protocol["arms"])
        for check in engineering:
            assert check["input"]["alias"] in train
            assert check["drift"] <= protocol["max_behavior_logprob_drift"]
            assert np.isfinite(check["gradient_norm"]) and check["gradient_norm"] > 0
            native(check["input"])
        for mode in protocol["arms"]:
            selected = [r for r in records if r["mode"] == mode]
            assert (
                abs(
                    anchored_probe["methods"][mode]["mean_acquisition_wfss"]
                    - np.mean([r["best_wfss"] for r in selected])
                )
                < 1e-8
            )
            assert (
                abs(
                    anchored_probe["methods"][mode]["mean_inspection_gain"]
                    - np.mean([r["best_wfss"] - r["initial_wfss"] for r in selected])
                )
                < 1e-8
            )
    result = {
        "status": "passed",
        "scope": "development_only" if development_only else "full_pilot",
        "heldout_opened": not development_only,
        "native_inputs": checked_inputs,
        "unique_source_frames": len(cache),
        "representative_native_processor_roundtrips": len(processed_roundtrips),
        "supervised_updates": len(warm_records),
        "rl_episodes": train_episodes,
        "training_signal": learning,
        "pre_rl_tuning_interventions": {
            "subject_start_pairs": len(before_tune["full"]),
            "blanking_images_changes_first_action": visual_changes,
            "removing_older_images_changes_second_action": history_changes,
            "history_changes_final_acquired_set": set_changes,
            "interpretation": "Input sensitivity is not evidence of improved acquisition",
        },
        "tuning_older_image_probe": history_probe,
        "tuning_anchored_action_probe": anchored_probe,
        "evaluation_rows": len(rows),
        "reader_rows": len(readers),
        "split_boundaries": "32 train / 8 tune / 20 held-out subjects; screened subjects train only",
        "checks": [
            "source and presented RGB hashes",
            "native prompt tokens and image grids",
            "representative processed-pixel hashes across each input shape and mode",
            "exact legal decoding support",
            "causal visual-history snapshots",
            "all action and reward replay",
            "12 unique frames",
            "checkpoint lock",
            "train-only updates",
        ],
        "breastvid_test_opened": False,
    }
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development-only", action="store_true")
    args = p.parse_args()
    audit(args.data, args.run, args.output, args.development_only)
