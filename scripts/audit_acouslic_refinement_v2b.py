#!/usr/bin/env python3
"""Independently replay the development refinement campaign without opening held-out data."""

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
from sonomed_vlm.active_reader.acouslic.environment import initial, inspect, outcome
from sonomed_vlm.active_reader.io import digest, file_hash, read_json, write_json


def audit(args):
    campaign = read_json(args.run / "campaign.json")
    protocol = campaign["protocol"]
    report = read_json(args.run / "report.json")
    assert report["status"] == "complete" and not report["heldout_opened"]
    assert report["protocol_sha256"] == campaign["protocol_sha256"] == digest(protocol)
    assert file_hash(args.data / "partition.json") == protocol["data_partition_sha256"]
    assert all(file_hash(Path(p)) == expected for p, expected in protocol["source_sha256"].items())
    assert report["frozen_vision_reference_unchanged"] and report["save_reload_identical"]
    assert report["language_adapter_changed"]
    assert (
        file_hash(Path(report["checkpoint"]) / "policy/adapter_model.safetensors")
        == report["checkpoint_sha256"]
    )
    data = Cases(args.data)
    train = data.subset("train")
    train_aliases = {r["alias"] for r in train}
    tune_aliases = {r["alias"] for r in data.subset("tune")}
    tokenizer = Tokenizer.from_file(str(args.parent / "warm/checkpoint/processor/tokenizer.json"))
    processor = AutoProcessor.from_pretrained(
        args.parent / "warm/checkpoint/processor", local_files_only=True
    )
    processor.image_processor.size["longest_edge"] = protocol["selector_max_pixels"]
    blank_hash = hashlib.sha256(Image.new("RGB", (744, 562)).tobytes()).hexdigest()
    native_count, pixel_roundtrips, source_cache = 0, set(), {}
    mode = campaign["mode"]

    def native(record, alias, observed, latest):
        nonlocal native_count
        context = record["context"]
        assert record["alias"] == alias and context["mode"] == mode
        assert context["observed"] == observed and context["latest"] == latest
        assert context["budget_remaining"] == 12 - len(observed)
        indices = latest if mode == "current" else observed
        assert record["image_indices"] == indices
        assert record["blank_images"] == (mode == "geometry")
        labels = [string.ascii_uppercase[observed.index(i)] for i in indices]
        assert record["displayed_image_labels"] == labels
        mapping = {}
        for j, anchor in enumerate(observed):
            for center in range(anchor // 140 * 140, (anchor // 140 + 1) * 140):
                if center not in observed:
                    mapping[f"{string.ascii_uppercase[j]}{center - anchor:+04d}"] = center
        assert record["legal_choices"] == list(mapping) and record["action_mapping"] == mapping
        assert set(mapping.values()) == set(range(840)) - set(observed)
        assert mapping[record["raw_output"]] == record["action_center"]
        text = "<|im_start|>user\n"
        for label, index, frame, grid in zip(
            labels, indices, record["clean_images"], record["image_grid_thw"], strict=True
        ):
            key = (alias, index)
            if key not in source_cache:
                path = args.data / "cases" / alias / "frames" / f"{index:04d}.png"
                assert file_hash(path) == data.admission(alias)["frames"][str(index)]
                with Image.open(path) as image:
                    assert image.size == (744, 562) and not image.info
                    source_cache[key] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            assert frame["index"] == index and frame["source_rgb_sha256"] == source_cache[key]
            assert frame["presented_rgb_sha256"] == (
                blank_hash if mode == "geometry" else source_cache[key]
            )
            text += (
                f"Image {label}:<|vision_start|>"
                + "<|image_pad|>" * (int(np.prod(grid)) // 4)
                + "<|vision_end|>"
            )
        text += record["question"] + "<|im_end|>\n<|im_start|>assistant\n"
        assert tokenizer.encode(text, add_special_tokens=False).ids == record["input_ids"]
        sequences = [tokenizer.encode(t, add_special_tokens=False).ids for t in mapping]
        tokens = record["generated_token_ids"]
        assert (
            len(tokens) == 5
            and tokens == tokenizer.encode(record["raw_output"], add_special_tokens=False).ids
        )
        assert len(record["token_constraints"]) == len(record["behavior_log_probs"]) == 5
        for j, allowed in enumerate(record["token_constraints"]):
            expected = sorted({s[j] for s in sequences if s[:j] == tokens[:j]})
            assert allowed == expected and tokens[j] in allowed
        assert (
            np.isfinite(record["behavior_log_probs"]).all() and record["deterministic_termination"]
        )
        shape = (mode, len(observed), tuple(labels))
        if shape not in pixel_roundtrips:
            images, _ = data.images(alias, indices, blank=mode == "geometry")
            content = []
            for label, image in zip(labels, images, strict=True):
                content.extend(
                    [{"type": "text", "text": f"Image {label}:"}, {"type": "image", "image": image}]
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
            pixel_roundtrips.add(shape)
        native_count += 1
        return mapping[record["raw_output"]]

    def private_values(alias):
        q, labels = data.quality(alias)
        nearby = np.zeros(840)
        for target in np.flatnonzero(q > 0):
            indices = np.arange(target // 140 * 140, (target // 140 + 1) * 140)
            nearby[indices] = np.maximum(
                nearby[indices],
                q[target]
                * np.exp(-0.5 * ((indices - target) / protocol["proximity_sigma_frames"]) ** 2),
            )
        return q, labels, nearby

    def standardize(values):
        array = np.asarray(values, dtype=np.float32)
        return (array - array.mean()) / (array.std() + 1e-8)

    steps = sorted((args.run / "training/steps").glob("*.json"))
    assert (
        len(steps)
        == report["updates"]
        == (2 if campaign["stage"] == "smoke" else protocol["updates"])
    )
    first_signal, second_signal = 0, 0
    for position, path in enumerate(steps):
        row = read_json(path)
        order = list(train)
        random.Random(protocol["training_seed"] + position // 32).shuffle(order)
        assert row["alias"] == order[position % 32]["alias"] and row["alias"] in train_aliases
        assert row["split"] == "train" and row["mode"] == mode and row["step"] == position + 1
        assert row["start_seed"] == protocol["training_seed"] + position * 17
        assert row["max_logprob_drift"] <= protocol["max_behavior_logprob_drift"]
        assert len(row["branches"]) == 4
        q, labels, nearby = private_values(row["alias"])
        observed = initial(row["start_seed"])
        assert row["initial_indices"] == observed and row["initial_wfss"] == float(
            q[observed].max()
        )
        assert row["initial_proximity"] == float(nearby[observed].max())
        assert row["auxiliary_initial_potential"] == 0
        branch_values, exact_gains = [], []
        for j, branch in enumerate(row["branches"]):
            center = native(branch["first_input"], row["alias"], observed, observed)
            assert center == branch["first_center"]
            latest = inspect(center, observed)
            assert branch["returned_indices"] == latest
            middle = observed + latest
            first_q = float(q[middle].max() - q[observed].max())
            first_near = float(nearby[middle[6:]].max())
            assert abs(branch["exact_first_gain"] - first_q) < 1e-8
            assert abs(branch["auxiliary_first_gain"] - first_near) < 1e-8
            first_reward = first_q + protocol["proximity_weight"] * first_near
            assert abs(branch["first_reward"] - first_reward) < 1e-8
            assert len(branch["continuations"]) == 2
            terminals, increments = [], []
            for second in branch["continuations"]:
                center = native(second["input"], row["alias"], middle, latest)
                assert center == second["center"]
                returned = inspect(center, middle)
                assert second["returned_indices"] == returned
                selected = middle + returned
                assert all(second[k] == v for k, v in outcome(q, labels, selected).items())
                gain = float(q[selected].max() - q[middle].max())
                near_gain = float(nearby[selected[6:]].max() - nearby[middle[6:]].max())
                reward = gain + protocol["proximity_weight"] * near_gain
                assert abs(second["exact_second_gain"] - gain) < 1e-8
                assert abs(second["auxiliary_second_gain"] - near_gain) < 1e-8
                assert abs(second["second_reward"] - reward) < 1e-8
                assert abs(second["terminal_reward"] - (first_reward + reward)) < 1e-8
                terminals.append(first_reward + reward)
                increments.append(reward)
                exact_gains.append(float(q[selected].max() - q[observed].max()))
            assert np.allclose(row["second_advantages"][j], standardize(increments), atol=1e-5)
            assert row["second_zero_variance"][j] == (
                np.std(np.array(increments, dtype=np.float32)) < 1e-8
            )
            second_signal += not row["second_zero_variance"][j]
            branch_values.append(np.mean(terminals))
        assert np.allclose(row["first_advantages"], standardize(branch_values), atol=1e-5)
        assert row["first_zero_variance"] == (
            np.std(np.array(branch_values, dtype=np.float32)) < 1e-8
        )
        first_signal += not row["first_zero_variance"]
        assert abs(row["mean_exact_gain"] - np.mean(exact_gains)) < 1e-8
        assert abs(row["mean_training_reward"] - np.mean(branch_values)) < 1e-8
    assert report["first_reward_varying_groups"] == first_signal
    assert report["second_reward_varying_groups"] == second_signal
    for phase in ("before_tune", "after_tune"):
        if campaign["stage"] == "smoke":
            assert not (args.run / phase).exists()
            continue
        rows = read_json(args.run / phase / "acquisition.json")
        assert len(rows) == 16 and {(r["alias"], r["seed"]) for r in rows} == {
            (a, s) for a in tune_aliases for s in protocol["evaluation_starts"]
        }
        for row in rows:
            assert row["split"] == "tune" and row["mode"] == mode
            q, labels = data.quality(row["alias"])
            observed, latest = initial(row["seed"]), initial(row["seed"])
            assert row["initial_wfss"] == float(q[observed].max())
            for turn in row["turns"]:
                center = native(turn["input"], row["alias"], observed, latest)
                assert center == turn["center"]
                latest = inspect(center, observed)
                assert latest == turn["returned_indices"]
                old = float(q[observed].max())
                observed.extend(latest)
                assert (
                    abs(turn["private_quality_increment"] - (float(q[observed].max()) - old)) < 1e-8
                )
            assert all(row[k] == v for k, v in outcome(q, labels, observed).items())
        assert abs(report[f"{phase}_wfss"] - np.mean([r["best_wfss"] for r in rows])) < 1e-8
    result = {
        "status": "passed",
        "native_inputs": native_count,
        "source_frames": len(source_cache),
        "processed_roundtrips": len(pixel_roundtrips),
        "updates": len(steps),
        "first_groups_with_signal": first_signal,
        "second_groups_with_signal": second_signal,
        "conditional_credit_replayed": True,
        "source_and_native_inputs_replayed": True,
        "heldout_opened": False,
        "breastvid_test_opened": False,
    }
    write_json(args.run / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    audit(p.parse_args())
