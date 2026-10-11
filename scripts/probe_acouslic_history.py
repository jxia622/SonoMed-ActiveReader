#!/usr/bin/env python3
"""Tuning-only older-image intervention at the fixed final full-history checkpoint.

This mechanism diagnostic preserves the first action, all prompt tokens, image
slots, coordinates, budget, and latest three images. Only older visual pixels
are replaced with black. It cannot select a checkpoint or establish a held-out
performance claim.
"""

import argparse
import hashlib
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image

from sonomed_vlm.active_reader.acouslic.data import Cases
from sonomed_vlm.active_reader.acouslic.environment import inspect, outcome
from sonomed_vlm.active_reader.acouslic.policy import Choices
from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.io import code_revision, file_hash, read_json, write_json
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy
from sonomed_vlm.active_reader.training.grpo import parameter_digest


class HideOlderImages:
    def __init__(self, data, hidden):
        self.data, self.hidden = data, set(hidden)

    def images(self, alias, indices, blank=False):
        assert not blank
        images, hashes = self.data.images(alias, indices)
        for j, index in enumerate(indices):
            if index in self.hidden:
                images[j] = Image.new("RGB", images[j].size)
                hashes[j]["presented_rgb_sha256"] = hashlib.sha256(images[j].tobytes()).hexdigest()
        return images, hashes


def probe(args):
    campaign = read_json(args.run / "campaign.json")
    protocol = campaign["protocol"]
    assert campaign["code_revision"] == code_revision()
    assert campaign["data_partition_sha256"] == file_hash(args.data / "partition.json")
    training = read_json(args.run / "full/training/report.json")
    assert training["updates"] == protocol["rl_updates"]
    checkpoint = args.run / "full/training" / f"step-{protocol['rl_updates']:04d}"
    data = Cases(args.data)  # The process never admits held-out subjects.
    tune = {c["alias"] for c in data.subset("tune")}
    originals = read_json(args.run / "full/after_tune/acquisition.json")
    assert len(originals) == 16 and {r["alias"] for r in originals} == tune
    assert all(r["split"] == "tune" for r in originals)
    assert not args.output.exists(), "Preserve an existing diagnostic; do not overwrite"
    config = load_config(args.config).model.model_copy(
        update={"dtype": "float32", "image_max_pixels": protocol["selector_max_pixels"]}
    )
    assert (
        file_hash(Path(config.adapter) / "adapter_model.safetensors")
        == protocol["original_adapter_sha256"]
    )
    torch.set_float32_matmul_precision("highest")
    policy = QwenPolicy(config)
    policy.load_policy(checkpoint)
    before = parameter_digest(policy.model, lambda name, p: ".policy." in name)
    rows = []
    blank_hash = hashlib.sha256(Image.new("RGB", (744, 562)).tobytes()).hexdigest()
    for row in originals:
        reference = row["turns"][1]["input"]
        observed = reference["context"]["observed"]
        latest = reference["context"]["latest"]
        older = sorted(set(observed) - set(latest))
        assert len(observed) == 9 and len(latest) == 3 and len(older) == 6
        real = Choices(policy, data).inspect(row["alias"], observed, latest, "full", 31)
        assert real.text == reference["raw_output"], "Native final-checkpoint replay differs"
        for key in ("input_ids", "processed_pixels_sha256", "image_grid_thw", "legal_choices"):
            assert real.record[key] == reference[key]
        hidden = Choices(policy, HideOlderImages(data, older)).inspect(
            row["alias"], observed, latest, "full", 31
        )
        for key in ("input_ids", "image_grid_thw", "legal_choices", "image_indices", "question"):
            assert hidden.record[key] == real.record[key]
        for a, b in zip(real.record["clean_images"], hidden.record["clean_images"], strict=True):
            assert a["index"] == b["index"] and a["source_rgb_sha256"] == b["source_rgb_sha256"]
            assert b["presented_rgb_sha256"] == (
                blank_hash if b["index"] in older else a["presented_rgb_sha256"]
            )
        hidden.record["hidden_older_indices"] = older
        quality, labels = data.quality(row["alias"])
        full_outcome = outcome(quality, labels, observed + inspect(int(real.text), observed))
        hidden_outcome = outcome(quality, labels, observed + inspect(int(hidden.text), observed))
        assert full_outcome["indices"] == row["indices"]
        assert full_outcome["best_wfss"] == row["best_wfss"]
        rows.append(
            {
                "alias": row["alias"],
                "split": "tune",
                "seed": row["seed"],
                "fixed_first_action": row["turns"][0]["center"],
                "older_indices": older,
                "latest_indices": latest,
                "full_history": {"input": real.record, **full_outcome},
                "hidden_older_images": {"input": hidden.record, **hidden_outcome},
            }
        )
        write_json(args.output / "interventions.json", rows)
    assert before == parameter_digest(policy.model, lambda name, p: ".policy." in name)
    differences = np.array(
        [
            np.mean(
                [
                    r["full_history"]["best_wfss"] - r["hidden_older_images"]["best_wfss"]
                    for r in rows
                    if r["alias"] == alias
                ]
            )
            for alias in sorted(tune)
        ]
    )
    sample = np.random.default_rng(20261011).integers(0, 8, size=(10000, 8))
    report = {
        "status": "complete",
        "scope": "Exploratory tuning-only mechanism diagnostic, not a held-out claim",
        "subjects": 8,
        "subject_start_pairs": len(rows),
        "checkpoint_sha256": file_hash(checkpoint / "policy/adapter_model.safetensors"),
        "script_sha256": file_hash(Path(__file__)),
        "data_partition_sha256": campaign["data_partition_sha256"],
        "full_history_mean_wfss": float(np.mean([r["full_history"]["best_wfss"] for r in rows])),
        "hidden_older_mean_wfss": float(
            np.mean([r["hidden_older_images"]["best_wfss"] for r in rows])
        ),
        "paired_subject_mean_difference": float(differences.mean()),
        "paired_subject_percentile_95": np.quantile(
            differences[sample].mean(1), [0.025, 0.975]
        ).tolist(),
        "second_action_changed": sum(
            r["full_history"]["input"]["raw_output"]
            != r["hidden_older_images"]["input"]["raw_output"]
            for r in rows
        ),
        "native_full_history_replay_identical": True,
        "prompt_tokens_image_slots_latest_images_and_first_action_fixed": True,
        "policy_unchanged": True,
        "heldout_opened": False,
        "breastvid_test_opened": False,
    }
    write_json(args.output / "report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/navigation_crc.yaml"))
    probe(parser.parse_args())
