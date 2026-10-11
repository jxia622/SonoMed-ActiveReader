#!/usr/bin/env python3
"""Run an independent, matched training arm after a completed development selection gate.

Writes only to a separate worker directory. The campaign can consume its completed
training directory via a symlink created before that arm starts in the main process.
"""

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sonomed_vlm.active_reader.cli import audited_cases, data_root
from sonomed_vlm.active_reader.config import Config
from sonomed_vlm.active_reader.io import code_revision, file_hash, read_json, write_json
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory
from sonomed_vlm.active_reader.navigation.reader import FrozenReader
from sonomed_vlm.active_reader.navigation.training import train_navigation


def run(campaign, arm, output):
    provenance = read_json(campaign / "campaign.json")
    gate = read_json(campaign / "selection_gate.json")
    if (
        not gate["reader_scores_vary_by_selection"]
        or not (campaign / "before/report.json").is_file()
    ):
        raise ValueError("A completed development selection comparison must precede RL")
    if provenance["code_revision"] != code_revision():
        raise ValueError("Worker code differs from the baseline campaign")
    config = Config.model_validate(provenance["config"])
    torch.set_float32_matmul_precision(provenance["float32_matmul_precision"])
    selector_config = config.model.model_copy(update={"dtype": config.navigation.selector_dtype})
    if inventory(config.model) != provenance["reader"]:
        raise ValueError("Frozen-reader assets or runtime differ from the campaign")
    if inventory(selector_config) != provenance["selector_initial"]:
        raise ValueError("Selector assets or runtime differ from the campaign")
    cases, audit = audited_cases(config)
    if (
        audit["manifest_fingerprint"] != provenance["data"]
        or file_hash(config.data.splits) != provenance["splits"]
    ):
        raise ValueError("Worker data differs from the campaign")
    history = "current_frame" if arm == "current_frame" else "full"
    config.reward.relevance_weight = 0.1 if arm == "full_box_auxiliary" else 0.0
    config.training.steps = provenance["config"]["training"]["steps"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "worker.json",
        {"campaign": str(campaign), "arm": arm, "slurm_job": os.environ.get("SLURM_JOB_ID")},
    )
    print(f"Matched worker {arm}: loading immutable reader and fresh selector", flush=True)
    reader = FrozenReader(QwenPolicy(config.model))
    selector = QwenPolicy(selector_config, for_training=True)
    train_navigation(
        selector,
        reader,
        [c for c in cases if c.split == "train"],
        data_root(config),
        config,
        output / "training",
        provenance,
        history,
    )
    print(f"Matched worker {arm} complete: {output / 'training'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--arm", choices=["current_frame", "full_box_auxiliary"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign, args.arm, args.output)
