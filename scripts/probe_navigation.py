#!/usr/bin/env python3
"""Corrected dev controls, train-only action values, then a small preference proof."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sonomed_vlm.active_reader.cli import audited_cases, data_root
from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.io import (
    code_revision,
    digest,
    file_hash,
    read_json,
    versions,
    write_json,
)
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory
from sonomed_vlm.active_reader.navigation.evaluation import run_evaluation
from sonomed_vlm.active_reader.navigation.preference import (
    learning_comparison,
    score_states,
    train_preferences,
)
from sonomed_vlm.active_reader.navigation.probe import collect, partition_cases, summarize_signal
from sonomed_vlm.active_reader.navigation.prompts import READER_HASH, SELECTOR_HASH
from sonomed_vlm.active_reader.navigation.reader import FrozenReader
from sonomed_vlm.active_reader.navigation.training import reset_selector
from sonomed_vlm.active_reader.training.grpo import parameter_digest


def run(args):
    config = load_config(args.config)
    config.reward.relevance_weight = 0
    config.environment.initial_images = 4
    config.environment.max_unique_frames = 12
    config.evaluation.budgets = [8, 12]
    config.evaluation.seeds = [7, 19, 43]
    cases, audit = audited_cases(config)
    train = [c for c in cases if c.split == "train"]
    partition = partition_cases(train, args.cases_per_subset, config.seed)
    torch.set_float32_matmul_precision("highest")
    selector_config = config.model.model_copy(update={"dtype": "float32"})
    provenance = {
        "protocol": "counterfactual-evidence-v1",
        "runner_sha256": file_hash(Path(__file__)),
        "code_revision": code_revision(),
        "config": config.model_dump(mode="json"),
        "reader": inventory(config.model),
        "selector": inventory(selector_config),
        "versions": versions(),
        "reader_prompt": READER_HASH,
        "selector_prompt": SELECTOR_HASH,
        "data": audit["manifest_fingerprint"],
        "splits": file_hash(config.data.splits),
        "cases_per_subset": args.cases_per_subset,
        "menu_size": args.menu_size,
        "steps_per_arm": args.steps,
        "learning_rate": args.learning_rate,
        "minimum_preference_gap": 0.1,
        "probe_start_seed": 7,
        "partition": {role: [c.video_id for c in subset] for role, subset in partition.items()},
        "baselines": not args.skip_baselines,
        "test_policy": "sealed; only train and dev permitted",
    }
    out = args.output
    meta = out / "provenance.json"
    if meta.exists() and read_json(meta) != provenance:
        raise ValueError("Probe provenance differs; use a new output directory")
    write_json(meta, provenance)
    reader = FrozenReader(QwenPolicy(config.model))
    reader_hash = parameter_digest(reader.policy.model, lambda n, p: True)
    if not args.skip_baselines:
        print("Corrected full-development fixed-reader controls at three matched seeds", flush=True)
        run_evaluation(
            None,
            reader,
            [c for c in cases if c.split == "dev"],
            data_root(config),
            config,
            out / "corrected_baselines",
            provenance,
            methods=["uniform", "random", "coverage", "scripted_coverage"],
        )
    rows = collect(
        reader, partition, data_root(config), config, out / "probe", menu_size=args.menu_size
    )
    signal = summarize_signal(rows)
    write_json(out / "selection_signal.json", signal)
    if not signal["fit_gate_passed"]:
        write_json(
            out / "report.json",
            {"status": "insufficient_fit_signal", "signal": signal, "test_opened": False},
        )
        print(
            "Insufficient fit-only action-value signal; no selector training launched", flush=True
        )
        return
    print("Fit-only signal gate passed; loading FP32 selector", flush=True)
    selector = QwenPolicy(selector_config, for_training=True)
    by_id = {c.video_id: c for c in train}
    results = {}
    for mode in ("full", "current_frame"):
        reset_selector(selector)
        before = score_states(
            selector, rows, by_id, data_root(config), config, out / mode / "before", mode
        )
        training = train_preferences(
            selector,
            rows,
            before,
            by_id,
            data_root(config),
            config,
            out / mode / "training",
            digest(provenance),
            mode,
            args.steps,
            args.learning_rate,
        )
        after = score_states(
            selector, rows, by_id, data_root(config), config, out / mode / "after", mode
        )
        learning = learning_comparison(before, after)
        write_json(out / mode / "learning.json", learning)
        results[mode] = {"training": training, "learning": learning}
        if mode == "full":
            # Same trained weights, unchanged clean branch values, fixed menus/counts.
            for ablation in ("current_frame", "shuffled_history", "blank_history", "no_visual"):
                score_states(
                    selector,
                    rows,
                    by_id,
                    data_root(config),
                    config,
                    out / mode / "ablations" / ablation,
                    ablation,
                )
    if parameter_digest(reader.policy.model, lambda n, p: True) != reader_hash:
        raise RuntimeError("Frozen reader weights changed")
    write_json(
        out / "report.json",
        {
            "status": "complete",
            "signal": signal,
            "arms": results,
            "frozen_reader_unchanged": True,
            "test_opened": False,
            "scope": "Small fixed-state train-only preference proof; no automatic large RL or test run",
            "experiment_id": digest(provenance),
        },
    )
    print("Evidence probe and both preference arms complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/navigation_crc.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases-per-subset", type=int, default=8)
    p.add_argument("--menu-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--skip-baselines", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.menu_size < 2 or args.learning_rate <= 0:
        p.error("Steps and learning rate must be positive; menu size must be at least two")
    run(args)
