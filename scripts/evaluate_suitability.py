#!/usr/bin/env python3
"""Frozen-reader headroom and train-only history values. No optimizer or RL."""

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sonomed_vlm.active_reader.cli import audited_cases, data_root
from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.io import code_revision, file_hash, read_json, versions, write_json
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory
from sonomed_vlm.active_reader.navigation.reader import FrozenReader
from sonomed_vlm.active_reader.suitability import (
    AuditReader,
    evaluate_development,
    evaluate_history,
)
from sonomed_vlm.active_reader.training.grpo import parameter_digest


def main(args):
    config = load_config(args.config)
    config.reward.relevance_weight = 0
    config.environment.initial_images = 4
    config.environment.max_unique_frames = 12
    inventory_cases, audit = audited_cases(config)
    counts = dict(Counter(c.split for c in inventory_cases))
    cases = [c for c in inventory_cases if c.split in ("train", "dev")]
    provenance = {
        "protocol": "suitability-v1",
        "config": config.model_dump(mode="json"),
        "code_revision": code_revision(),
        "runner_sha256": file_hash(Path(__file__)),
        "manifest_sha256": file_hash(config.data.manifest),
        "splits_sha256": file_hash(config.data.splits),
        "data_audit_sha256": file_hash(config.data.audit),
        "manifest_fingerprint": audit["manifest_fingerprint"],
        "model": inventory(config.model),
        "versions": versions(),
        "inventory": counts,
        "phase": args.phase,
        "history_cases_per_role": args.history_cases,
        "training": False,
        "test_opened": False,
    }
    meta = args.output / "provenance.json"
    if meta.exists() and read_json(meta) != provenance:
        raise ValueError("Audit provenance changed; use a fresh output directory")
    write_json(meta, provenance)
    torch.set_float32_matmul_precision("highest")
    reader = FrozenReader(QwenPolicy(config.model))
    before = parameter_digest(reader.policy.model, lambda n, p: True)
    reader.policy.processor.save_pretrained(args.output / "processor")
    scorer = AuditReader(reader, data_root(config), args.output)
    if args.phase in ("development", "all"):
        evaluate_development(cases, scorer, args.output)
    if args.phase in ("history", "all"):
        evaluate_history(cases, scorer, config, data_root(config), args.output, args.history_cases)
    reader.assert_frozen()
    after = parameter_digest(reader.policy.model, lambda n, p: True)
    assert before == after
    write_json(
        args.output / "report.json",
        {
            "status": "complete",
            "frozen_reader_unchanged": True,
            "reader_parameter_sha256": after,
            "test_opened": False,
        },
    )
    print("Suitability evaluation complete; no training performed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/navigation_crc.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("development", "history", "all"), default="all")
    p.add_argument("--history-cases", type=int, default=32)
    main(p.parse_args())
