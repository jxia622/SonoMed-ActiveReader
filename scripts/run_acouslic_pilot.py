#!/usr/bin/env python3
"""Run isolated stages of the predeclared ACOUSLIC pilot on CRC."""

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sonomed_vlm.active_reader.acouslic.data import Cases
from sonomed_vlm.active_reader.acouslic.environment import initial
from sonomed_vlm.active_reader.acouslic.evaluation import (
    baselines,
    frozen_reader,
    policy_evaluation,
)
from sonomed_vlm.active_reader.acouslic.policy import Choices
from sonomed_vlm.active_reader.acouslic.training import train_rl, warm_start
from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.io import code_revision, digest, file_hash, read_json, write_json
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory


def run(args):
    protocol = read_json(args.protocol)
    config = load_config(args.config)
    assert protocol["protocol"] == "acouslic-acquisition-pilot-v1"
    assert (
        file_hash(Path(config.model.adapter) / "adapter_model.safetensors")
        == protocol["original_adapter_sha256"]
    )
    assert (
        read_json(args.data / "partition.json")["partition_fingerprint"]
        == protocol["partition_fingerprint"]
    )
    assert args.mode in {"full", "current", "geometry"}
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": protocol,
        "protocol_sha256": file_hash(args.protocol),
        "code_revision": code_revision(),
        "runner_sha256": file_hash(Path(__file__)),
        "data_partition_sha256": file_hash(args.data / "partition.json"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "breastvid_test_opened": False,
    }
    # Every stage verifies that the same scientific protocol and data partition are used.
    if (args.output / "campaign.json").exists():
        original = read_json(args.output / "campaign.json")
        for key in ("protocol_sha256", "code_revision", "runner_sha256", "data_partition_sha256"):
            assert original[key] == common[key], f"Campaign {key} differs; use a fresh campaign"
    else:
        assert args.stage == "smoke", "Begin with a fresh engineering smoke stage"
        write_json(args.output / "campaign.json", common)
    if args.stage in {"evaluate", "reader"}:
        assert read_json(args.output / "warm/report.json")["gate_passed"]
        for mode in ("full", "current", "geometry"):
            assert (
                read_json(args.output / mode / "training/report.json")["updates"]
                == protocol["rl_updates"]
            )
        lock = {
            "protocol_sha256": common["protocol_sha256"],
            "data_partition_sha256": common["data_partition_sha256"],
            "checkpoint_rule": "fixed final update, no held-out checkpoint selection",
            "adapter_hashes": {
                mode: file_hash(
                    args.output
                    / mode
                    / "training"
                    / f"step-{protocol['rl_updates']:04d}"
                    / "policy/adapter_model.safetensors"
                )
                for mode in ("full", "current", "geometry")
            },
        }
        path = args.output / "evaluation_lock.json"
        if path.exists():
            assert read_json(path) == lock
        else:
            assert args.stage == "evaluate"
            write_json(path, lock)
        data = Cases(args.data, allow_evaluation=True)
    else:
        data = Cases(args.data)
    if args.stage == "reader":
        policy = QwenPolicy(config.model)
        rows = read_json(args.output / "evaluation/acquisition.json")
        frozen_reader(policy, data, rows, args.output / "reader")
        return
    selector_config = config.model.model_copy(
        update={"dtype": "float32", "image_max_pixels": protocol["selector_max_pixels"]}
    )
    policy = QwenPolicy(selector_config, for_training=args.stage in {"smoke", "warm", "arm"})
    if args.stage == "smoke":
        out = args.output / "smoke"
        assert not (out / "report.json").exists()
        write_json(out / "inventory.json", inventory(selector_config))
        choices = Choices(policy, data)
        case = data.subset("train")[0]
        observed = initial(7)
        generation = choices.inspect(
            case["alias"], observed, observed, "full", 1701, sample=True, retain=True
        )
        old = generation.behavior_log_probs
        current = policy.log_probs(generation)
        drift = float((current.detach().cpu() - old).abs().max())
        assert drift <= protocol["max_behavior_logprob_drift"], drift
        (-current.sum()).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.model.parameters() if p.requires_grad], 1
        )
        assert torch.isfinite(norm) and norm > 0
        write_json(out / "input.json", generation.record)
        write_json(
            out / "report.json",
            {
                "passed": True,
                "drift": drift,
                "gradient_norm": float(norm),
                "optimizer_updates": 0,
                "heldout_opened": False,
            },
        )
        print(f"Smoke passed: drift={drift:.7f}, gradient_norm={float(norm):.5f}", flush=True)
    elif args.stage == "warm":
        assert read_json(args.output / "smoke/report.json")["passed"]
        baselines(
            data,
            data.subset("train") + data.subset("tune"),
            protocol,
            args.output / "development_baselines",
        )
        warm_start(policy, data, protocol, args.output / "warm")
    elif args.stage == "arm":
        warm = read_json(args.output / "warm/report.json")
        assert warm["gate_passed"], "Recognition feasibility gate failed; do not launch RL"
        policy.load_policy(args.output / "warm/checkpoint")
        policy_evaluation(
            policy,
            data,
            data.subset("tune"),
            protocol,
            args.output / args.mode / "before_tune",
            args.mode,
            "pre_rl_" + args.mode,
        )
        train_rl(
            policy,
            data,
            protocol,
            args.output / args.mode / "training",
            args.output / "warm/checkpoint",
            args.mode,
        )
        policy_evaluation(
            policy,
            data,
            data.subset("tune"),
            protocol,
            args.output / args.mode / "after_tune",
            args.mode,
            "rl_" + args.mode,
        )
    elif args.stage == "evaluate":
        out = args.output / "evaluation"
        assert not out.exists(), "Held-out evaluation is a single frozen campaign pass"
        cases = data.subset("evaluation")
        assert len(cases) == 20 and not any(c["screening_exposed"] for c in cases)
        rows = baselines(data, cases, protocol, out / "baselines")
        rows += policy_evaluation(
            policy, data, cases, protocol, out / "original", "full", "original_prompted"
        )
        policy.load_policy(args.output / "warm/checkpoint")
        for mode in ("full", "current", "geometry"):
            rows += policy_evaluation(
                policy, data, cases, protocol, out / f"pre_rl_{mode}", mode, "pre_rl_" + mode
            )
        for mode in ("full", "current", "geometry"):
            policy.load_policy(
                args.output / mode / "training" / f"step-{protocol['rl_updates']:04d}"
            )
            rows += policy_evaluation(
                policy, data, cases, protocol, out / f"rl_{mode}", mode, "rl_" + mode
            )
        write_json(out / "acquisition.json", rows)
        write_json(
            out / "report.json",
            {
                "status": "complete",
                "subjects": 20,
                "protocol_fingerprint": digest(protocol),
                "rows": len(rows),
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["smoke", "warm", "arm", "evaluate", "reader"], required=True)
    p.add_argument("--mode", choices=["full", "current", "geometry"], default="full")
    p.add_argument("--config", type=Path, default=Path("configs/navigation_crc.yaml"))
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
