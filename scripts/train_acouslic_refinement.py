#!/usr/bin/env python3
"""Development-only image-relative GRPO with conditional inspection credit.

The proximity reward is a training-only auxiliary objective, not an observed
anatomy label or a policy-invariant shaping claim. Evaluation uses original WFSS.
"""

import argparse
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from probe_acouslic_anchored_actions import AnchoredChoices

from sonomed_vlm.active_reader.acouslic.data import Cases
from sonomed_vlm.active_reader.acouslic.environment import initial, inspect, outcome
from sonomed_vlm.active_reader.acouslic.training import (
    frozen_digest,
    set_reference_to_policy,
    trainable,
)
from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.io import code_revision, digest, file_hash, read_json, write_json
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy
from sonomed_vlm.active_reader.training.grpo import advantages, grpo_loss, parameter_digest


def proximity(quality, sigma):
    """Smooth private training labels within each sweep; never across sweeps."""
    assert quality.shape == (840,) and sigma > 0
    values = np.zeros(840)
    for sweep in range(6):
        start = sweep * 140
        q = quality[start : start + 140]
        supported = np.flatnonzero(q > 0)
        if len(supported):
            distance = np.arange(140)[:, None] - supported[None, :]
            values[start : start + 140] = (
                q[supported][None, :] * np.exp(-0.5 * (distance / sigma) ** 2)
            ).max(axis=1)
    return values


def utility(quality, nearby, indices, weight):
    return float(quality[indices].max() + weight * nearby[indices].max())


def conditional_credit(terminal_returns, second_increments):
    terminal = np.asarray(terminal_returns, dtype=float)
    increments = np.asarray(second_increments, dtype=float)
    assert terminal.shape == increments.shape == (4, 2)
    assert np.isfinite(terminal).all() and np.isfinite(increments).all()
    first, first_zero = advantages(terminal.mean(axis=1).tolist())
    second, second_zero = [], []
    for branch in increments:
        adv, zero = advantages(branch.tolist())
        second.append(adv)
        second_zero.append(zero)
    return first, second, first_zero, second_zero


def tree(choices, case, start_seed, mode, sample_seed, weight, sigma):
    assert case["split"] == "train"
    alias = case["alias"]
    q, labels = choices.data.quality(alias)
    nearby = proximity(q, sigma)
    observed = initial(start_seed)
    initial_u = utility(q, nearby, observed, weight)
    initial_q = float(q[observed].max())
    branches, generations, returns, increments = [], [], [], []
    for branch in range(4):
        first = choices.inspect(
            alias, observed, observed, mode, sample_seed + branch * 1009, True, True
        )
        first_center = int(first.text)
        latest = inspect(first_center, observed)
        middle = observed + latest
        middle_u = utility(q, nearby, middle, weight)
        second_generations, second_rows, terminal, second_gains = [], [], [], []
        for continuation in range(2):
            second = choices.inspect(
                alias,
                middle,
                latest,
                mode,
                sample_seed + branch * 1009 + 101 + continuation * 31,
                True,
                True,
            )
            center = int(second.text)
            new = inspect(center, middle)
            selected = middle + new
            final_u = utility(q, nearby, selected, weight)
            terminal.append(final_u - initial_u)
            second_gains.append(final_u - middle_u)
            second_generations.append(second)
            second_rows.append(
                {
                    "center": center,
                    "returned_indices": new,
                    "input": second.record,
                    "exact_second_gain": float(q[selected].max() - q[middle].max()),
                    "auxiliary_second_gain": float(nearby[selected].max() - nearby[middle].max()),
                    "second_reward": final_u - middle_u,
                    "terminal_reward": final_u - initial_u,
                    **outcome(q, labels, selected),
                }
            )
        branches.append(
            {
                "first_center": first_center,
                "returned_indices": latest,
                "first_input": first.record,
                "exact_first_gain": float(q[middle].max()) - initial_q,
                "auxiliary_first_gain": float(nearby[middle].max() - nearby[observed].max()),
                "first_reward": middle_u - initial_u,
                "continuations": second_rows,
            }
        )
        generations.append((first, second_generations))
        returns.append(terminal)
        increments.append(second_gains)
    first_adv, second_adv, first_zero, second_zero = conditional_credit(returns, increments)
    # Equal loss weight for each decision stage, then average within that stage.
    terms = []
    for branch, (first, seconds) in enumerate(generations):
        terms.append((first, first_adv[branch], 0.5 / 4))
        for continuation, second in enumerate(seconds):
            terms.append((second, second_adv[branch][continuation], 0.5 / 8))
    record = {
        "alias": alias,
        "split": "train",
        "mode": mode,
        "start_seed": start_seed,
        "initial_indices": observed,
        "initial_wfss": initial_q,
        "initial_proximity": float(nearby[observed].max()),
        "branches": branches,
        "first_advantages": first_adv.tolist(),
        "second_advantages": [a.tolist() for a in second_adv],
        "first_zero_variance": first_zero,
        "second_zero_variance": second_zero,
        "mean_exact_gain": float(
            np.mean([s["best_wfss"] - initial_q for b in branches for s in b["continuations"]])
        ),
        "mean_training_reward": float(np.mean(returns)),
    }
    return record, terms


def evaluate(choices, cases, starts, mode, output):
    rows = []
    for case in cases:
        assert case["split"] == "tune"
        q, labels = choices.data.quality(case["alias"])
        for seed in starts:
            observed, latest = initial(seed), initial(seed)
            initial_q = float(q[observed].max())
            turns = []
            for turn in range(2):
                before = float(q[observed].max())
                gen = choices.inspect(case["alias"], observed, latest, mode, seed=turn * 31)
                center = int(gen.text)
                latest = inspect(center, observed)
                observed.extend(latest)
                turns.append(
                    {
                        "center": center,
                        "returned_indices": latest,
                        "input": gen.record,
                        "private_quality_increment": float(q[observed].max()) - before,
                    }
                )
            rows.append(
                {
                    "alias": case["alias"],
                    "split": "tune",
                    "mode": mode,
                    "seed": seed,
                    "initial_wfss": initial_q,
                    "turns": turns,
                    **outcome(q, labels, observed),
                }
            )
        write_json(output / "acquisition.json", rows)
    return float(np.mean([r["best_wfss"] for r in rows]))


def run(args):
    protocol = read_json(args.protocol)
    parent = read_json(args.parent / "campaign.json")
    assert protocol["parent_protocol_sha256"] == parent["protocol_sha256"]
    assert parent["code_revision"] == code_revision() == protocol["code_revision"]
    assert file_hash(args.data / "partition.json") == protocol["data_partition_sha256"]
    assert (
        file_hash(args.parent / "warm/checkpoint/policy/adapter_model.safetensors")
        == protocol["warm_checkpoint_sha256"]
    )
    for filename, expected in protocol["source_sha256"].items():
        assert file_hash(Path(filename)) == expected, filename
    assert read_json(args.parent / "anchored_probe/report.json")["policy_unchanged"]
    assert not (args.parent / "evaluation").exists(), (
        "Hold out evaluation until development is done"
    )
    data = Cases(args.data)
    assert (
        not data.allow_evaluation
        and len(data.subset("train")) == 32
        and len(data.subset("tune")) == 8
    )
    assert args.mode in protocol["arms"]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "campaign.json",
        {
            "protocol": protocol,
            "protocol_sha256": digest(protocol),
            "mode": args.mode,
            "stage": args.stage,
            "heldout_opened": False,
        },
    )
    config = load_config(args.config).model.model_copy(
        update={"dtype": "float32", "image_max_pixels": parent["protocol"]["selector_max_pixels"]}
    )
    assert (
        file_hash(Path(config.adapter) / "adapter_model.safetensors")
        == parent["protocol"]["original_adapter_sha256"]
    )
    torch.set_float32_matmul_precision("highest")
    policy = QwenPolicy(config, for_training=True)
    policy.load_policy(args.parent / "warm/checkpoint")
    set_reference_to_policy(policy)
    named = trainable(policy)
    names = {n for n, _ in named}
    params = [p for _, p in named]
    frozen = frozen_digest(policy)
    before_weights = parameter_digest(policy.model, lambda n, p: n in names)
    choices = AnchoredChoices(policy, data)
    before_score = None
    if args.stage == "arm":
        before_score = evaluate(
            choices,
            data.subset("tune"),
            protocol["evaluation_starts"],
            args.mode,
            args.output / "before_tune",
        )
        expected = read_json(args.parent / "anchored_probe/report.json")["methods"][args.mode][
            "mean_acquisition_wfss"
        ]
        assert abs(before_score - expected) < 1e-8, "Matched pre-RL replay must agree"
    assert parameter_digest(policy.model, lambda n, p: n in names) == before_weights
    optimizer = torch.optim.AdamW(params, lr=protocol["learning_rate"], weight_decay=0)
    trace = []
    steps = 2 if args.stage == "smoke" else protocol["updates"]
    for step in range(steps):
        epoch = step // 32
        order = data.subset("train")
        random.Random(protocol["training_seed"] + epoch).shuffle(order)
        case = order[step % 32]
        optimizer.zero_grad(set_to_none=True)
        record, terms = tree(
            choices,
            case,
            protocol["training_seed"] + step * 17,
            args.mode,
            protocol["training_seed"] + step * 100003,
            protocol["proximity_weight"],
            protocol["proximity_sigma_frames"],
        )
        losses, drifts, kls = [], [], []
        for gen, advantage, scale in terms:
            assert gen.record["context"]["protocol"] == "acouslic_anchored_actions_probe_v1"
            with torch.no_grad():
                reference = policy.log_probs(gen, "reference").detach()
            current = policy.log_probs(gen)
            old = gen.behavior_log_probs.to(current.device)
            drift = float((current.detach() - old).abs().max())
            assert drift <= protocol["max_behavior_logprob_drift"], drift
            loss, kl = grpo_loss(
                current,
                old,
                reference,
                advantage.to(current.device),
                protocol["clip_epsilon"],
                protocol["kl_weight"],
            )
            loss = loss * scale / len(current)
            assert torch.isfinite(loss)
            loss.backward()
            losses.append(float(loss.detach()))
            drifts.append(drift)
            kls.append(float(kl))
        norm = torch.nn.utils.clip_grad_norm_(params, 1)
        assert torch.isfinite(norm)
        optimizer.step()
        record.update(
            step=step + 1,
            max_logprob_drift=max(drifts),
            loss=sum(losses),
            mean_kl=float(np.mean(kls)),
            gradient_norm=float(norm),
        )
        write_json(args.output / "training/steps" / f"{step + 1:04d}.json", record)
        trace.append({k: v for k, v in record.items() if k != "branches"})
        print(
            f"Refinement {args.mode} {step + 1}/{steps}: exact_gain={record['mean_exact_gain']:.3f}, "
            f"first_signal={not record['first_zero_variance']}, second_signals={sum(not z for z in record['second_zero_variance'])}, drift={max(drifts):.7f}",
            flush=True,
        )
        if (step + 1) % 16 == 0 or step + 1 == steps:
            checkpoint = args.output / "training" / f"step-{step + 1:04d}"
            policy.save(checkpoint)
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    assert frozen_digest(policy) == frozen
    after_weights = parameter_digest(policy.model, lambda n, p: n in names)
    assert before_weights != after_weights, "No learning signal changed the adapter"
    policy.load_policy(checkpoint)
    assert parameter_digest(policy.model, lambda n, p: n in names) == after_weights
    after_score = None
    if args.stage == "arm":
        after_score = evaluate(
            choices,
            data.subset("tune"),
            protocol["evaluation_starts"],
            args.mode,
            args.output / "after_tune",
        )
    assert frozen_digest(policy) == frozen
    assert parameter_digest(policy.model, lambda n, p: n in names) == after_weights
    report = {
        "status": "complete",
        "stage": args.stage,
        "mode": args.mode,
        "updates": steps,
        "type": "on_policy_tree_GRPO_with_training_only_proximity_auxiliary",
        "before_tune_wfss": before_score,
        "after_tune_wfss": after_score,
        "first_reward_varying_groups": sum(not r["first_zero_variance"] for r in trace),
        "second_reward_varying_groups": sum(
            sum(not z for z in r["second_zero_variance"]) for r in trace
        ),
        "second_groups_total": 4 * steps,
        "max_behavior_logprob_drift": max(r["max_logprob_drift"] for r in trace),
        "frozen_vision_reference_unchanged": True,
        "save_reload_identical": True,
        "language_adapter_changed": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint / "policy/adapter_model.safetensors"),
        "protocol_sha256": digest(protocol),
        "heldout_opened": False,
        "breastvid_test_opened": False,
        "trace": trace,
    }
    write_json(args.output / "report.json", report)
    print({k: v for k, v in report.items() if k != "trace"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/navigation_crc.yaml"))
    p.add_argument("--mode", choices=["full", "current", "geometry"], required=True)
    p.add_argument("--stage", choices=["smoke", "arm"], required=True)
    run(p.parse_args())
