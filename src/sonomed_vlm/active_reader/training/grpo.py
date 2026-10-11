"""Small on-policy multi-turn GRPO, with a frozen ultrasound SFT reference.

This is a correctness-first single-GPU integration, not a throughput trainer.
Every generated action token is optimized with its own causal image context.
"""

import hashlib
import math
import random
from pathlib import Path

import torch

from sonomed_vlm.active_reader.env.replay import Replay
from sonomed_vlm.active_reader.env.rewards import class_weights
from sonomed_vlm.active_reader.io import digest, read_json, write_json
from sonomed_vlm.active_reader.training.rollouts import grouped_rollouts


def advantages(returns):
    values = torch.tensor(returns, dtype=torch.float32)
    std = values.std(unbiased=False)
    return (values - values.mean()) / (std + 1e-8), bool(std < 1e-8)


def grpo_loss(current, old, reference, advantage, clip_epsilon=0.2, kl_weight=0.02):
    if current.shape != old.shape or current.shape != reference.shape or not current.numel():
        raise ValueError("GRPO needs aligned nonempty assistant-token likelihoods")
    ratio = (current - old.detach()).exp()
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
    )
    delta = reference.detach() - current
    kl = delta.exp() - delta - 1
    return (-surrogate + kl_weight * kl).sum(), kl.detach().mean()


def parameter_digest(model, predicate):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if predicate(name, value):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def train(policy, cases, root, config, output: Path, provenance, resume=None):
    if not cases or any(c.split != "train" for c in cases):
        raise ValueError("RL can sample training cases only")
    policy.use_adapter("policy")
    named = [(n, p) for n, p in policy.model.named_parameters() if p.requires_grad]
    if not named or any(
        "lora_" not in n or any(x in n.lower() for x in ("visual", "vision", ".reference."))
        for n, _ in named
    ):
        raise ValueError("Trainable set must contain language-side policy LoRA parameters only")
    parameters = [p for _, p in named]
    trainable_names = {n for n, _ in named}
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate, weight_decay=0.0)
    signature = digest(
        {
            "provenance": provenance,
            "environment": config.environment.model_dump(),
            "reward": config.reward.model_dump(),
            "training": config.training.model_dump(exclude={"steps"}),
            "seed": config.seed,
            "cases": [c.video_id for c in cases],
        }
    )
    start = 0
    if resume:
        state = read_json(resume / "state.json")
        if state["signature"] != signature:
            raise ValueError("Resume provenance/configuration does not match")
        policy.load_policy(resume)
        optimizer.load_state_dict(
            torch.load(
                resume / "optimizer.pt",
                map_location=next(policy.model.parameters()).device,
                weights_only=True,
            )
        )
        start = state["completed_steps"]
    output.mkdir(parents=True, exist_ok=True)
    weights = class_weights(cases, config.reward.class_balance)

    def frozen_check(n, p):
        return ".reference." in n or any(x in n.lower() for x in ("visual", "vision"))

    frozen_before = parameter_digest(policy.model, frozen_check)
    policy_before = parameter_digest(policy.model, lambda n, p: n in trainable_names)
    total_updates, last_checkpoint = 0, None
    for step in range(start, config.training.steps):
        optimizer.zero_grad(set_to_none=True)
        records, group_stats, losses = [], [], []
        for accumulation in range(config.training.accumulation_groups):
            index = step * config.training.accumulation_groups + accumulation
            case = random.Random(config.seed + index).choice(cases)

            def factory(case=case):
                return Replay(case, root, config.environment, config.reward, weights)

            trajectories = grouped_rollouts(
                factory,
                policy,
                config.seed + index,
                config.seed + index * 100003,
                config.environment.max_unique_frames,
                config.training.group_size,
            )
            adv, zero = advantages([t.record["return"] for t in trajectories])
            max_drift = 0.0
            for t, advantage in zip(trajectories, adv, strict=True):
                token_count = sum(g.tokens.numel() for g in t.generations)
                if not token_count:
                    raise ValueError("Empty policy trajectory cannot receive a gradient")
                for generation in t.generations:
                    with torch.no_grad():
                        reference = policy.log_probs(generation, "reference").detach()
                    current = policy.log_probs(generation)
                    old = generation.behavior_log_probs.to(current.device)
                    drift = float((current.detach() - old).abs().max())
                    max_drift = max(max_drift, drift)
                    # No updates occur between sampling and this likelihood reconstruction.
                    if drift > 0.15:
                        raise ValueError(
                            "Rollout/teacher-forcing probabilities disagree; Qwen compatibility gate failed"
                        )
                    loss, _ = grpo_loss(
                        current,
                        old,
                        reference,
                        advantage.to(current.device),
                        config.training.clip_epsilon,
                        config.training.kl_weight,
                    )
                    loss = loss / (
                        token_count
                        * config.training.group_size
                        * config.training.accumulation_groups
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite policy loss; refusing optimizer update")
                    loss.backward()
                    losses.append(float(loss.detach()))
                records.append(t.record)
            outputs = [
                turn.get("raw_output", "") for t in trajectories for turn in t.record["turns"]
            ]
            from collections import Counter

            counts = Counter(outputs)
            entropy = -sum((n / len(outputs)) * math.log(n / len(outputs)) for n in counts.values())
            group_stats.append(
                {
                    "zero_variance": zero,
                    "mean_return": sum(t.record["return"] for t in trajectories)
                    / len(trajectories),
                    "action_string_entropy": entropy,
                    "unique_action_strings": len(counts),
                    "immediate_stop_rate": sum(
                        t.record["decision_turns"] == 1 for t in trajectories
                    )
                    / len(trajectories),
                    "invalid_action_rate": sum(t.record["invalid_actions"] for t in trajectories)
                    / sum(t.record["decision_turns"] for t in trajectories),
                    "max_behavior_logprob_drift": max_drift,
                }
            )
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.max_grad_norm)
        if not torch.isfinite(norm):
            raise ValueError("Non-finite gradient; refusing optimizer update")
        optimizer.step()
        total_updates += 1
        last_checkpoint = output / f"step-{step + 1:06d}"
        policy.save(last_checkpoint)
        torch.save(optimizer.state_dict(), last_checkpoint / "optimizer.pt")
        write_json(last_checkpoint / "trajectories.json", records)
        write_json(
            last_checkpoint / "state.json",
            {
                "signature": signature,
                "completed_steps": step + 1,
                "provenance": provenance,
                "groups": group_stats,
                "loss": sum(losses),
                "gradient_norm": float(norm),
                "trainable_parameters": sorted(trainable_names),
            },
        )
        print(
            f"RL step {step + 1}: gradient_norm={float(norm):.5g}, zero_variance_groups={sum(g['zero_variance'] for g in group_stats)}",
            flush=True,
        )
    frozen_after = parameter_digest(policy.model, frozen_check)
    changed = parameter_digest(policy.model, lambda n, p: n in trainable_names) != policy_before
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen vision/reference parameters changed")
    report = {
        "updates": total_updates,
        "language_adapter_changed": changed,
        "frozen_vision_and_reference_unchanged": True,
        "checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "empirical_status": "training-only debugging; no generalization claim",
    }
    if last_checkpoint and records:
        # Save/reload weight equivalence; real behavioral gate is also exercised by smoke-test-model.
        before_reload = parameter_digest(policy.model, lambda n, p: n in trainable_names)
        policy.load_policy(last_checkpoint)
        report["save_reload_weights_identical"] = before_reload == parameter_digest(
            policy.model, lambda n, p: n in trainable_names
        )
        if not report["save_reload_weights_identical"]:
            raise RuntimeError("Saved adapter does not reload equivalently")
    write_json(output / "training_report.json", report)
    return report
