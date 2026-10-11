"""GRPO over inspection actions only; the separate frozen reader supplies returns."""

import random
from pathlib import Path

import torch

from sonomed_vlm.active_reader.env.rewards import class_weights
from sonomed_vlm.active_reader.io import digest, read_json, write_json
from sonomed_vlm.active_reader.navigation.rollout import select_episode
from sonomed_vlm.active_reader.training.grpo import advantages, grpo_loss, parameter_digest


def reset_selector(selector):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    state = get_peft_model_state_dict(selector.model, adapter_name="reference")
    set_peft_model_state_dict(selector.model, state, adapter_name="policy")
    selector.use_adapter("policy")


def train_navigation(
    selector, reader, cases, root, config, output, provenance, history_mode="full", resume=None
):
    if not cases or any(c.split != "train" for c in cases):
        raise ValueError("Navigation updates use training cases only")
    if config.environment.max_unique_frames <= config.environment.initial_images:
        raise ValueError("Navigation RL needs a budget allowing inspection actions")
    output = Path(output)
    signature = digest(
        {
            "provenance": provenance,
            "config": config.model_dump(mode="json", exclude={"training": {"steps"}}),
            "history_mode": history_mode,
            "cases": [c.video_id for c in cases],
        }
    )
    reset_selector(selector)
    named = [(n, p) for n, p in selector.model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in n or ".policy." not in n or "visual" in n for n, p in named):
        raise ValueError("Only language-side policy LoRA may train")
    names = {n for n, p in named}
    parameters = [p for n, p in named]
    original_policy = parameter_digest(selector.model, lambda n, p: n in names)
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate, weight_decay=0)
    start = 0
    if resume:
        state = read_json(Path(resume) / "state.json")
        if state["signature"] != signature:
            raise ValueError("Navigation resume provenance differs")
        selector.load_policy(Path(resume))
        optimizer.load_state_dict(
            torch.load(
                Path(resume) / "optimizer.pt",
                weights_only=True,
                map_location=next(selector.model.parameters()).device,
            )
        )
        start = state["completed_steps"]
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output or explicitly resume navigation training")
    output.mkdir(parents=True, exist_ok=True)
    weights = class_weights(cases, config.reward.class_balance)
    buckets = [[c for c in cases if c.label == label] for label in ("benign", "malignant")]
    for index, bucket in enumerate(buckets):
        random.Random(config.seed + index).shuffle(bucket)
    # Group normalization cancels a common class weight within a trajectory group.
    # Balanced case sampling supplies actual class balance across optimizer steps.
    order = [bucket[i % len(bucket)] for i in range(max(map(len, buckets))) for bucket in buckets]

    def frozen(n, p):
        return "visual" in n or "vision" in n or ".reference." in n

    frozen_before = parameter_digest(selector.model, frozen)
    reader_before = parameter_digest(reader.policy.model, lambda n, p: True)
    policy_before = original_policy
    trace = []
    checkpoint = Path(resume) if resume else None
    for step in range(start, config.training.steps):
        case = order[step % len(order)]
        optimizer.zero_grad(set_to_none=True)
        trajectories = [
            select_episode(
                case,
                root,
                config,
                selector,
                reader,
                history_mode,
                config.seed + step,
                config.environment.max_unique_frames,
                sample=True,
                sample_seed=config.seed + step * 100003 + i * 1009,
                retain=True,
                weights=weights,
            )
            for i in range(config.training.group_size)
        ]
        assert len({tuple(t.record["initial_indices"]) for t in trajectories}) == 1
        adv, zero = advantages([t.record["return"] for t in trajectories])
        losses, drifts = [], []
        for trajectory, advantage in zip(trajectories, adv, strict=True):
            for generation in trajectory.generations:
                if generation.record["context"]["protocol"] != "navigation":
                    raise RuntimeError("Only inspection generations may receive policy loss")
                with torch.no_grad():
                    ref = selector.log_probs(generation, "reference").detach()
                current = selector.log_probs(generation)
                old = generation.behavior_log_probs.to(current.device)
                drift = float((current.detach() - old).abs().max())
                drifts.append(drift)
                if drift > config.navigation.max_behavior_logprob_drift:
                    raise ValueError(
                        f"Navigation sampling/teacher likelihood mismatch: {drift:.5f}"
                    )
                loss, _ = grpo_loss(
                    current,
                    old,
                    ref,
                    advantage.to(current.device),
                    config.training.clip_epsilon,
                    config.training.kl_weight,
                )
                # Average by trajectories; sum over all inspection decisions.
                # Forced singleton-support tokens have zero log probability and gradient.
                # No JSON-length or trajectory-length normalization.
                loss = loss / config.training.group_size
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite navigation loss")
                loss.backward()
                losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.max_grad_norm)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite navigation gradient")
        optimizer.step()
        reader.assert_frozen()
        summary = {
            "step": step + 1,
            "video_id": case.video_id,
            "gradient_norm": float(norm),
            "loss": sum(losses),
            "zero_variance_group": zero,
            "max_behavior_logprob_drift": max(drifts),
            "mean_return": sum(t.record["return"] for t in trajectories) / len(trajectories),
            "reader_gain": [t.record["private_reader_gain"] for t in trajectories],
            "box_bonus": [t.record["private_box_bonus"] for t in trajectories],
            "inspections": [len(t.generations) for t in trajectories],
        }
        trace.append(summary)
        write_json(
            output / "steps" / f"{step + 1:06d}.json",
            {"summary": summary, "trajectories": [t.record for t in trajectories]},
        )
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == config.training.steps:
            checkpoint = output / f"step-{step + 1:06d}"
            selector.save(checkpoint)
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
            write_json(
                checkpoint / "state.json",
                {
                    "signature": signature,
                    "completed_steps": step + 1,
                    "provenance": provenance,
                    "history_mode": history_mode,
                },
            )
        print(
            f"navigation RL {history_mode} step={step + 1} grad={float(norm):.4g} zero={zero} drift={max(drifts):.4f}",
            flush=True,
        )
    frozen_after = parameter_digest(selector.model, frozen)
    reader_after = parameter_digest(reader.policy.model, lambda n, p: True)
    after = parameter_digest(selector.model, lambda n, p: n in names)
    if frozen_before != frozen_after or reader_before != reader_after:
        raise RuntimeError("Frozen reference, vision, or reader weights changed")
    if checkpoint:
        selector.load_policy(checkpoint)
        if parameter_digest(selector.model, lambda n, p: n in names) != after:
            raise RuntimeError("Navigation checkpoint reload differs")
    report = {
        "signature": signature,
        "history_mode": history_mode,
        "steps": trace,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "language_adapter_changed": after != policy_before,
        "frozen_reader_unchanged": True,
        "frozen_reference_and_vision_unchanged": True,
        "save_reload_identical": checkpoint is not None,
        "credit": "inspection tokens only, shared trajectory advantage; reader deltas audited per turn",
        "relevance_weight": config.reward.relevance_weight,
        "test_opened": False,
    }
    write_json(output / "training_report.json", report)
    return report
