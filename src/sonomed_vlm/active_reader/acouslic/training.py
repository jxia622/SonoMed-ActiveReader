"""Plane-recognition warm start and bounded on-policy acquisition GRPO."""

import random
import string
from pathlib import Path

import numpy as np
import torch

from sonomed_vlm.active_reader.acouslic.environment import initial, inspect, outcome
from sonomed_vlm.active_reader.acouslic.policy import Choices
from sonomed_vlm.active_reader.io import digest, write_json
from sonomed_vlm.active_reader.training.grpo import advantages, grpo_loss, parameter_digest


def trainable(policy):
    policy.use_adapter("policy")
    named = [(n, p) for n, p in policy.model.named_parameters() if p.requires_grad]
    assert named and all(
        "lora_" in n
        and ".policy." in n
        and not any(k in n for k in ("visual", "vision", "projector"))
        for n, _ in named
    )
    return named


def frozen_digest(policy):
    return parameter_digest(
        policy.model, lambda n, p: ".reference." in n or "visual" in n or "vision" in n
    )


def set_reference_to_policy(policy):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    state = {
        k: v.detach().clone()
        for k, v in get_peft_model_state_dict(policy.model, adapter_name="policy").items()
    }
    set_peft_model_state_dict(policy.model, state, adapter_name="reference")
    policy.use_adapter("policy")


def control(data, alias, seed):
    quality, _ = data.quality(alias)
    if quality.max() == 0 or (quality == 0).sum() < 3:
        return None
    rng = random.Random(seed)
    positive = rng.choice(np.flatnonzero(quality == quality.max()).tolist())
    indices = [positive, *rng.sample(np.flatnonzero(quality == 0).tolist(), 3)]
    rng.shuffle(indices)
    return indices, positive


def recognition(choices, cases, output, adapter="policy"):
    records = []
    for case in cases:
        for seed in (701, 1901):
            fixture = control(choices.data, case["alias"], seed)
            if fixture is None:
                continue
            indices, positive = fixture
            for order in ("original", "reverse"):
                shown = indices if order == "original" else list(reversed(indices))
                result = choices.recognize(
                    case["alias"], shown, "recognition_control", adapter=adapter
                )
                selected = shown[string.ascii_uppercase.index(result.text)]
                records.append(
                    {
                        "alias": case["alias"],
                        "seed": seed,
                        "order": order,
                        "correct": selected == positive,
                        "selected": selected,
                        "input": result.record,
                    }
                )
        write_json(output / "recognition_inputs.json", records)
    accuracy = float(np.mean([r["correct"] for r in records])) if records else 0.0
    pairs = {(r["alias"], r["seed"]) for r in records}
    stable = sum(
        all(r["correct"] for r in records if (r["alias"], r["seed"]) == pair) for pair in pairs
    )
    report = {
        "accuracy": accuracy,
        "correct": sum(r["correct"] for r in records),
        "presentations": len(records),
        "stable_correct_pairs": stable,
        "pairs": len(pairs),
        "stable_fraction": stable / len(pairs) if pairs else 0,
        "subjects": len({r["alias"] for r in records}),
        "annotation_based_control": True,
    }
    write_json(output / "report.json", report)
    return report


def warm_start(policy, data, protocol, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    choices = Choices(policy, data)
    train, tune = data.subset("train"), data.subset("tune")
    assert len(train) == 32 and len(tune) == 8
    before = recognition(choices, tune, output / "before")
    frozen = frozen_digest(policy)
    named = trainable(policy)
    names = {n for n, _ in named}
    optimizer = torch.optim.AdamW(
        [p for _, p in named], lr=protocol["warm_learning_rate"], weight_decay=0
    )
    rows = []
    order = list(train)
    random.Random(protocol["training_seed"]).shuffle(order)
    for step in range(protocol["warm_updates"]):
        case = order[step % len(order)]
        fixture = control(data, case["alias"], protocol["training_seed"] + step * 313)
        if fixture is None:
            rows.append(
                {
                    "step": step + 1,
                    "alias": case["alias"],
                    "skipped": "No annotated positive/distractor set; subject retained in evaluation",
                }
            )
            continue
        indices, positive = fixture
        target = string.ascii_uppercase[indices.index(positive)]
        example = choices.recognize(
            case["alias"], indices, "supervised_recognition", target=target, retain=True
        )
        optimizer.zero_grad(set_to_none=True)
        loss = -policy.log_probs(example).sum()
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1)
        assert torch.isfinite(norm)
        optimizer.step()
        rows.append(
            {
                "step": step + 1,
                "alias": case["alias"],
                "split": "train",
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "input": example.record,
            }
        )
        write_json(output / "steps" / f"{step + 1:04d}.json", rows[-1])
        if (step + 1) % 8 == 0:
            print(
                f"Recognition warm start {step + 1}/{protocol['warm_updates']}; loss {float(loss.detach()):.4f}",
                flush=True,
            )
    after = recognition(choices, tune, output / "after")
    assert frozen == frozen_digest(policy)
    checkpoint = output / "checkpoint"
    policy.save(checkpoint)
    before_reload = parameter_digest(policy.model, lambda n, p: n in names)
    policy.load_policy(checkpoint)
    assert before_reload == parameter_digest(policy.model, lambda n, p: n in names)
    gate = (
        after["subjects"] >= 6
        and after["accuracy"] >= protocol["recognition_min_accuracy"]
        and after["stable_fraction"] >= protocol["recognition_min_stability"]
    )
    report = {
        "type": "supervised_warm_start_not_RL",
        "before": before,
        "after": after,
        "gate_passed": gate,
        "checkpoint": str(checkpoint),
        "updates": sum("loss" in r for r in rows),
        "frozen_vision_reference_unchanged": True,
        "save_reload_identical": True,
        "trainable_parameter_names": sorted(names),
        "protocol_sha256": digest(protocol),
        "heldout_opened": False,
    }
    write_json(output / "report.json", report)
    return report


def episode(choices, case, seed, mode, sample=False, sample_seed=0, retain=False):
    alias = case["alias"]
    quality, labels = choices.data.quality(alias)
    observed, latest = initial(seed), initial(seed)
    initial_value = float(quality[observed].max())
    generations, turns = [], []
    for turn in range(2):
        previous = float(quality[observed].max())
        generation = choices.inspect(
            alias, observed, latest, mode, sample_seed + turn * 31, sample, retain
        )
        center = int(generation.text)
        latest = inspect(center, observed)
        observed.extend(latest)
        generations.append(generation)
        turns.append(
            {
                "center": center,
                "returned_indices": latest,
                "private_quality_increment": float(quality[observed].max()) - previous,
                "input": generation.record,
            }
        )
    result = outcome(quality, labels, observed)
    result.update(
        alias=alias,
        split=case["split"],
        seed=seed,
        mode=mode,
        initial_indices=initial(seed),
        initial_wfss=initial_value,
        reward=result["best_wfss"] - initial_value,
        turns=turns,
    )
    assert abs(sum(t["private_quality_increment"] for t in turns) - result["reward"]) < 1e-8
    return result, generations


def train_rl(policy, data, protocol, output, warm_checkpoint, mode):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    policy.load_policy(warm_checkpoint)
    set_reference_to_policy(policy)
    named = trainable(policy)
    names = {n for n, _ in named}
    frozen = frozen_digest(policy)
    before = parameter_digest(policy.model, lambda n, p: n in names)
    optimizer = torch.optim.AdamW(
        [p for _, p in named], lr=protocol["rl_learning_rate"], weight_decay=0
    )
    choices = Choices(policy, data)
    order = data.subset("train")
    assert len(order) == 32 and all(r["split"] == "train" for r in order)
    random.Random(protocol["training_seed"]).shuffle(order)
    trace = []
    for step in range(protocol["rl_updates"]):
        case = order[step % len(order)]
        start_seed = protocol["training_seed"] + step * 17
        optimizer.zero_grad(set_to_none=True)
        trajectories = [
            episode(
                choices,
                case,
                start_seed,
                mode,
                True,
                protocol["training_seed"] + step * 100003 + i * 1009,
                True,
            )
            for i in range(protocol["group_size"])
        ]
        adv, zero = advantages([r["reward"] for r, _ in trajectories])
        losses, drifts, kl_values = [], [], []
        for (_record, generations), advantage in zip(trajectories, adv, strict=True):
            for generation in generations:
                assert generation.record["context"]["protocol"] == "acouslic_navigation"
                with torch.no_grad():
                    reference = policy.log_probs(generation, "reference").detach()
                current = policy.log_probs(generation)
                old = generation.behavior_log_probs.to(current.device)
                drift = float((current.detach() - old).abs().max())
                if drift > protocol["max_behavior_logprob_drift"]:
                    raise ValueError(
                        f"Rollout/teacher-forcing drift {drift:.6f} exceeds locked threshold"
                    )
                loss, kl = grpo_loss(
                    current,
                    old,
                    reference,
                    advantage.to(current.device),
                    0.2,
                    protocol["kl_weight"],
                )
                loss = loss / protocol["group_size"]
                assert torch.isfinite(loss)
                loss.backward()
                losses.append(float(loss.detach()))
                drifts.append(drift)
                kl_values.append(float(kl))
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1)
        assert torch.isfinite(norm)
        optimizer.step()
        row = {
            "step": step + 1,
            "alias": case["alias"],
            "split": "train",
            "mode": mode,
            "mean_reward": float(np.mean([r["reward"] for r, _ in trajectories])),
            "zero_variance": zero,
            "max_logprob_drift": max(drifts),
            "loss": sum(losses),
            "mean_kl": float(np.mean(kl_values)),
            "gradient_norm": float(norm),
        }
        trace.append(row)
        write_json(
            output / "steps" / f"{step + 1:04d}.json",
            {**row, "trajectories": [r for r, _ in trajectories]},
        )
        print(
            f"GRPO {mode} {step + 1}/{protocol['rl_updates']}: reward={row['mean_reward']:.3f}, zero_variance={zero}, drift={max(drifts):.6f}",
            flush=True,
        )
        if (step + 1) % 8 == 0 or step + 1 == protocol["rl_updates"]:
            policy.save(output / f"step-{step + 1:04d}")
    assert frozen == frozen_digest(policy)
    after = parameter_digest(policy.model, lambda n, p: n in names)
    checkpoint = output / f"step-{protocol['rl_updates']:04d}"
    policy.load_policy(checkpoint)
    assert after == parameter_digest(policy.model, lambda n, p: n in names)
    report = {
        "type": "on_policy_GRPO",
        "mode": mode,
        "updates": len(trace),
        "steps": trace,
        "checkpoint": str(checkpoint),
        "language_adapter_changed": before != after,
        "frozen_vision_reference_unchanged": True,
        "save_reload_identical": True,
        "reference": "matched supervised warm-start checkpoint",
        "credit": "shared terminal acquisition-gain advantage to inspection tokens; per-action increments telescope",
        "trainable_parameter_names": sorted(names),
        "protocol_sha256": digest(protocol),
        "heldout_opened": False,
    }
    write_json(output / "report.json", report)
    return report
