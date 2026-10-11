"""Causal interaction records; scores are never passed back to policy.generate."""

import time
from dataclasses import asdict, dataclass, field

from sonomed_vlm.active_reader.agents.fixed_sampling import coarse_to_fine
from sonomed_vlm.active_reader.models.qwen3_adapter import Generation


@dataclass
class Trajectory:
    record: dict
    generations: list[Generation] = field(default_factory=list, repr=False)


def rollout(
    env, policy, seed, budget, sample=False, retain=False, method="prompted", initial_indices=None
):
    started = time.perf_counter()
    observation = env.reset(seed, budget, initial_indices=initial_indices)
    records, generations = [], []
    while not env.done:
        scripted = coarse_to_fine(observation) if method == "coarse_to_fine" else None
        if scripted is not None:
            output = asdict(scripted)
            turn_record = {
                "scripted_action": output,
                "image_presentations": 0,
                "generated_tokens": 0,
                "latency_seconds": 0.0,
                "context": observation.public_dict(),
            }
        else:
            generation = policy.generate(
                observation,
                sample=sample,
                seed=seed + env.turns,
                final_only=method == "coarse_to_fine",
                retain=retain,
            )
            output, turn_record = generation.text, dict(generation.record)
            generations.append(generation)
        observation, reward, _, info = env.step(output)
        turn_record.update(
            {
                "parsed": observation.history[-1],
                "private_reward": reward,
                "observed_after": list(observation.indices),
                "private_components": info["reward_components"],
            }
        )
        records.append(turn_record)
    record = {
        **env.private_info(),
        "seed": seed,
        "budget": budget,
        "method": method,
        "turns": records,
        "model_calls": len(generations),
        "image_presentations": sum(r["image_presentations"] for r in records),
        "generated_tokens": sum(r["generated_tokens"] for r in records),
        "latency_seconds": time.perf_counter() - started,
        "latency_scope": "episode including source verification, decoding, processing, and generation; model loading excluded",
        "generation_seconds": sum(
            r.get("generation_seconds", r["latency_seconds"]) for r in records
        ),
        "peak_memory_bytes": max((r.get("peak_memory_bytes") or 0 for r in records), default=0),
        "visual_input_tokens": sum(r.get("visual_input_tokens") or 0 for r in records),
    }
    return Trajectory(record, generations)


def grouped_rollouts(make_env, policy, initial_seed, sample_seed, budget, group_size):
    probe = make_env()
    initial = list(probe.reset(initial_seed, budget).indices)
    trajectories = [
        rollout(
            make_env(),
            policy,
            sample_seed + i * 1009,
            budget,
            sample=True,
            retain=True,
            initial_indices=initial,
        )
        for i in range(group_size)
    ]
    assert all(t.record["initial_indices"] == initial for t in trajectories)
    return trajectories
