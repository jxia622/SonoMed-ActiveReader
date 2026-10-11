"""Matched-start baselines and explicit, counted dense chunk aggregation."""

import time
from collections import Counter

from sonomed_vlm.active_reader.agents.fixed_sampling import select_frames
from sonomed_vlm.active_reader.env.replay import Replay
from sonomed_vlm.active_reader.env.rewards import relevance
from sonomed_vlm.active_reader.training.rollouts import rollout


def fixed_read(case, root, policy, config, method, seed, budget, selected=None):
    if selected is not None and len(set(selected)) > budget:
        raise ValueError(
            "Fixed-reader selection exceeds its frame budget; use the chunked dense reference separately"
        )
    started = time.perf_counter()
    env = Replay(case, root, config.environment, config.reward)
    initial = list(env.reset(seed, budget).indices)
    indices = (
        selected
        if selected is not None
        else select_frames(
            method,
            case.num_frames,
            budget,
            initial,
            seed,
            case if method == "privileged_area" else None,
        )
    )
    if not set(initial) <= set(indices) and selected is None:
        raise ValueError("Baseline omitted mandatory starting images")
    observation = env.reset(seed, max(budget, len(indices)), initial_indices=indices)
    env.initial_indices = initial if selected is None else []
    if set(indices) - set(initial):
        env.calls = 1
        env.components["calls"] -= config.reward.inspection_call_cost
    generation = policy.generate(observation, sample=False, seed=seed, final_only=True)
    _, reward, _, _ = env.step(generation.text)
    # Fixed reader gets exactly one final-answer attempt, including invalid output.
    if not env.done:
        env._finish()
    record = {
        **env.private_info(),
        "method": method,
        "budget": budget,
        "seed": seed,
        "privileged": method == "privileged_area",
        "model_calls": 1,
        "turns": [{**generation.record, "private_reward": reward}],
        **{
            key: generation.record.get(key, 0)
            for key in (
                "image_presentations",
                "visual_input_tokens",
                "generated_tokens",
                "latency_seconds",
                "peak_memory_bytes",
            )
        },
    }
    record["generation_seconds"] = generation.record.get(
        "generation_seconds", generation.record["latency_seconds"]
    )
    record["latency_seconds"] = time.perf_counter() - started
    record["latency_scope"] = (
        "episode including source verification, decoding, processing, and generation; model loading excluded"
    )
    return record


def dense_read(case, root, policy, config, seed, chunk_size):
    started = time.perf_counter()
    initial = list(Replay(case, root, config.environment).reset(seed, chunk_size).indices)
    order = initial + [i for i in range(case.num_frames) if i not in initial]
    chunks = []
    for offset in range(0, len(order), chunk_size):
        indices = sorted(order[offset : offset + chunk_size])
        chunks.append(
            fixed_read(case, root, policy, config, "dense", seed, chunk_size, selected=indices)
        )
    votes = Counter(c["prediction"] or "invalid" for c in chunks)
    ranked = votes.most_common()
    prediction = ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else None
    if prediction == "invalid":
        prediction = None
    q, relevance_status = relevance(case)
    result = {
        **chunks[0],
        "prediction": prediction,
        "invalid_terminal": prediction is None,
        "selected_indices": list(range(case.num_frames)),
        "initial_indices": initial,
        "unique_frames": case.num_frames,
        "fraction_viewed": 1.0,
        "effective_frame_budget": case.num_frames,
        "chunk_size": chunk_size,
        "aggregation": "plurality over chunk categorical outputs; ties/invalid plurality fail",
        "dense_chunks": chunks,
        "turns": [t for c in chunks for t in c["turns"]],
        "return": None,
        "reward_components": {},
        "evidence_frames": [],
        "relevance_status": relevance_status,
        "max_relevance": max(q) if q else None,
        "mean_relevance": sum(q) / len(q) if q else None,
        "peak_memory_bytes": max((c.get("peak_memory_bytes") or 0 for c in chunks), default=0),
        **{
            key: sum(c.get(key) or 0 for c in chunks)
            for key in (
                "model_calls",
                "decision_turns",
                "inspection_calls",
                "image_presentations",
                "visual_input_tokens",
                "generated_tokens",
                "latency_seconds",
                "generation_seconds",
            )
        },
    }
    result["latency_seconds"] = time.perf_counter() - started
    return result


def run_case(case, root, policy, config, method, seed, budget):
    if method in ("first_frame", "center_frame"):
        index = 0 if method == "first_frame" else case.num_frames // 2
        result = fixed_read(case, root, policy, config, method, seed, 1, selected=[index])
        result["initial_indices"] = [index]
        result["shortcut_probe"] = True
        return result
    if method == "dense":
        return dense_read(case, root, policy, config, seed, budget)
    if method in ("prompted", "coarse_to_fine"):
        return rollout(
            Replay(case, root, config.environment, config.reward),
            policy,
            seed,
            budget,
            method=method,
        ).record
    return fixed_read(case, root, policy, config, method, seed, budget)
