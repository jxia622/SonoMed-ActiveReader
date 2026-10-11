"""Inspection-only trajectories; frozen-reader outcomes provide private credit."""

import json
import math
import time

from sonomed_vlm.active_reader.agents.fixed_sampling import select_frames
from sonomed_vlm.active_reader.navigation.browser import NavigationBrowser, coverage_action
from sonomed_vlm.active_reader.training.rollouts import Trajectory

NONADAPTIVE = ("uniform", "random", "coverage", "privileged_area", "scripted_coverage")
HISTORIES = ("full", "current_batch", "current_frame")


def log_score(result, label):
    return math.log(max(1e-8, result["probabilities"][label]))


def select_episode(
    case,
    root,
    config,
    selector,
    reader,
    method,
    seed,
    budget,
    sample=False,
    sample_seed=42,
    retain=False,
    weights=None,
):
    started = time.perf_counter()
    mode = method if method in HISTORIES else "full"
    browser = NavigationBrowser(case, root, config, mode)
    observation = browser.reset(seed, budget)
    initial = list(observation.indices)
    initial_q = browser.env._potential()
    generations, turns, reader_calls = [], [], []
    weight = (weights or {"benign": 1.0, "malignant": 1.0})[case.label]
    initial_read = reader.score(browser.observation(reader=True)) if retain else None
    previous = initial_read
    if initial_read is not None:
        reader_calls.append(initial_read)
    scripted_actions = []
    if method == "scripted_coverage":
        while not browser.done:
            action = coverage_action(browser)
            observation = browser.step(action)
            scripted_actions.append(
                {"action": action, "returned_indices": observation.returned_indices}
            )
    elif method in NONADAPTIVE:
        selected = select_frames(
            method,
            case.num_frames,
            budget,
            initial,
            seed,
            case if method == "privileged_area" else None,
        )
        browser.env._reveal(selected)
        browser.done = True
    elif method in HISTORIES:
        while not browser.done:
            before_indices = list(observation.indices)
            action_targets = [
                {"action": a, "new_indices": indices} for a, indices in browser.proposals().values()
            ]
            generation = selector.generate(
                observation, sample=sample, seed=sample_seed + len(turns), retain=retain
            )
            # The environment validates the constrained output again.
            observation = browser.step(generation.text)
            turn = dict(generation.record)
            turn["public_action_frame_sets"] = action_targets
            turn.update(
                action=json.loads(generation.text),
                selected_before=before_indices,
                selected_after=list(observation.indices),
                credit_target="inspection tokens only; no reader or diagnosis loss",
            )
            if retain:
                after = reader.score(browser.observation(reader=True))
                reader_calls.append(after)
                turn["private_reader_logscore_increment"] = log_score(
                    after, case.label
                ) - log_score(previous, case.label)
                turn["private_reader_before"] = previous
                turn["private_reader_after"] = after
                previous = after
            generations.append(generation)
            turns.append(turn)
    else:
        raise ValueError("Unknown selection method")
    final = previous if retain and turns else reader.score(browser.observation(reader=True))
    if not retain or not turns:
        reader_calls.append(final)
    if len(browser.env.visited) != min(budget, case.num_frames):
        raise RuntimeError("Refusing an under-budget navigation comparison")
    gain = (
        log_score(final, case.label) - log_score(initial_read, case.label) if initial_read else None
    )
    bonus = config.reward.relevance_weight * max(0.0, browser.env._potential() - initial_q)
    record = {
        "video_id": case.video_id,
        "group_id": case.group_id,
        "split": case.split,
        "label": case.label,
        "prediction": final["prediction"],
        "method": method,
        "budget": budget,
        "seed": seed,
        "sample_seed": sample_seed,
        "initial_indices": initial,
        "selected_indices": sorted(browser.env.visited),
        "unique_frames": len(browser.env.visited),
        "fraction_viewed": len(browser.env.visited) / case.num_frames,
        "inspection_calls": len(turns) + len(scripted_actions),
        "decision_turns": len(turns) + len(scripted_actions),
        "scripted_actions": scripted_actions,
        "invalid_actions": 0,
        "invalid_terminal": False,
        "revisits": 0,
        "turns": turns,
        "reader": final,
        "reader_calls": reader_calls,
        "reader_true_label_log_score": log_score(final, case.label),
        "reader_true_label_probability": final["probabilities"][case.label],
        "private_reader_gain": gain,
        "private_box_bonus": bonus,
        "return": weight * gain + bonus if gain is not None else None,
        "credit_assignment": "shared terminal group-relative return to each inspection token; per-turn deltas are audit only",
        "selector_model_calls": len(turns),
        "reader_model_calls": sum(r["model_calls"] for r in reader_calls),
        "executed_reader_model_calls": sum(r["executed_model_calls"] for r in reader_calls),
        "model_calls": len(turns) + sum(r["model_calls"] for r in reader_calls),
        "image_presentations": sum(t["image_presentations"] for t in turns)
        + sum(r["image_presentations"] for r in reader_calls),
        "selector_image_presentations": sum(t["image_presentations"] for t in turns),
        "generated_tokens": sum(t["generated_tokens"] for t in turns),
        "latency_seconds": time.perf_counter() - started,
        "privileged": method == "privileged_area",
        "reward_has_observation_cost": False,
    }
    # Reader/score/annotation data stay in this private record, never in Observation.
    reader.assert_frozen()
    return Trajectory(record, generations)
