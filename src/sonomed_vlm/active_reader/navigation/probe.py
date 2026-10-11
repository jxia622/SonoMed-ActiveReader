"""Train-only counterfactual inspections with matched final evidence budgets.

Labels and reader scores exist only in private records. Candidate actions and the
common completion rule use public browser state. This is an action-ranking probe,
not an oracle available to a deployed selector.
"""

import random
from dataclasses import replace
from statistics import mean

from PIL import Image

from sonomed_vlm.active_reader.io import digest, read_json, write_json
from sonomed_vlm.active_reader.navigation.browser import (
    NavigationBrowser,
    action_key,
    coverage_action,
)
from sonomed_vlm.active_reader.navigation.rollout import log_score

PROBE_VERSION = 1
CONTEXTS = ("full", "current_frame", "shuffled_history", "blank_history", "no_visual")


def partition_cases(cases, count=8, seed=42):
    """Balanced fit/validation subsets entirely inside train, disjoint by group."""
    if count < 2 or count % 2:
        raise ValueError("Probe subset size must be positive and even")
    if not cases or any(c.split != "train" for c in cases):
        raise ValueError("Evidence probes use train cases only; dev/test are excluded")
    used = set()
    result = {"fit": [], "validation": []}
    for label in ("benign", "malignant"):
        bucket = sorted(
            [c for c in cases if c.label == label],
            key=lambda c: digest([seed, c.group_id, c.video_id]),
        )
        for role in result:
            for c in bucket:
                if c.group_id not in used:
                    result[role].append(c)
                    used.add(c.group_id)
                    if sum(x.label == label for x in result[role]) == count // 2:
                        break
            else:
                raise ValueError("Not enough distinct training groups for both probe subsets")
    return result


def replay_state(case, root, config, seed, budget, prefix):
    if case.split != "train":
        raise ValueError("Probe reconstruction requires the train split before image access")
    b = NavigationBrowser(case, root, config)
    b.reset(seed, budget)
    for action in prefix:
        b.step(action)
    return b


def candidate_menu(browser, count=8, seed=42):
    """Label-independent menu, including coverage and varied legal action families."""
    candidates = [(a, indices) for a, indices in browser.proposals().values() if len(indices) == 4]
    if len(candidates) < 2:
        raise ValueError("Probe requires two legal four-new-frame inspections")
    count = min(count, len(candidates))
    if count < 2:
        raise ValueError("Probe menu needs at least two candidates")
    by_key = {action_key(a): (a, indices) for a, indices in candidates}
    chosen = [by_key[action_key(coverage_action(browser))]]
    rng = random.Random(seed)
    shuffled = sorted(candidates, key=lambda x: action_key(x[0]))
    rng.shuffle(shuffled)
    families = ("coarse", "local", "left", "right")
    for family in families:
        for item in shuffled:
            a = item[0]
            if a.get("scale", a.get("direction")) == family and item not in chosen:
                chosen.append(item)
                break
        if len(chosen) >= count:
            break
    while len(chosen) < count:
        remaining = [x for x in shuffled if x not in chosen]
        # Different returned sets, not alternative JSON spellings of one choice.
        item = max(
            remaining,
            key=lambda x: min(len(set(x[1]) ^ set(y[1])) for y in chosen),
        )
        chosen.append(item)
    # Menu ordering must not disclose which proposal was the coverage baseline.
    rng.shuffle(chosen)
    return [a for a, _ in chosen]


def context_observation(browser, actions, mode):
    if mode not in CONTEXTS:
        raise ValueError("Unknown probe context")
    browser.history_mode = "current_frame" if mode == "current_frame" else "full"
    obs = replace(browser.observation(), legal_actions=tuple(actions))
    if mode in ("full", "current_frame"):
        return obs
    images = list(obs.images)
    old = [i for i, index in enumerate(obs.image_indices) if index not in obs.returned_indices]
    if mode == "shuffled_history":
        # Same earlier observed pixels and count, reversed temporal assignments.
        for i, source in zip(old, reversed(old), strict=True):
            images[i] = obs.images[source]
    else:
        positions = range(len(images)) if mode == "no_visual" else old
        for i in positions:
            images[i] = Image.new("RGB", images[i].size, (127, 127, 127))
    # Counterfactual pixels never replace clean evidence used by the reader.
    return replace(obs, images=tuple(images), history_mode=mode)


def collect(reader, cases_by_role, root, config, output, seed=7, menu_size=8):
    if config.environment.initial_images != 4 or config.environment.max_unique_frames != 12:
        raise ValueError("This probe is prespecified for four initial and twelve final frames")
    all_cases = [c for cases in cases_by_role.values() for c in cases]
    if any(c.split != "train" for c in all_cases):
        raise ValueError("Train-only probe; refusing image access for any other split")
    groups = [{c.group_id for c in cases} for cases in cases_by_role.values()]
    if len(groups) != 2 or groups[0] & groups[1]:
        raise ValueError("Fit and validation must be disjoint connected training groups")
    rows = []
    for role, cases in cases_by_role.items():
        for case in cases:
            prefix = []
            for depth in (0, 1):
                state_id = digest([PROBE_VERSION, case.video_id, seed, prefix, menu_size])
                path = output / "states" / f"{state_id}.json"
                b = replay_state(case, root, config, seed, 12, prefix)
                coverage = coverage_action(b)
                menu = candidate_menu(b, menu_size, seed + depth)
                expected = {
                    "state_id": state_id,
                    "video_id": case.video_id,
                    "group_id": case.group_id,
                    "split": case.split,
                    "role": role,
                    "label": case.label,
                    "seed": seed,
                    "budget": 12,
                    "prefix": list(prefix),
                    "observed_indices": list(b.observation().indices),
                    "returned_indices": list(b.observation().returned_indices),
                    "actions": menu,
                    "coverage_index": menu.index(coverage),
                }
                if path.exists():
                    row = read_json(path)
                    if any(row[k] != v for k, v in expected.items()):
                        raise ValueError("Saved counterfactual state differs from this run")
                else:
                    initial = reader.score(b.observation(reader=True))
                    outcomes = []
                    for action in menu:
                        branch = replay_state(case, root, config, seed, 12, prefix)
                        branch.step(action)
                        completion = []
                        while not branch.done:
                            follow = coverage_action(branch)
                            completion.append(follow)
                            branch.step(follow)
                        result = reader.score(branch.observation(reader=True))
                        if len(branch.env.visited) != 12:
                            raise RuntimeError("Counterfactual final exposure differs")
                        outcomes.append(
                            {
                                "action": action,
                                "completion": completion,
                                "selected_indices": list(branch.observation().indices),
                                "reader": result,
                                "log_score": log_score(result, case.label),
                                "correct": result["prediction"] == case.label,
                            }
                        )
                    row = {**expected, "initial_reader": initial, "outcomes": outcomes}
                    write_json(path, row)
                rows.append(row)
                print(f"probe collected {role} state {len(rows)} ({depth=})", flush=True)
                prefix.append(coverage)
    write_json(output / "states.json", rows)
    return rows


def summarize_signal(rows, minimum_gap=0.1):
    result = {}
    for role in ("fit", "validation"):
        subset = [r for r in rows if r["role"] == role]
        gaps = [
            max(x["log_score"] for x in r["outcomes"]) - min(x["log_score"] for x in r["outcomes"])
            for r in subset
        ]
        result[role] = {
            "cases": len({r["video_id"] for r in subset}),
            "states": len(subset),
            "states_with_preference_gap": sum(g >= minimum_gap for g in gaps),
            "mean_best_minus_worst_log_score": mean(gaps),
            "mean_best_minus_uniform_menu_log_score": mean(
                max(x["log_score"] for x in r["outcomes"])
                - mean(x["log_score"] for x in r["outcomes"])
                for r in subset
            ),
            "mean_best_minus_coverage_log_score": mean(
                max(x["log_score"] for x in r["outcomes"])
                - r["outcomes"][r["coverage_index"]]["log_score"]
                for r in subset
            ),
            "states_with_correctness_difference": sum(
                len({x["correct"] for x in r["outcomes"]}) > 1 for r in subset
            ),
            "mean_distinct_final_sets": mean(
                len({tuple(x["selected_indices"]) for x in r["outcomes"]}) for r in subset
            ),
        }
    result["minimum_pair_gap"] = minimum_gap
    result["fit_gate_passed"] = result["fit"]["states_with_preference_gap"] >= 4
    result["interpretation"] = (
        "Label-informed sampled-menu opportunity, not a deployable oracle or global performance bound. Gate uses fit states only."
    )
    result["test_opened"] = False
    return result
