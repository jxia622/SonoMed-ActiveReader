"""Predeclared acquisition baselines and frozen-reader outcome evaluation."""

import random
import string

import numpy as np

from sonomed_vlm.active_reader.acouslic.environment import (
    audit_reachability,
    coverage,
    initial,
    inspect,
    legal_centers,
    outcome,
    prior_action,
    uniform,
)
from sonomed_vlm.active_reader.acouslic.policy import Choices
from sonomed_vlm.active_reader.acouslic.training import episode
from sonomed_vlm.active_reader.io import write_json
from sonomed_vlm.active_reader.training.grpo import parameter_digest


def baselines(data, cases, protocol, output):
    train_quality = np.stack([data.quality(c["alias"])[0] for c in data.subset("train")])
    rows, reachability = [], []
    for case in cases:
        q, labels = data.quality(case["alias"])
        for seed in protocol["evaluation_starts"]:
            reachability.append(
                {"alias": case["alias"], "seed": seed, **audit_reachability(q, seed)}
            )
            sets = {"uniform": uniform(), "coverage": coverage(seed)}
            observed = initial(seed)
            for _ in range(2):
                observed.extend(inspect(prior_action(train_quality, observed), observed))
            sets["geometry_prior"] = sorted(observed)
            sets["privileged"] = sorted(sorted(range(840), key=lambda i: (-q[i], i))[:12])
            for method, indices in sets.items():
                rows.append(
                    {
                        "alias": case["alias"],
                        "split": case["split"],
                        "method": method,
                        "seed": seed,
                        "primary": True,
                        "reader_subset": True,
                        **outcome(q, labels, indices),
                    }
                )
        random_seeds = (
            list(range(10000, 10000 + protocol["random_repeats"])) + protocol["evaluation_starts"]
        )
        for seed in random_seeds:
            sets = {"random": sorted(random.Random(seed).sample(range(840), 12))}
            rng, observed = random.Random(seed), initial(seed)
            for _ in range(2):
                observed.extend(inspect(rng.choice(legal_centers(observed)), observed))
            sets["random_actions"] = sorted(observed)
            for method, indices in sets.items():
                rows.append(
                    {
                        "alias": case["alias"],
                        "split": case["split"],
                        "method": method,
                        "seed": seed,
                        "primary": seed >= 10000,
                        "reader_subset": seed in protocol["evaluation_starts"],
                        **outcome(q, labels, indices),
                    }
                )
        write_json(output / "acquisition.json", rows)
    write_json(output / "reachability.json", reachability)
    return rows


def policy_evaluation(policy, data, cases, protocol, output, mode, method):
    choices, rows = Choices(policy, data), []
    before = parameter_digest(policy.model, lambda n, p: ".policy." in n)
    for case in cases:
        for seed in protocol["evaluation_starts"]:
            row, _ = episode(choices, case, seed, mode)
            row.update(method=method, primary=True, reader_subset=True)
            rows.append(row)
        write_json(output / "acquisition.json", rows)
        print(f"Evaluated {method}: {case['alias']} ({case['split']})", flush=True)
    assert before == parameter_digest(policy.model, lambda n, p: ".policy." in n)
    return rows


def frozen_reader(policy, data, rows, output):
    choices, results, cache = Choices(policy, data), [], {}
    before = parameter_digest(policy.model, lambda n, p: True)
    for row in rows:
        if not row["reader_subset"] or row["method"] == "privileged":
            continue
        key = (row["alias"], tuple(row["indices"]))
        if key not in cache:
            generation = choices.recognize(row["alias"], row["indices"], "original_frozen_reader")
            selected = row["indices"][string.ascii_uppercase.index(generation.text)]
            q, _ = data.quality(row["alias"])
            cache[key] = {
                "selected_frame": selected,
                "reader_wfss": float(q[selected]),
                "input": generation.record,
            }
        result = {
            "alias": row["alias"],
            "split": row["split"],
            "seed": row["seed"],
            "method": row["method"],
            **cache[key],
        }
        results.append(result)
        if len(results) % 20 == 0:
            write_json(output / "reader_results.json", results)
            print(
                f"Frozen reader: {len(results)} method/start rows, {len(cache)} distinct inputs",
                flush=True,
            )
    assert before == parameter_digest(policy.model, lambda n, p: True)
    write_json(output / "reader_results.json", results)
    write_json(
        output / "report.json",
        {
            "rows": len(results),
            "distinct_inputs": len(cache),
            "frozen_unchanged": True,
            "parameter_sha256": before,
        },
    )
    return results
