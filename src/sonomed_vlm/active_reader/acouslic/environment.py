"""Full-range inspection actions with exactly three new frames per action."""

import random

import numpy as np


def initial(seed):
    rng = random.Random(seed)
    return [140 * sweep + rng.randrange(35, 105) for sweep in range(6)]


def inspect(center, observed):
    if not isinstance(center, int) or not 0 <= center < 840 or center in observed:
        raise ValueError("Inspection center must be an unseen legal frame")
    sweep = center // 140
    candidates = [i for i in range(sweep * 140, (sweep + 1) * 140) if i not in observed]
    if len(candidates) < 3:
        raise ValueError("Fewer than three unseen frames remain in sweep")
    return sorted(sorted(candidates, key=lambda i: (abs(i - center), i))[:3])


def legal_centers(observed):
    return [i for i in range(840) if i not in observed]


def uniform(budget=12):
    assert budget % 6 == 0
    per_sweep = budget // 6
    return [
        140 * s + round((j + 1) * 139 / (per_sweep + 1)) for s in range(6) for j in range(per_sweep)
    ]


def coverage(seed):
    selected = initial(seed)
    while len(selected) < 12:
        selected.append(
            max(
                (i for i in range(840) if i not in selected),
                key=lambda i: (min(abs(i - j) for j in selected), -i),
            )
        )
    return sorted(selected)


def prior_action(train_quality, observed):
    """Train-only geometry baseline maximizing empirical acquisition utility."""
    old = train_quality[:, observed].max(axis=1)
    centers = legal_centers(observed)
    gains = [
        np.maximum(old, train_quality[:, inspect(i, observed)].max(axis=1)).mean() for i in centers
    ]
    return centers[int(np.argmax(gains))]


def outcome(quality, labels, indices):
    assert len(indices) == len(set(indices)) == 12
    return {
        "indices": sorted(indices),
        "best_wfss": float(quality[indices].max()),
        "any_supported": bool((quality[indices] > 0).any()),
        "any_optimal": bool((labels[indices] == 1).any()),
    }


def audit_reachability(quality, seed):
    observed = initial(seed)
    reachable = set(observed)
    for center in legal_centers(observed):
        returned = inspect(center, observed)
        assert len(set(returned) - set(observed)) == 3 and len({i // 140 for i in returned}) == 1
        reachable.update(returned)
    assert reachable == set(range(840))
    return {
        "reachable_frames": len(reachable),
        "initial_wfss": float(quality[observed].max()),
        "reachable_wfss": float(quality.max()),
    }
