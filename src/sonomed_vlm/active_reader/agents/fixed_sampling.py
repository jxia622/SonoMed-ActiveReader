"""Label-independent fixed selection; privileged selector is explicitly separate."""

import random

from sonomed_vlm.active_reader.env.actions import evenly_spaced

METHODS = {
    "initial",
    "first",
    "center",
    "first_frame",
    "center_frame",
    "uniform",
    "random",
    "coverage",
    "coarse_to_fine",
    "prompted",
    "dense",
    "privileged_area",
}


def select_frames(method, n, budget, initial, seed=42, case=None):
    if method not in METHODS:
        raise ValueError(f"Unknown baseline {method}")
    budget = min(n, budget)
    chosen = set(initial)
    if len(chosen) > budget:
        raise ValueError("Initial views exceed fixed budget")
    if method == "initial":
        return sorted(chosen)
    if method == "privileged_area":
        if case is None:
            raise ValueError("Privileged selector requires explicit evaluator-private case")

        def area(i):
            f = case.frames[i]
            return sum((b[2] - b[0]) * (b[3] - b[1]) for b in f.boxes_xyxy) / (f.width * f.height)

        candidates = sorted(range(n), key=lambda i: (-area(i), i))
    elif method == "first":
        candidates = list(range(n))
    elif method == "center":
        candidates = sorted(range(n), key=lambda i: (abs(i - (n - 1) / 2), i))
    elif method == "random":
        candidates = list(range(n))
        random.Random(seed).shuffle(candidates)
    elif method == "coverage":
        candidates = []  # Farthest-point completion below uses all initial views.
    else:
        # The initial observations already consume budget. Traversing a total-budget
        # grid until it fills would silently discard its later temporal positions.
        candidates = evenly_spaced(0, n - 1, budget - len(chosen))
    for index in candidates:
        if len(chosen) >= budget:
            break
        chosen.add(index)
    # Fill collisions with initial frames without scanning unused full-video subsets.
    while len(chosen) < budget:
        remaining = set(range(n)) - chosen
        chosen.add(max(remaining, key=lambda i: (min(abs(i - j) for j in chosen), -i)))
    return sorted(chosen)


def coarse_to_fine(observation):
    from sonomed_vlm.active_reader.env.actions import Inspect, window_indices

    candidates = observation.legal_inspections
    if not candidates or observation.remaining_turns <= 1:
        return None

    # Coverage heuristic, not claimed expert demonstrations.
    def score(action):
        indices = window_indices(observation.total_frames, *action)
        new = len(set(indices) - set(observation.indices))
        return new, int(action[1] == ("coarse" if len(observation.history) < 2 else "local"))

    best = max(candidates, key=score)
    return Inspect(*best) if score(best)[0] > 0 else None
