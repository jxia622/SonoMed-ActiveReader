#!/usr/bin/env python3
"""Exercise the real acquisition environment with synthetic scores; no RL or clinical data."""

import json
import random

import _bootstrap  # noqa: F401
import numpy as np

from sonomed_vlm.active_reader.acouslic.environment import (
    initial,
    inspect,
    legal_centers,
    outcome,
    uniform,
)


def main():
    rng = random.Random(7)
    observed = initial(7)
    starting = list(observed)
    trajectory = []
    # Scores exist only in the evaluator; this random policy never sees them.
    for turn in range(2):
        center = rng.choice(legal_centers(observed))
        fresh = inspect(center, observed)
        assert len(fresh) == 3 and not set(fresh).intersection(observed)
        observed.extend(fresh)
        trajectory.append({"turn": turn + 1, "center": center, "new_frames": fresh})
    quality = np.zeros(840)
    quality[52:57] = 1.0
    quality[465:470] = 0.6
    labels = np.where(quality == 1.0, 1, np.where(quality > 0, 2, 0))
    report = {
        "scope": "Synthetic software demonstration; not an RL or medical result",
        "policy": "seeded random legal actions; no training",
        "initial_indices": starting,
        "inspections": trajectory,
        "unique_frame_count": len(set(observed)),
        "random_action_outcome": outcome(quality, labels, observed),
        "uniform_outcome": outcome(quality, labels, uniform()),
        "budget_check": "passed",
    }
    assert report["unique_frame_count"] == 12
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
