"""Reuse audited source decoding; add exact budgets and observed-anchor refinement.

The existing joint inspect/stop Replay contract is preserved. Navigation changes
are explicit: new frames only, progress required, and no policy-selected stopping.
"""

import json

from sonomed_vlm.active_reader.env.actions import window_indices
from sonomed_vlm.active_reader.env.observation import Observation
from sonomed_vlm.active_reader.env.replay import Replay


def action_key(action):
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def coverage_action(browser):
    """Public-state-only baseline: reveal four frames, then prefer distant coverage."""
    seen = browser.env.visited

    def score(item):
        action, indices = item
        return (
            len(indices),
            sum(min(abs(i - j) for j in seen) for i in indices) / len(indices),
            max(indices) - min(indices),
            action_key(action),
        )

    return max(browser.proposals().values(), key=score)[0]


class NavigationBrowser:
    def __init__(self, case, root, config, history_mode="full"):
        if history_mode not in ("full", "current_batch", "current_frame"):
            raise ValueError("Unknown visual history mode")
        self.env = Replay(case, root, config.environment, config.reward)
        self.history_mode = history_mode

    def reset(self, seed, budget, initial=None):
        obs = self.env.reset(seed, budget, initial_indices=initial)
        self.returned = list(obs.indices)
        self.done = len(self.env.visited) == self.env.budget
        return self.observation()

    def proposals(self):
        e = self.env
        remaining = e.budget - len(e.visited)
        if remaining <= 0:
            return {}
        result = {}

        def add(action, candidates):
            new = [i for i in candidates if i not in e.visited][: min(4, remaining)]
            if new:
                result[action_key(action)] = (action, sorted(new))

        for center in range(16):
            for scale in ("coarse", "local"):
                add(
                    {"action": "inspect", "center_bin": center, "scale": scale},
                    window_indices(e._case.num_frames, center, scale),
                )
        for anchor in sorted(e.visited):
            for direction, sign in (("left", -1), ("right", 1)):
                candidates = [
                    anchor + sign * distance
                    for distance in range(1, 9)
                    if 0 <= anchor + sign * distance < e._case.num_frames
                ]
                add(
                    {"action": "refine", "anchor_frame": anchor, "direction": direction}, candidates
                )
        if not result:
            raise RuntimeError("Fixed-budget browser cannot make progress")

        # One action per returned frame set: duplicate JSON aliases must not multiply
        # the probability mass of a location. Prefer the closest observed anchor.
        def priority(item):
            action, indices = item
            if action["action"] == "refine":
                return (
                    0,
                    min(abs(action["anchor_frame"] - i) for i in indices),
                    action_key(action),
                )
            return (1, 0, action_key(action))

        unique = {}
        for action, indices in sorted(result.values(), key=priority):
            unique.setdefault(tuple(indices), (action, indices))
        return {action_key(a): (a, indices) for a, indices in unique.values()}

    def observation(self, reader=False):
        e = self.env
        observed = tuple(sorted(e.visited))
        visible = observed
        if not reader and self.history_mode == "current_batch":
            visible = tuple(self.returned)
        elif not reader and self.history_mode == "current_frame":
            visible = (self.returned[len(self.returned) // 2],)
        return Observation(
            indices=observed,
            images=tuple(e._cache[i].copy() for i in visible),
            returned_indices=tuple(self.returned),
            total_frames=e._case.num_frames,
            remaining_frames=e.budget - len(observed),
            remaining_turns=e.budget - len(observed),
            history=tuple(e.history),
            protocol="reader" if reader else "navigation",
            visible_indices=visible,
            history_mode=self.history_mode,
            legal_actions=() if reader else tuple(a for a, _ in self.proposals().values()),
        )

    def step(self, raw):
        if self.done:
            raise RuntimeError("Navigation already reached its fixed budget")
        try:
            action = json.loads(raw) if isinstance(raw, str) else raw
            action, returned = self.proposals()[action_key(action)]
        except (ValueError, TypeError, KeyError):
            # Fail closed: no privileged fallback and no under-budget scored episode.
            raise ValueError("Selector must return exactly one legal inspection action") from None
        self.env._reveal(returned)
        self.env.turns += 1
        self.env.calls += 1
        self.env.history.append({"action": action, "valid": True, "returned_indices": returned})
        self.returned = returned
        self.done = len(self.env.visited) == self.env.budget
        return self.observation()
