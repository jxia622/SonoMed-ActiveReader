"""Deterministic recorded-video browser with private scoring and bounded memory."""

import random
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from sonomed_vlm.active_reader.config import Environment, Reward
from sonomed_vlm.active_reader.data.manifest import source_path
from sonomed_vlm.active_reader.env.actions import Inspect, Stop, parse_action, window_indices
from sonomed_vlm.active_reader.env.observation import Observation
from sonomed_vlm.active_reader.env.rewards import relevance
from sonomed_vlm.active_reader.io import file_hash


class Replay:
    def __init__(self, case, root: Path, config=None, reward=None, weights=None):
        case.validate()
        self._case, self._root = case, root
        self.config, self.reward = config or Environment(), reward or Reward()
        self._weights = weights or {"benign": 1.0, "malignant": 1.0}
        self._q, self._relevance_status = relevance(case)
        self.done = True

    def reset(self, seed=42, budget=None, start="random", initial_indices=None):
        requested_budget = self.config.max_unique_frames if budget is None else budget
        if requested_budget < 1:
            raise ValueError("Frame budget must be positive")
        self.budget = min(self._case.num_frames, requested_budget)
        self.turns = self.calls = self.invalid = self.revisits = 0
        self.done, self.prediction, self.evidence = False, None, []
        self.history, self.visited, self._cache = [], set(), {}
        count = min(self.config.initial_images, self.budget)
        if initial_indices is None:
            center = {
                "first": 0,
                "center": self._case.num_frames // 2,
                "random": random.Random(seed).randrange(self._case.num_frames),
            }.get(start)
            if center is None:
                raise ValueError("Unknown initial observation mode")
            begin = min(max(0, center - (count - 1) // 2), self._case.num_frames - count)
            indices = list(range(begin, begin + count))
        else:
            indices = sorted(set(initial_indices))
            if (
                not indices
                or len(indices) > self.budget
                or any(type(x) is not int or not 0 <= x < self._case.num_frames for x in indices)
            ):
                raise ValueError("Invalid initial frame selection")
        self._reveal(indices)
        self._phi = self._potential()
        self.components = {
            "classification": 0.0,
            "relevance": 0.0,
            "frames": -self.reward.unique_frame_cost_weight * len(indices) / self.budget,
            "calls": 0.0,
            "invalid": 0.0,
        }
        self.initial_indices = indices
        return self._observation(indices)

    def _reveal(self, indices):
        for index in indices:
            if index not in self._cache:
                try:
                    frame = self._case.frames[index]
                    path = source_path(self._root, frame.relative_path)
                    if frame.sha256 and file_hash(path) != frame.sha256:
                        raise ValueError("Source pixels changed since the audit")
                    with Image.open(path) as source:
                        # Fresh image strips filenames, EXIF, and other private metadata.
                        rgb = source.convert("RGB")
                        self._cache[index] = Image.frombytes("RGB", rgb.size, rgb.tobytes())
                except (OSError, ValueError):
                    raise RuntimeError(
                        "Unable to decode requested frame; private dataset audit required"
                    ) from None
        self.visited.update(indices)

    def _potential(self):
        return max((self._q[i] for i in self.visited), default=0.0) if self._q is not None else 0.0

    def legal_inspections(self):
        if self.done or self.turns >= self.config.max_decision_turns - 1:
            return ()
        return tuple(
            (b, s)
            for b in range(16)
            for s in ("coarse", "local")
            if len(self.visited | set(window_indices(self._case.num_frames, b, s))) <= self.budget
        )

    def _observation(self, returned=(), error=None):
        indices = tuple(sorted(self.visited))
        return Observation(
            indices,
            tuple(self._cache[i].copy() for i in indices),
            tuple(returned),
            self._case.num_frames,
            self.budget - len(indices),
            max(0, self.config.max_decision_turns - self.turns),
            self.legal_inspections(),
            tuple(self.history),
            error,
            self.config.max_evidence_frames,
        )

    def _finish(self):
        self.done = True
        weight = self._weights[self._case.label]
        self.components["classification"] = (
            weight if self.prediction == self._case.label else -weight
        )

    def step(self, raw):
        if self.done:
            raise RuntimeError("Episode has ended; reset before another action")
        before = sum(self.components.values())
        legal = self.legal_inspections()
        self.turns += 1
        returned, error, action = [], None, None
        # Every syntactically identifiable inspection request is charged, even if illegal.
        try:
            import json

            candidate = (
                json.loads(raw)
                if isinstance(raw, str)
                else asdict(raw)
                if isinstance(raw, (Inspect, Stop))
                else raw
            )
        except (ValueError, TypeError):
            candidate = None
        if isinstance(candidate, dict) and candidate.get("action") == "inspect":
            self.calls += 1
            self.components["calls"] -= self.reward.inspection_call_cost
        try:
            action = parse_action(raw, self.config.max_evidence_frames)
            if isinstance(action, Stop):
                if not set(action.evidence_frames) <= self.visited:
                    raise ValueError("Invalid action")
                self.prediction, self.evidence = action.prediction, list(action.evidence_frames)
                self._finish()
            else:
                if (action.center_bin, action.scale) not in legal:
                    raise ValueError("Invalid action")
                returned = window_indices(self._case.num_frames, action.center_bin, action.scale)
                new = len(set(returned) - self.visited)
                self.revisits += len(returned) - new
                self._reveal(returned)
                phi = self._potential()
                self.components["relevance"] += self.reward.relevance_weight * max(
                    0, phi - self._phi
                )
                self._phi = phi
                self.components["frames"] -= (
                    self.reward.unique_frame_cost_weight * new / self.budget
                )
        except ValueError:
            self.invalid += 1
            self.components["invalid"] -= self.reward.invalid_action_cost
            error = "Invalid action: follow the schema, legal inspections, and observed evidence indices."
        self.history.append(
            {
                "action": asdict(action) if action is not None and error is None else None,
                "valid": error is None,
                "returned_indices": returned,
            }
        )
        if not self.done and self.turns >= self.config.max_decision_turns:
            self._finish()
        return (
            self._observation(returned, error),
            sum(self.components.values()) - before,
            self.done,
            self.private_info(),
        )

    def private_info(self):
        values = [self._q[i] for i in self.visited] if self._q is not None else []
        return {
            "video_id": self._case.video_id,
            "group_id": self._case.group_id,
            "split": self._case.split,
            "label": self._case.label,
            "prediction": self.prediction,
            "evidence_frames": self.evidence,
            "invalid_terminal": self.done and self.prediction is None,
            "unique_frames": len(self.visited),
            "fraction_viewed": len(self.visited) / self._case.num_frames,
            "selected_indices": sorted(self.visited),
            "initial_indices": self.initial_indices,
            "inspection_calls": self.calls,
            "decision_turns": self.turns,
            "invalid_actions": self.invalid,
            "revisits": self.revisits,
            "reward_components": dict(self.components),
            "return": sum(self.components.values()),
            "relevance_status": self._relevance_status,
            "max_relevance": max(values) if values else None,
            "mean_relevance": sum(values) / len(values) if values else None,
        }
