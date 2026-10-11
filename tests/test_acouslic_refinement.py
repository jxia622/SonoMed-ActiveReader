"""Causal contexts and conditional credit for the second development campaign."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sonomed_vlm.active_reader.acouslic.environment import initial, legal_centers


def module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "refinement", scripts / "train_acouslic_refinement.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_second_action_credit_excludes_reward_already_acquired():
    m = module()
    # Branch zero already earned .6, but neither second action adds evidence.
    # Branch one has a better and a worse continuation from the identical state.
    first, second, zero_first, zeros_second = m.conditional_credit(
        [[0.6, 0.6], [0, 1], [0, 0], [0, 0]], [[0, 0], [0, 1], [0, 0], [0, 0]]
    )
    assert not zero_first and first[0] > first[1] > first[2]
    assert second[0].tolist() == [0, 0]
    assert second[1][0] < 0 < second[1][1]
    assert zeros_second == [True, False, True, True]


def test_private_proximity_stays_within_sweep_and_rewards_telescope():
    m = module()
    quality = np.zeros(840)
    quality[139] = 1
    nearby = m.proximity(quality, 12)
    assert nearby[139] == 1 and 0 < nearby[137] < nearby[138] < 1
    assert not nearby[140:].any()
    assert nearby[136] > 0 and quality[136] == 0  # auxiliary is not an annotation
    a = initial(7)
    b = [*a, 135, 136, 137]
    c = [*b, 138, 139, 134]
    values = [m.utility(quality, nearby, ids, 0.2) for ids in [a, b, c]]
    assert (values[1] - values[0]) + (values[2] - values[1]) == pytest.approx(values[2] - values[0])
    assert m.utility(quality, nearby, b, 0) == quality[b].max()


def test_tree_continuations_share_state_without_observing_other_branches():
    m = module()
    calls = []
    quality = np.zeros(840)
    quality[100] = 1

    class Choices:
        data = SimpleNamespace(quality=lambda _: (quality, np.zeros(840)))

        def inspect(self, alias, observed, latest, mode, seed, sample, retain):
            assert sample and retain and alias == "training-case"
            calls.append((list(observed), list(latest), mode))
            center = legal_centers(observed)[seed % len(legal_centers(observed))]
            return SimpleNamespace(text=str(center), record={"center": center})

    row, terms = m.tree(
        Choices(), {"alias": "training-case", "split": "train"}, 7, "full", 101, 0.2, 12
    )
    assert len(terms) == len(calls) == 12
    assert sum(scale for _, _, scale in terms) == pytest.approx(1)
    for branch in range(4):
        first, second_a, second_b = calls[branch * 3 : branch * 3 + 3]
        assert first[0] == first[1] == initial(7)
        assert second_a == second_b
        assert len(second_a[0]) == 9 and len(second_a[1]) == 3
        for continuation in row["branches"][branch]["continuations"]:
            assert len(set(continuation["indices"])) == 12
    with pytest.raises(AssertionError):
        m.tree(Choices(), {"alias": "training-case", "split": "tune"}, 7, "full", 101, 0.2, 12)
