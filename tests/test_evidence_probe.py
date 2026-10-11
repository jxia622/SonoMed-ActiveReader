from copy import deepcopy

import pytest
from PIL import Image

from sonomed_vlm.active_reader.agents.fixed_sampling import select_frames
from sonomed_vlm.active_reader.config import Config
from sonomed_vlm.active_reader.data.manifest import Case, Frame
from sonomed_vlm.active_reader.navigation.browser import NavigationBrowser, coverage_action
from sonomed_vlm.active_reader.navigation.probe import (
    candidate_menu,
    context_observation,
    partition_cases,
    replay_state,
)


@pytest.fixture
def nav_case(tmp_path):
    frames = []
    for i in range(40):
        Image.new("RGB", (8, 8), (i * 6, 80, 10)).save(tmp_path / f"{i}.png")
        frames.append(Frame(i, str(i), f"{i}.png", 8, 8, "present", [[0, 0, 1 + i / 10, 5]]))
    return Case(
        "private", "malignant/private", "malignant", frames, group_id="group", split="train"
    )


@pytest.mark.parametrize("budget", [6, 8, 12, 24])
@pytest.mark.parametrize("initial", [[58, 59, 60, 61], [0, 1, 2, 3], [116, 117, 118, 119]])
def test_uniform_covers_both_ends_despite_initial_views(budget, initial):
    selected = select_frames("uniform", 120, budget, initial)
    assert len(selected) == budget
    assert set(initial) <= set(selected)
    assert selected[0] == 0 and selected[-1] == 119


def test_baseline_exact_budget_with_grid_collisions():
    for n in (4, 5, 16, 40):
        for budget in range(4, n + 1):
            for mode in ("uniform", "coverage", "random"):
                selected = select_frames(mode, n, budget, [0, 1, 2, 3])
                assert len(selected) == budget and len(set(selected)) == budget
                assert all(0 <= i < n for i in selected)


def test_probe_partition_is_group_disjoint_train_only(nav_case):
    cases = []
    for i in range(10):
        c = deepcopy(nav_case)
        c.video_id = str(i)
        c.group_id = str(i // 2)
        c.label = "benign" if i < 4 else "malignant"
        cases.append(c)
    subsets = partition_cases(cases, count=2)
    assert {c.group_id for c in subsets["fit"]}.isdisjoint(
        c.group_id for c in subsets["validation"]
    )
    for subset in subsets.values():
        assert {c.label for c in subset} == {"benign", "malignant"}
    cases[0].split = "dev"
    with pytest.raises(ValueError, match="train cases only"):
        partition_cases(cases, count=2)


def test_probe_rejects_test_before_image_access(nav_case, tmp_path, monkeypatch):
    from PIL import Image

    nav_case.split = "test"
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("Opened test pixels"))
    with pytest.raises(ValueError, match="train split"):
        replay_state(nav_case, tmp_path, Config(), 7, 12, [])


def test_candidate_and_completion_are_annotation_independent(nav_case, tmp_path):
    config = Config()
    config.environment.max_unique_frames = 12
    original = NavigationBrowser(nav_case, tmp_path, config)
    original.reset(7, 12)
    changed = deepcopy(nav_case)
    changed.label = "benign"
    for f in changed.frames:
        f.boxes_xyxy = []
        f.annotation_status = "unannotated"
    alternate = NavigationBrowser(changed, tmp_path, config)
    alternate.reset(7, 12)
    menu = candidate_menu(original)
    assert menu == candidate_menu(alternate)
    assert coverage_action(original) == coverage_action(alternate)
    assert len(menu) == 8
    for action in menu:
        branch = replay_state(nav_case, tmp_path, config, 7, 12, [])
        branch.step(action)
        assert len(branch.env.visited) == 8
        branch.step(coverage_action(branch))
        assert branch.done and len(branch.env.visited) == 12


def test_history_counterfactual_preserves_bookkeeping_and_current_pixels(nav_case, tmp_path):
    b = NavigationBrowser(nav_case, tmp_path, Config())
    b.reset(7, 12)
    b.step(coverage_action(b))
    menu = candidate_menu(b)
    full = context_observation(b, menu, "full")
    blank = context_observation(b, menu, "blank_history")
    shuffled = context_observation(b, menu, "shuffled_history")
    current = context_observation(b, menu, "current_frame")
    assert full.indices == blank.indices == shuffled.indices == current.indices
    assert (
        full.legal_actions == blank.legal_actions == shuffled.legal_actions == current.legal_actions
    )
    assert full.history == blank.history == current.history
    old = [i for i, idx in enumerate(full.image_indices) if idx not in full.returned_indices]
    assert old
    for i, idx in enumerate(full.image_indices):
        if idx in full.returned_indices:
            assert (
                full.images[i].tobytes()
                == blank.images[i].tobytes()
                == shuffled.images[i].tobytes()
            )
        else:
            assert full.images[i].tobytes() != blank.images[i].tobytes()
    assert sorted(full.images[i].tobytes() for i in old) == sorted(
        shuffled.images[i].tobytes() for i in old
    )
    clean = b.observation(reader=True)
    assert [im.tobytes() for im in clean.images] == [im.tobytes() for im in full.images]
