import json
from copy import deepcopy
from dataclasses import replace

import pytest
from PIL import Image

from sonomed_vlm.active_reader.config import Config
from sonomed_vlm.active_reader.data.manifest import Case, Frame
from sonomed_vlm.active_reader.models.constraints import ActionGrammar
from sonomed_vlm.active_reader.navigation.browser import NavigationBrowser
from sonomed_vlm.active_reader.navigation.evaluation import paired, run_evaluation
from sonomed_vlm.active_reader.navigation.prompts import messages


@pytest.fixture
def nav_case(tmp_path):
    frames = []
    for i in range(40):
        Image.new("RGB", (8, 8), (i * 6, 80, 10)).save(tmp_path / f"{i}.png")
        frames.append(Frame(i, str(i), f"{i}.png", 8, 8, "present", [[0, 0, 1 + i / 10, 5]]))
    return Case(
        "private", "malignant/private", "malignant", frames, group_id="group", split="train"
    )


@pytest.mark.parametrize("budget", [5, 8, 11, 12, 24])
def test_navigation_exact_budget_refinement_and_no_stop(nav_case, tmp_path, budget):
    browser = NavigationBrowser(nav_case, tmp_path, Config())
    observation = browser.reset(42, budget, initial=[13, 14, 15, 16])
    assert {
        "action": "refine",
        "anchor_frame": 16,
        "direction": "right",
    } in observation.legal_actions
    observation = browser.step({"action": "refine", "anchor_frame": 16, "direction": "right"})
    assert observation.returned_indices == tuple(range(17, 17 + min(4, budget - 4)))
    while not browser.done:
        observation = browser.step(observation.legal_actions[-1])
    assert len(observation.indices) == budget
    assert sum(len(h["returned_indices"]) for h in observation.history) == budget - 4
    assert all(h["action"]["action"] in ("inspect", "refine") for h in observation.history)
    browser.reset(42, 12)
    targets = [tuple(indices) for action, indices in browser.proposals().values()]
    assert len(targets) == len(set(targets))
    with pytest.raises(ValueError):
        browser.step({"action": "stop", "prediction": "malignant", "evidence_frames": [13]})


def test_actual_visual_history_and_private_invariance(nav_case, tmp_path):
    altered = deepcopy(nav_case)
    altered.label = "benign"
    altered.source_relative_path = "benign/secret"
    altered.video_id = "another_private_id"
    for frame in altered.frames:
        frame.boxes_xyxy = []
        frame.annotation_status = "unannotated"
    browsers = [
        NavigationBrowser(c, tmp_path, Config(), mode)
        for c in (nav_case, altered)
        for mode in ("full", "current_batch", "current_frame")
    ]
    for b in browsers:
        b.reset(42, 12, initial=[13, 14, 15, 16])
        b.step({"action": "refine", "anchor_frame": 16, "direction": "right"})
    full, batch, current = [b.observation() for b in browsers[:3]]
    assert full.image_indices == tuple(range(13, 21))
    assert batch.image_indices == (17, 18, 19, 20)
    assert current.image_indices == (19,)
    assert full.indices == batch.indices == current.indices
    assert full.history == batch.history == current.history
    for a, b in zip(browsers[:3], browsers[3:], strict=True):
        assert a.observation().public_dict() == b.observation().public_dict()
        assert [im.tobytes() for im in a.observation().images] == [
            im.tobytes() for im in b.observation().images
        ]
    for b in browsers:
        text = json.dumps(
            [
                {**m, "content": [c for c in m["content"] if c["type"] != "image"]}
                for m in messages(b.observation())
            ]
        )
        assert "secret" not in text and "private" not in text
        assert "boxes" not in text and "reward" not in text


def test_reader_prompt_discards_selection_history_and_budget(nav_case, tmp_path):
    browser = NavigationBrowser(nav_case, tmp_path, Config())
    browser.reset(42, 12)
    obs = browser.observation(reader=True)
    other = replace(
        obs,
        indices=tuple(i + 1000 for i in obs.indices),
        visible_indices=tuple(i + 1000 for i in obs.image_indices),
        history=({"action": "privileged_area", "reward": 100},),
        remaining_frames=999,
        history_mode="current_frame",
    )
    assert messages(obs) == messages(other)
    assert messages(obs)[1]["content"][0]["text"] == "Frame 1"


def test_navigation_grammar_has_inspection_only(nav_case, tmp_path):
    b = NavigationBrowser(nav_case, tmp_path, Config())
    obs = b.reset(42, 12)

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            self.actions = [json.loads(t) for t in texts]
            return {"input_ids": [[i + 10] for i in range(len(texts))]}

    t = Tokenizer()
    g = ActionGrammar(obs, t, 999)
    assert t.actions == list(obs.legal_actions)
    assert all(a["action"] in ("inspect", "refine") for a in t.actions)
    assert g.constraints_for([10, 999]) == [list(range(10, 10 + len(t.actions))), [999]]


def test_navigation_test_split_always_sealed(nav_case, tmp_path):
    nav_case.split = "test"
    with pytest.raises(ValueError, match="test is sealed"):
        run_evaluation(None, None, [nav_case], tmp_path, Config(), tmp_path / "out", {})


def test_paired_comparison_requires_equal_exposure():
    row = {
        "video_id": "a",
        "group_id": "g",
        "seed": 7,
        "budget": 8,
        "method": "uniform",
        "label": "benign",
        "prediction": "benign",
        "initial_indices": [0, 1, 2, 3],
        "unique_frames": 8,
        "reader_true_label_log_score": -0.5,
    }
    other = {**row, "method": "full", "unique_frames": 7}
    with pytest.raises(ValueError, match="Unmatched"):
        paired([row, other], bootstrap=0)
