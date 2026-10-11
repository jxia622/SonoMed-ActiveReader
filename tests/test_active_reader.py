from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

import pytest
from PIL import Image

from sonomed_vlm.active_reader.agents.fixed_sampling import select_frames
from sonomed_vlm.active_reader.config import Config, Environment, load_config
from sonomed_vlm.active_reader.data.corrections import SHARED, apply_corrections
from sonomed_vlm.active_reader.data.manifest import (
    Case,
    Frame,
    resize_box,
    source_path,
    xywh_to_xyxy,
)
from sonomed_vlm.active_reader.data.source_loader import load_source
from sonomed_vlm.active_reader.data.splits import attach_splits, connect_groups, make_splits
from sonomed_vlm.active_reader.env.actions import Inspect, Stop, parse_action, window_indices
from sonomed_vlm.active_reader.env.replay import Replay
from sonomed_vlm.active_reader.env.rewards import class_weights, relevance
from sonomed_vlm.active_reader.evaluation.baselines import dense_read, fixed_read
from sonomed_vlm.active_reader.evaluation.metrics import evaluate, summarize
from sonomed_vlm.active_reader.models.prompts import build_messages
from sonomed_vlm.active_reader.models.qwen3_adapter import Generation, resolve_assets
from sonomed_vlm.active_reader.training.rollouts import grouped_rollouts, rollout


@pytest.fixture
def case(tmp_path):
    frames = []
    for i in range(60):
        Image.new("RGB", (32, 24), (i * 4, 100, 90)).save(tmp_path / f"{i}.png")
        frames.append(
            Frame(
                i,
                str(i),
                f"{i}.png",
                32,
                24,
                "present",
                [[0, 0, 1 + i / 2, 20]],
                pixel_sha256=f"pixel-{i}",
            )
        )
    return Case("PRIVATE_CASE", "benign/PRIVATE_PATH", "benign", frames, group_id="PRIVATE_GROUP")


class MockPolicy:
    def __init__(self, inspect=False):
        self.observations = []
        self.inspect = inspect

    def generate(self, observation, **kwargs):
        self.observations.append(observation)
        if (
            self.inspect
            and observation.legal_inspections
            and not observation.history
            and not kwargs.get("final_only")
        ):
            b, s = observation.legal_inspections[-1]
            text = json.dumps(asdict(Inspect(b, s)))
        else:
            text = json.dumps(asdict(Stop("benign", (observation.indices[0],))))
        return Generation(
            text,
            {
                "raw_output": text,
                "image_presentations": len(observation.images),
                "generated_tokens": 12,
                "latency_seconds": 0.0,
            },
        )


def test_box_geometry():
    assert xywh_to_xyxy([2, 3, 4, 5], 20, 30) == [2, 3, 6, 8]
    assert resize_box([2, 3, 6, 8], (20, 30), (40, 90)) == [4, 9, 12, 24]
    for box in ([0, 0, 0, 1], [0, 0, 21, 1], [0, 0, float("nan"), 1], [-1, 0, 3, 3]):
        with pytest.raises(ValueError):
            xywh_to_xyxy(box, 20, 20)


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        "```json\n{}\n```",
        '{"action":"inspect","center_bin":true,"scale":"local"}',
        '{"action":"inspect","center_bin":16,"scale":"local"}',
        '{"action":"stop","prediction":"benign","evidence_frames":[]}',
        '{"action":"stop","prediction":"benign","evidence_frames":[1,1]}',
    ],
)
def test_strict_actions(raw):
    with pytest.raises(ValueError):
        parse_action(raw)


def test_all_windows():
    assert window_indices(100, 8, "coarse") == [44, 50, 57, 63]
    for n in range(1, 150):
        for b in range(16):
            for scale in ("local", "coarse"):
                indices = window_indices(n, b, scale)
                assert indices == sorted(set(indices))
                assert len(indices) == min(n, 4)
                assert all(0 <= i < n for i in indices)


def test_corrections_idempotent_and_source_preserved(case):
    cases = []
    for name in SHARED | {"x66ef02e7f1b9a0ef", "x3b88488853e8b7d1"}:
        c = deepcopy(case)
        c.video_id, c.source_relative_path = name, "benign/" + name
        cases.append(c)
    original = deepcopy(cases)
    corrected, _ = apply_corrections(cases)
    twice, _ = apply_corrections(corrected)
    assert corrected == twice
    assert cases == original
    assert len(corrected) == 5 and all(c.label == "malignant" for c in corrected)
    assert len({c.group_id for c in corrected if c.video_id in SHARED}) == 1


def test_annotation_loader_natural_order_and_unknown(tmp_path):
    raw = tmp_path / "rawframes" / "benign" / "sequence"
    raw.mkdir(parents=True)
    images = []
    for i in (1, 10, 2):
        Image.new("RGB", (20, 10), (i, 0, 0)).save(raw / f"{i}.png")
        images.append({"id": i, "file_name": f"benign/sequence/{i}.png", "width": 20, "height": 10})
    images[-1]["annotation_status"] = "empty"
    document = {
        "images": images,
        "annotations": [{"image_id": 1, "bbox": [1, 2, 4, 5], "category_id": 1}],
        "categories": [{"id": 1, "name": "benign"}],
    }
    (tmp_path / "imagenet_vid_train_a.json").write_text(json.dumps(document))
    cases, metadata = load_source(tmp_path)
    assert not metadata["issues"]
    frames = cases[0].frames
    assert [f.source_frame_id for f in frames] == ["1", "2", "10"]
    assert [f.annotation_status for f in frames] == ["present", "empty", "unannotated"]
    assert frames[0].boxes_xyxy == [[1, 2, 5, 7]]
    with pytest.raises(ValueError):
        source_path(tmp_path, "../escape.png")


def test_group_splits_are_deterministic_and_disjoint(case):
    cases = []
    for i in range(40):
        c = deepcopy(case)
        c.video_id = c.group_id = f"case-{i}"
        c.label = "benign" if i % 2 else "malignant"
        c.frames = [deepcopy(c.frames[0])]
        c.frames[0].pixel_sha256 = f"pixel-{i}"
        cases.append(c)
    cases[1].frames[0].pixel_sha256 = cases[0].frames[0].pixel_sha256
    cases[2].patient_id = cases[3].patient_id = "patient-a"
    connect_groups(cases)
    split = make_splits(cases, 77)
    assert split == make_splits(cases, 77)
    result = attach_splits(deepcopy(cases), split)
    assert result[0].split == result[1].split
    assert result[2].split == result[3].split
    assert all(set(v) == {"benign", "malignant"} for v in split["counts"].values())
    cases[0].label = "malignant" if cases[0].label == "benign" else "benign"
    with pytest.raises(ValueError):
        attach_splits(cases, split)


def test_hidden_annotations_do_not_change_policy(case, tmp_path):
    other = deepcopy(case)
    other.label, other.video_id, other.group_id = "malignant", "SECRET", "SECRET"
    for frame in other.frames:
        frame.boxes_xyxy = [[0, 0, 2, 2]]
    left, right = Replay(case, tmp_path), Replay(other, tmp_path)
    lo, ro = left.reset(4), right.reset(4)
    for _ in range(3):
        assert lo.public_dict() == ro.public_dict()
        assert [i.tobytes() for i in lo.images] == [i.tobytes() for i in ro.images]
        assert all(not i.info and not getattr(i, "filename", None) for i in lo.images)
        serialized = json.dumps(build_messages(lo), default=lambda x: "IMAGE")
        assert not any(
            secret in serialized
            for secret in ("PRIVATE", "SECRET", "boxes_xyxy", "reward_components")
        )
        b, s = lo.legal_inspections[-1]
        lo, _, _, _ = left.step(Inspect(b, s))
        ro, _, _, _ = right.step(Inspect(b, s))
    _, lr, _, _ = left.step(Stop("benign", (lo.indices[0],)))
    _, rr, _, _ = right.step(Stop("benign", (ro.indices[0],)))
    assert lr != rr


def test_budget_initial_revisit_and_relevance(case, tmp_path):
    env = Replay(case, tmp_path)
    observation = env.reset(2, 12, start="first")
    assert observation.indices == (0, 1, 2, 3)
    assert env.private_info()["reward_components"]["relevance"] == 0
    assert env.private_info()["reward_components"]["frames"] == pytest.approx(-0.2 * 4 / 12)
    observation, _, _, _ = env.step(Inspect(15, "local"))
    before = env.private_info()
    observation, _, _, _ = env.step(Inspect(15, "local"))
    after = env.private_info()
    assert after["unique_frames"] == before["unique_frames"]
    assert after["reward_components"]["relevance"] == before["reward_components"]["relevance"]
    assert after["inspection_calls"] == before["inspection_calls"] + 1
    assert after["return"] == pytest.approx(before["return"] - 0.01)
    assert 0 <= after["reward_components"]["relevance"] <= 0.1
    for b, s in observation.legal_inspections:
        assert len(set(window_indices(60, b, s)) | set(observation.indices)) <= 12


def test_invalid_and_last_turn_terminal(case, tmp_path):
    env = Replay(case, tmp_path, Environment(max_decision_turns=2))
    initial = env.reset(1, 4)
    observation, _, done, _ = env.step(Inspect(15, "local"))
    assert not done and observation.error
    assert observation.indices == initial.indices
    assert observation.legal_inspections == ()
    _, _, done, info = env.step(Stop("benign", (59,)))
    assert done and info["invalid_terminal"]
    assert info["invalid_actions"] == 2
    assert info["reward_components"]["classification"] == -1
    with pytest.raises(RuntimeError):
        env.step("{}")


def test_unknown_and_multilesion_disable_proxy(case):
    assert relevance(case)[0] is not None
    case.frames[0].boxes_xyxy = []
    case.frames[0].annotation_status = "unannotated"
    assert relevance(case)[0] is None
    case.frames[0].annotation_status = "empty"
    assert relevance(case)[0] is not None
    case.frames[1].boxes_xyxy.append([1, 1, 3, 3])
    assert relevance(case)[0] is None


def test_training_only_class_weights(case):
    other = deepcopy(case)
    other.label = "malignant"
    with pytest.raises(ValueError):
        class_weights([case, other], "inverse_frequency_reward")
    case.split = other.split = "train"
    assert class_weights([case, case, case, other], "inverse_frequency_reward") == {
        "benign": 2 / 3,
        "malignant": 2,
    }
    assert class_weights([case, other], "unweighted") == {"benign": 1, "malignant": 1}


def test_matched_fixed_baselines_and_dense_cost(case, tmp_path):
    config, policy = Config(), MockPolicy()
    initial = list(Replay(case, tmp_path).reset(19, 12).indices)
    for method in ("first", "center", "uniform", "random"):
        chosen = select_frames(method, 60, 12, initial, 19)
        assert len(chosen) == 12 and set(initial) <= set(chosen)
        row = fixed_read(case, tmp_path, policy, config, method, 19, 12)
        assert row["initial_indices"] == initial and row["unique_frames"] == 12
    row = dense_read(case, tmp_path, policy, config, 19, 12)
    assert row["unique_frames"] == row["image_presentations"] == 60
    assert row["model_calls"] == 5 and row["prediction"] == "benign"


def test_fixed_reader_selection_cannot_expand_budget(case, tmp_path):
    policy = MockPolicy()
    with pytest.raises(ValueError, match="exceeds its frame budget"):
        fixed_read(
            case, tmp_path, policy, Config(), "fixed_reader:dense", 19, 12, selected=list(range(60))
        )
    assert policy.observations == []


def test_grouped_rollout_identical_start_and_private_boundary(case, tmp_path):
    policy = MockPolicy(inspect=True)
    group = grouped_rollouts(lambda: Replay(case, tmp_path), policy, 7, 43, 12, 4)
    assert len({tuple(t.record["initial_indices"]) for t in group}) == 1
    assert all(t.record["decision_turns"] >= 1 for t in group)
    assert all(not hasattr(o, "label") and not hasattr(o, "reward") for o in policy.observations)


def test_metrics_invalid_not_dropped(case, tmp_path):
    row = rollout(Replay(case, tmp_path), MockPolicy(), 1, 4).record
    invalid = {
        **row,
        "label": "malignant",
        "prediction": None,
        "group_id": "other",
        "video_id": "other",
    }
    result = summarize([row, invalid])
    assert result["accuracy"] == result["balanced_accuracy"] == 0.5
    assert result["malignant_sensitivity"] == 0
    assert result["invalid_terminal_rate"] == 0.5
    report = evaluate([row, invalid], 20)
    assert report["curves"][0]["groups"] == 2


def test_missing_model_assets_fail_closed():
    with pytest.raises(ValueError, match="No stock-model fallback"):
        resolve_assets(Config().model)


def test_changed_pixels_fail_before_observation(case, tmp_path):
    from sonomed_vlm.active_reader.io import file_hash

    for frame in case.frames:
        frame.sha256 = file_hash(tmp_path / frame.relative_path)
    Image.new("RGB", (32, 24), "white").save(tmp_path / "0.png")
    with pytest.raises(RuntimeError, match="private dataset audit"):
        Replay(case, tmp_path).reset(start="first")


def test_invalid_boxes_keep_frames_as_unknown(tmp_path):
    root = tmp_path / "rawframes" / "benign" / "case"
    root.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(root / "0.png")
    (tmp_path / "train.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "benign/case/0.png", "width": 8, "height": 8}],
                "categories": [{"id": 1, "name": "benign"}],
                "annotations": [{"image_id": 1, "category_id": 1, "bbox": [4, 2, -2, 3]}],
            }
        )
    )
    cases, metadata = load_source(tmp_path)
    assert cases[0].num_frames == 1
    assert cases[0].frames[0].annotation_status == "unannotated"
    assert cases[0].frames[0].boxes_xyxy == []
    assert (
        metadata["issues"][0]["resolution"]
        == "retain_pixels_mark_frame_unannotated_disable_case_relevance"
    )


def test_config_env_validation(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  root: ${NONEXISTENT_READER_TEST_VAR}\n")
    monkeypatch.delenv("NONEXISTENT_READER_TEST_VAR", raising=False)
    with pytest.raises(ValueError, match="environment variable"):
        load_config(path)


def test_cross_label_duplicates_quarantine_without_relabeling(case):
    from sonomed_vlm.active_reader.data.audit import quarantine_conflicts

    a, b, c = deepcopy(case), deepcopy(case), deepcopy(case)
    a.video_id, b.video_id, c.video_id = "a", "b", "c"
    a.group_id = b.group_id = "conflict"
    c.group_id = "safe"
    b.label = "malignant"
    retained, report = quarantine_conflicts([a, b, c], {"exact_overlap_pairs": [["a", "b"]]})
    assert [x.video_id for x in retained] == ["c"]
    assert report["quarantine"]["cases"] == ["a", "b"]
    assert (a.label, b.label) == ("benign", "malignant")


def test_grpo_actual_gradient_and_zero_variance():
    import torch

    from sonomed_vlm.active_reader.training.grpo import advantages, grpo_loss

    logits = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    optimizer = torch.optim.SGD([logits], lr=0.1)
    old = logits.detach().log_softmax(-1)
    adv, zero = advantages([1, -1])
    assert not zero
    loss, _ = grpo_loss(logits.log_softmax(-1), old, old, adv, kl_weight=0)
    loss.backward()
    optimizer.step()
    assert logits[0] > logits[1]
    adv, zero = advantages([2, 2, 2, 2])
    assert zero and torch.equal(adv, torch.zeros(4))


def test_action_grammar_limits_actions_and_probability_support():
    import torch

    from sonomed_vlm.active_reader.env.observation import Observation
    from sonomed_vlm.active_reader.models.constraints import ActionGrammar, restrict_logits

    class ByteTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(text.encode()) for text in texts]}

    observation = Observation((3, 9), (), legal_inspections=((2, "local"),), max_evidence_frames=1)
    grammar = ActionGrammar(observation, ByteTokenizer(), 256)
    valid = [*b'{"action":"inspect","center_bin":2,"scale":"local"}', 256]
    assert all(
        token in allowed
        for token, allowed in zip(valid, grammar.constraints_for(valid), strict=True)
    )
    with pytest.raises(ValueError):
        grammar.constraints_for(list(b'{"action":"inspect","center_bin":8,"scale":"local"}'))
    with pytest.raises(ValueError):
        grammar.constraints_for(
            list(b'{"action":"stop","prediction":"benign","evidence_frames":[4]}')
        )
    final = ActionGrammar(observation, ByteTokenizer(), 256, final_only=True)
    with pytest.raises(ValueError):
        final.constraints_for(valid)
    logits = torch.tensor([[2.0, 3.0, 100.0]], requires_grad=True)
    restricted = restrict_logits(logits, [[0, 1]]).log_softmax(-1)
    assert torch.allclose(restricted[0, :2], logits[0, :2].log_softmax(-1))
    restricted[0, 0].backward()
    assert logits.grad[0, 2] == 0
