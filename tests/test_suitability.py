from copy import deepcopy

import pytest

from sonomed_vlm.active_reader.data.manifest import Case, Frame
from sonomed_vlm.active_reader.suitability import AuditReader, development_plan, union_area


def case(split="dev"):
    frames = [Frame(i, str(i), f"{i}.png", 20, 20, "present", [[2, 2, 6, 6]]) for i in range(31)]
    return Case("private", "private", "benign", frames, group_id="g", split=split)


def test_sealed_rejected_before_reader_or_pixels(tmp_path):
    reader = AuditReader(None, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="sealed"):
        reader.score(case("test"), [0])
    with pytest.raises(ValueError, match="sealed"):
        development_plan(case("test"))


def test_dense_plan_covers_every_frame_exactly_once_and_budgets_match():
    plan = development_plan(case())
    dense = [i for row in plan if row["method"] == "dense_chunk" for i in row["indices"]]
    assert dense == list(range(31))
    singles = [r["indices"][0] for r in plan if r["method"] == "single"]
    assert singles == list(range(31))
    for row in plan:
        if "budget" in row:
            assert len(row["indices"]) == len(set(row["indices"])) == row["budget"]
        if row["method"] == "uniform":
            assert row["indices"][0] == 0 and row["indices"][-1] == 30


def test_nonprivileged_plan_independent_of_annotations_and_diagnosis():
    a, b = case(), deepcopy(case())
    b.label = "malignant"
    for frame in b.frames:
        frame.annotation_status, frame.boxes_xyxy = "unannotated", []

    def ordinary(x):
        return [r for r in development_plan(x) if r["method"] != "privileged_area"]

    assert ordinary(a) == ordinary(b)


def test_union_area_handles_overlap_and_empty():
    assert union_area([]) == 0
    assert union_area([[0, 0, 2, 2], [1, 1, 3, 3]]) == 7


def test_reader_rejects_source_change_and_strips_metadata(tmp_path):
    import hashlib

    from PIL import Image, PngImagePlugin

    c = case()
    image = Image.new("RGB", (20, 20), (20, 40, 60))
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "private-source-metadata")
    image.save(tmp_path / "0.png", pnginfo=info)
    c.frames[0].sha256 = "0" * 64
    with pytest.raises(ValueError, match="Source pixels changed"):
        AuditReader(None, tmp_path, tmp_path / "out").score(c, [0])
    c.frames[0].sha256 = hashlib.sha256((tmp_path / "0.png").read_bytes()).hexdigest()
    c.frames[0].pixel_sha256 = hashlib.sha256(b"20x20:" + image.tobytes()).hexdigest()

    class Reader:
        def __init__(self):
            self.cache = {}

        def score(self, obs):
            assert not obs.images[0].info
            assert not getattr(obs.images[0], "filename", None)
            assert obs.images[0].tobytes() == image.tobytes()
            return {"prediction": "benign"}

    _, result = AuditReader(Reader(), tmp_path, tmp_path / "out").score(c, [0])
    assert result["prediction"] == "benign"


def test_history_predictors_exclude_labels_and_future_pixels(monkeypatch):
    from pathlib import Path

    import numpy as np

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_suitability_history import design

    row = {
        "video_id": "private",
        "label": "benign",
        "context": {
            "total_frames": 31,
            "observed_indices": [0, 1, 2, 3, 16, 17, 18, 19],
            "returned_indices": [16, 17, 18, 19],
            "remaining_frames": 4,
        },
        "outcomes": [
            {
                "action": {"action": "inspect", "scale": "local"},
                "new_indices": [4, 5, 6, 7],
                "log_score": -0.1,
            },
            {
                "action": {"action": "refine", "direction": "right"},
                "new_indices": [20, 21, 22, 23],
                "log_score": -0.9,
            },
        ],
    }
    features = {"private": np.random.default_rng(3).normal(size=(31, 16))}
    proj = (np.zeros(16), np.eye(16)[:, :8])
    before = design([row], features, proj, "history")
    changed = deepcopy(row)
    changed["label"] = "malignant"
    changed["outcomes"][0]["log_score"] = -3
    alternative = deepcopy(features)
    unseen = sorted(set(range(31)) - set(row["context"]["observed_indices"]))
    alternative["private"][unseen] = 100
    assert np.array_equal(before, design([changed], alternative, proj, "history"))
    alternative["private"][:4] *= -1
    assert not np.allclose(before, design([row], alternative, proj, "history"))
    assert np.array_equal(
        design([row], features, proj, "current_frame"),
        design([row], alternative, proj, "current_frame"),
    )


def test_ssim_identity_and_large_change(monkeypatch):
    from pathlib import Path

    import numpy as np

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_breastvid_structure import ssim_pairs

    identical = np.zeros((2, 16, 16), dtype=np.uint8)
    assert ssim_pairs(identical, 1)[0] == pytest.approx(1)
    identical[1] = 255
    assert ssim_pairs(identical, 1)[0] < 0.001


def test_reader_summary_weights_short_dense_tail_by_frame_count(monkeypatch):
    import math
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_suitability_reader import summarize

    rows = []
    for number in range(4):
        c = case()
        label = "benign" if number < 2 else "malignant"
        evaluations = []
        for spec in development_plan(c):
            if spec["method"] == "dense_chunk":
                true_probability = 0.9 if spec["chunk"] == 2 else 0.3
            else:
                true_probability = sum(0.8 if i == 0 else 0.2 for i in spec["indices"]) / len(
                    spec["indices"]
                )
            probabilities = {
                label: true_probability,
                "malignant" if label == "benign" else "benign": 1 - true_probability,
            }
            evaluations.append(
                {
                    **spec,
                    "probabilities": probabilities,
                    "prediction": max(probabilities, key=probabilities.get),
                }
            )
        rows.append(
            {
                "video_id": str(number),
                "group_id": str(number),
                "label": label,
                "num_frames": 31,
                "evaluations": evaluations,
            }
        )
    result = summarize(rows)
    dense = result["methods"]["dense_chunk12_weighted_mean_probability"]
    assert dense["balanced_accuracy"] == 0
    assert dense["mean_log_score"] == pytest.approx(math.log((24 * 0.3 + 7 * 0.9) / 31))
    assert result["methods"]["privileged_best_single_all_frames"]["balanced_accuracy"] == 1
