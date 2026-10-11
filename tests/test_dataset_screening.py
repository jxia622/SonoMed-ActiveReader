"""Budget, sweep-boundary, and clean-input guarantees for the small dataset screen."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def screen_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "screen_acouslic", scripts / "screen_acouslic_qwen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_possible_screen_paths_respect_budget_and_sweep_boundaries():
    module = screen_module()
    for seed in (7, 19):
        start = module.initial(seed)
        assert len(set(start)) == 6
        for first in module.candidate_menu(start, start):
            observed = start + first["indices"]
            for second in module.candidate_menu(observed, first["indices"]):
                final = observed + second["indices"]
                assert len(set(final)) == 12
                for action in (first, second):
                    assert len(action["indices"]) == 3
                    assert {i // 140 + 1 for i in action["indices"]} == {action["sweep"]}
    for budget in (6, 12, 24):
        indices = module.stratified(budget)
        assert len(set(indices)) == budget
        assert all(sum(i // 140 == sweep for i in indices) == budget // 6 for sweep in range(6))


def test_screen_strips_metadata_and_rejects_changed_pixels(tmp_path):
    import torch
    from PIL import Image, PngImagePlugin

    module = screen_module()
    frames = tmp_path / "A01/frames"
    frames.mkdir(parents=True)
    path = frames / "0000.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("hidden_label", "optimal")
    Image.new("RGB", (4, 4), "red").save(path, pnginfo=metadata)
    (tmp_path / "frame_hashes.json").write_text(
        json.dumps({"A01": {"0": hashlib.sha256(path.read_bytes()).hexdigest()}})
    )

    class Processor:
        def __init__(self):
            self.tokenizer = self

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(text) - ord("A")]}

        def apply_chat_template(self, messages, **kwargs):
            images = [c["image"] for c in messages[0]["content"] if c["type"] == "image"]
            assert len(images) == 1 and not images[0].info
            assert not getattr(images[0], "filename", None)
            return {
                "input_ids": torch.tensor([[3, 4]]),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, **kwargs):
            assert kwargs["logits_to_keep"] == 1
            return SimpleNamespace(logits=torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))

    scorer = module.FrozenChoice(
        SimpleNamespace(processor=Processor(), model=Model()), tmp_path, tmp_path / "out"
    )
    assert scorer.choose("A01", [0], "Select A or B", 2, "fixture") == 1
    record = json.loads((tmp_path / "out/inputs/0000.json").read_text())
    assert sum(record["probabilities"]) == pytest.approx(1)
    Image.new("RGB", (4, 4), "blue").save(path)
    with pytest.raises(AssertionError):
        scorer.choose("A01", [0], "Select A or B", 2, "changed")
