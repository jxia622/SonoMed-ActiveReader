"""Real tiny Qwen architecture: selection likelihoods, immutable reader, and gradients."""

import json
from types import SimpleNamespace

import pytest


def test_navigation_qwen_reader_and_policy_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import GenerationConfig, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from sonomed_vlm.active_reader.config import Config
    from sonomed_vlm.active_reader.data.manifest import Case, Frame
    from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy
    from sonomed_vlm.active_reader.navigation.browser import NavigationBrowser
    from sonomed_vlm.active_reader.navigation.reader import FrozenReader
    from sonomed_vlm.active_reader.navigation.training import train_navigation
    from sonomed_vlm.active_reader.training.grpo import parameter_digest

    cfg = Qwen3VLConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        ),
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
        vision_end_token_id=4,
    )

    class Processor:
        def __init__(self):
            self.tokenizer = self
            self.mapping = {}

        def __call__(self, texts, **kwargs):
            def encode(text):
                value = json.loads(text)
                if "prediction" in value:
                    return [12 if value["prediction"] == "benign" else 13]
                return [self.mapping[text]]

            return {
                "input_ids": [encode(t) for t in texts]
                if isinstance(texts, list)
                else encode(texts)
            }

        def apply_chat_template(self, messages, **kwargs):
            first_text = messages[1]["content"][0]["text"]
            meta = json.loads(first_text) if first_text.startswith("{") else {}
            self.mapping = {
                json.dumps(a, separators=(",", ":")): 30 + i
                for i, a in enumerate(meta.get("legal_actions", []))
            }
            images = [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]
            ids = [5] + [v for _ in images for v in [3, 1, 4]] + [6]
            types = [0] + [v for _ in images for v in [0, 1, 0]] + [0]
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([types]),
                "pixel_values": torch.cat(
                    [torch.full((4, 24), im.getpixel((0, 0))[0] / 255) for im in images]
                ),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            }

        def decode(self, tokens, **kwargs):
            return next(text for text, token in self.mapping.items() if token == int(tokens[0]))

        def save_pretrained(self, path):
            path.mkdir()
            (path / "fixture.json").write_text("{}")

    def make_policy(training):
        torch.manual_seed(42)
        model = get_peft_model(
            Qwen3VLForConditionalGeneration(cfg),
            LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
            ),
            adapter_name="policy",
        )
        origin = tmp_path / ("selector" if training else "reader")
        model.save_pretrained(origin, selected_adapters=["policy"])
        if training:
            model.load_adapter(origin / "policy", adapter_name="reference", is_trainable=False)
        model.generation_config = GenerationConfig(pad_token_id=0, bos_token_id=5, eos_token_id=255)
        p = QwenPolicy.__new__(QwenPolicy)
        p.model, p.processor, p.for_training = model, Processor(), training
        p.config = SimpleNamespace(max_new_tokens=2, constrain_actions=True)
        p.use_adapter("policy")
        return p

    frames = []
    for i in range(24):
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(tmp_path / f"{i}.png")
        frames.append(Frame(i, str(i), f"{i}.png", 8, 8, "unannotated"))
    cases = [
        Case(label, "private/" + label, label, frames, group_id=label, split="train")
        for label in ("benign", "malignant")
    ]
    config = Config()
    config.environment.max_unique_frames = 12
    config.reward.relevance_weight = 0
    config.training.steps = 2
    config.training.learning_rate = 0.003
    selector = make_policy(True)
    reader = FrozenReader(make_policy(False))
    b = NavigationBrowser(cases[0], tmp_path, config)
    obs = b.reset(42, 12)
    gen = selector.generate(obs, sample=True, retain=True, seed=22)
    assert torch.allclose(selector.log_probs(gen).detach(), gen.behavior_log_probs, atol=1e-5)
    assert gen.record["visible_image_indices"] == list(obs.indices)
    assert len(gen.record["image_grid_thw"]) == len(obs.images)
    assert sum(gen.record["loss_mask"]) == len(gen.tokens)
    b.step(gen.text)
    full_before = selector.generate(b.observation(), seed=9).record
    b.history_mode = "current_frame"
    current_before = selector.generate(b.observation(), seed=9).record
    old = next(i for i in b.env.visited if i not in b.returned)
    b.env._cache[old] = Image.new("RGB", (8, 8), "white")
    current_after = selector.generate(b.observation(), seed=9).record
    b.history_mode = "full"
    full_after = selector.generate(b.observation(), seed=9).record
    assert current_before["processed_pixels_sha256"] == current_after["processed_pixels_sha256"]
    assert full_before["processed_pixels_sha256"] != full_after["processed_pixels_sha256"]
    assert full_before["input_ids"] == full_after["input_ids"]
    result = reader.score(b.observation(reader=True))
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert reader.score(b.observation(reader=True))["executed_model_calls"] == 0
    before = parameter_digest(reader.policy.model, lambda n, p: True)
    result = train_navigation(
        selector, reader, cases, tmp_path, config, tmp_path / "run", {"fixture": True}
    )
    assert result["language_adapter_changed"] and result["frozen_reader_unchanged"]
    assert parameter_digest(reader.policy.model, lambda n, p: True) == before
    assert all(p.grad is None for p in reader.policy.model.parameters())
    records = json.loads((tmp_path / "run" / "steps" / "000001.json").read_text())
    for row in records["trajectories"]:
        assert row["unique_frames"] == 12
        assert all(t["action"]["action"] in ("inspect", "refine") for t in row["turns"])
        assert sum(t["private_reader_logscore_increment"] for t in row["turns"]) == pytest.approx(
            row["private_reader_gain"]
        )
    uninterrupted = parameter_digest(selector.model, lambda n, p: ".policy." in n)
    resumed = train_navigation(
        selector,
        reader,
        cases,
        tmp_path,
        config,
        tmp_path / "resumed",
        {"fixture": True},
        resume=tmp_path / "run" / "step-000001",
    )
    assert resumed["save_reload_identical"]
    assert parameter_digest(selector.model, lambda n, p: ".policy." in n) == uninterrupted

    # The new warm-start must learn only from fit states and leave the reader fixed.
    from sonomed_vlm.active_reader.navigation.preference import (
        score_states,
        train_preferences,
    )
    from sonomed_vlm.active_reader.navigation.probe import collect
    from sonomed_vlm.active_reader.navigation.training import reset_selector

    rows = collect(
        reader,
        {"fit": cases[:1], "validation": cases[1:]},
        tmp_path,
        config,
        tmp_path / "probe",
        menu_size=3,
    )
    # Synthetic preferences verify mechanics; they are not medical evidence labels.
    for row in rows:
        for i, outcome in enumerate(row["outcomes"]):
            outcome["log_score"] = float(i)
    reset_selector(selector)
    index = {c.video_id: c for c in cases}
    initial = score_states(selector, rows, index, tmp_path, config, tmp_path / "rank", "full")
    preference = train_preferences(
        selector,
        rows,
        initial,
        index,
        tmp_path,
        config,
        tmp_path / "preference",
        "fixture",
        "full",
        steps=2,
        learning_rate=0.003,
    )
    assert preference["language_adapter_changed"]
    assert preference["frozen_reference_and_vision_unchanged"]
    assert preference["save_reload_identical"]
    learned = score_states(selector, rows, index, tmp_path, config, tmp_path / "rank_after", "full")
    initial_by_state = {r["state_id"]: r for r in initial}
    fit = [r for r in learned if r["role"] == "fit"]
    assert sum(r["log_probabilities"][2] - r["log_probabilities"][0] for r in fit) > sum(
        initial_by_state[r["state_id"]]["log_probabilities"][2]
        - initial_by_state[r["state_id"]]["log_probabilities"][0]
        for r in fit
    )
    for path in (tmp_path / "preference" / "steps").glob("*.json"):
        assert json.loads(path.read_text())["role"] == "fit"
    assert parameter_digest(reader.policy.model, lambda n, p: True) == before
    assert all(p.grad is None for p in reader.policy.model.parameters())
