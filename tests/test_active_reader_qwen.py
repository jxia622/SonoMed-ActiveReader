"""Tiny random Qwen3-VL integration: software evidence, not a medical-model test."""

import json
from types import SimpleNamespace

import pytest


def test_qwen_multimodal_likelihood_mask_gradient_reference_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import GenerationConfig, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from sonomed_vlm.active_reader.env.observation import Observation
    from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy
    from sonomed_vlm.active_reader.training.grpo import grpo_loss, parameter_digest

    torch.manual_seed(12)
    configuration = Qwen3VLConfig(
        text_config=dict(
            vocab_size=32,
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
    base = Qwen3VLForConditionalGeneration(configuration)
    lora = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, lora, adapter_name="policy")
    origin = tmp_path / "original"
    model.save_pretrained(origin, selected_adapters=["policy"])
    model.load_adapter(origin / "policy", adapter_name="reference", is_trainable=False)
    model.generation_config = GenerationConfig(pad_token_id=0, eos_token_id=31, bos_token_id=5)

    class TinyProcessor:
        action_vocabulary = False

        def apply_chat_template(self, messages, **kwargs):
            if self.action_vocabulary:
                self.first_index = json.loads(messages[1]["content"][0]["text"])[
                    "observed_indices"
                ][0]
            images = [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]
            ids = [5] + [token for _ in images for token in [3, 1, 4]] + [6]
            types = [0] + [token for _ in images for token in [0, 1, 0]] + [0]
            pixels = torch.cat([torch.full((4, 24), im.getpixel((0, 0))[0] / 255) for im in images])
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([types]),
                "pixel_values": pixels,
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            }

        def decode(self, tokens, **kwargs):
            if self.action_vocabulary:
                # Tiny synthetic vocabulary: each sampled token spells a full legal action.
                return json.dumps(
                    {
                        "action": "stop",
                        "prediction": "benign" if int(tokens[0]) % 2 else "malignant",
                        "evidence_frames": [self.first_index],
                    }
                )
            return str(tokens.tolist())

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "synthetic_processor.json").write_text(json.dumps({"fixture": True}))

    policy = QwenPolicy.__new__(QwenPolicy)
    policy.model, policy.processor = model, TinyProcessor()
    policy.config, policy.for_training = SimpleNamespace(max_new_tokens=4), True
    obs = Observation(
        (0, 5),
        (Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "black")),
        total_frames=6,
        remaining_turns=2,
    )
    generation = policy.generate(obs, sample=True, seed=12, retain=True)
    assert generation.record["image_presentations"] == 2
    assert generation.inputs["pixel_values"][:4].sum() > generation.inputs["pixel_values"][4:].sum()
    assert generation.record["loss_mask"][:8] == [0] * 8
    assert sum(generation.record["loss_mask"]) == len(generation.tokens)
    frozen = parameter_digest(model, lambda n, p: "visual" in n or ".reference." in n)
    with torch.no_grad():
        reference = policy.log_probs(generation, "reference")
    current = policy.log_probs(generation)
    assert torch.allclose(current.detach(), generation.behavior_log_probs, atol=1e-5)
    assert torch.allclose(current.detach(), reference, atol=1e-5)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n and ".policy." in n for n, _ in trainable)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=0.01, weight_decay=0)
    loss, _ = grpo_loss(current, generation.behavior_log_probs, reference, 1.0)
    loss.backward()
    optimizer.step()
    assert parameter_digest(model, lambda n, p: "visual" in n or ".reference." in n) == frozen
    after = policy.log_probs(generation).detach()
    assert not torch.allclose(after, current.detach())
    policy.save(tmp_path / "saved")
    policy.load_policy(tmp_path / "saved")
    assert torch.allclose(policy.log_probs(generation).detach(), after)

    # Exercise actual rollout -> grouped return -> optimizer -> checkpoint -> resume.
    # This tiny random model and artificial vocabulary have no medical meaning.
    from sonomed_vlm.active_reader.config import Config, Environment, Training
    from sonomed_vlm.active_reader.data.manifest import Case, Frame
    from sonomed_vlm.active_reader.training.grpo import train

    frames = []
    for index in range(6):
        Image.new("RGB", (4, 4), (index * 30, 0, 0)).save(tmp_path / f"{index}.png")
        frames.append(Frame(index, str(index), f"{index}.png", 4, 4, "unannotated"))
    cases = [
        Case(label, "private/" + label, label, frames, group_id=label, split="train")
        for label in ("benign", "malignant")
    ]
    config = Config(
        environment=Environment(max_unique_frames=4, max_decision_turns=2),
        training=Training(steps=2, learning_rate=0.001),
    )
    policy.config.max_new_tokens = 1
    policy.processor.action_vocabulary = True

    class TinyActionTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": [
                    [13 if json.loads(text)["prediction"] == "benign" else 12] for text in texts
                ]
            }

    policy.processor.tokenizer = TinyActionTokenizer()
    policy.config.max_new_tokens = 2
    policy.config.constrain_actions = True
    constrained = policy.generate(obs, sample=True, seed=21, retain=True)
    assert constrained.record["token_constraints"] == [[12, 13], [31]]
    assert torch.allclose(
        policy.log_probs(constrained).detach(), constrained.behavior_log_probs, atol=1e-5
    )
    policy.config.constrain_actions = False
    policy.config.max_new_tokens = 1
    result = train(policy, cases, tmp_path, config, tmp_path / "uninterrupted", {"fixture": True})
    assert result["language_adapter_changed"] and result["save_reload_weights_identical"]
    uninterrupted = parameter_digest(model, lambda n, p: ".policy." in n)
    result = train(
        policy,
        cases,
        tmp_path,
        config,
        tmp_path / "resumed",
        {"fixture": True},
        resume=tmp_path / "uninterrupted" / "step-000001",
    )
    assert result["updates"] == 1
    assert uninterrupted == parameter_digest(model, lambda n, p: ".policy." in n)
