"""Reachability, split isolation, and real tiny-Qwen constrained action gradients."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sonomed_vlm.active_reader.acouslic.environment import (
    audit_reachability,
    initial,
    inspect,
    legal_centers,
)


def test_every_unseen_frame_is_reachable_and_refinement_can_return_to_old_region():
    for seed in (7, 19, 42):
        observed = initial(seed)
        for center in legal_centers(observed):
            new = inspect(center, observed)
            assert len(new) == 3 and not set(new) & set(observed)
            assert center in new and len({i // 140 for i in new}) == 1
        old_region = observed[0]
        elsewhere = inspect(839, observed)
        seen = observed + elsewhere
        refined = inspect(old_region - 1, seen)
        assert max(abs(i - old_region) for i in refined) <= 3
        assert len(set(seen + refined)) == 12
        assert audit_reachability(np.zeros(840), seed)["reachable_frames"] == 840
        with pytest.raises(ValueError):
            inspect(observed[0], observed)


def test_partition_groups_repeats_and_excludes_screened_subjects_from_evaluation():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "prepare_acouslic", scripts / "prepare_acouslic_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"subject_id": str(i), "uuid": f"scan-{i}-{j}"} for i in range(70) for j in range(2)]
    screened = [{"subject_id": str(i), "source_id": f"scan-{i}-1"} for i in range(5)]
    partition = module.partition(rows, screened, 42)
    assert len({r["subject_id"] for r in partition}) == 60
    assert [
        sum(r["split"] == split for r in partition) for split in ("train", "tune", "evaluation")
    ] == [32, 8, 20]
    assert all(r["split"] == "train" for r in partition if r["screening_exposed"])
    assert partition == module.partition(list(reversed(rows)), screened, 42)


def test_case_loader_refuses_heldout_access_by_default(tmp_path):
    import json

    from sonomed_vlm.active_reader.acouslic.data import Cases

    (tmp_path / "partition.json").write_text(
        json.dumps({"protocol": "acouslic-acquisition-pilot-v1"})
    )
    (tmp_path / "development_manifest.json").write_text(
        json.dumps([{"alias": "train", "subject_id": "one", "split": "train"}])
    )
    (tmp_path / "sealed_evaluation_manifest.json").write_text(
        json.dumps([{"alias": "heldout", "subject_id": "two", "split": "evaluation"}])
    )
    data = Cases(tmp_path)
    with pytest.raises(ValueError, match="Held-out"):
        data.subset("evaluation")
    with pytest.raises(AssertionError, match="not admitted"):
        data.quality("heldout")
    assert Cases(tmp_path, allow_evaluation=True).subset("evaluation")[0]["alias"] == "heldout"


def test_tiny_qwen_action_probabilities_credit_and_history(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import GenerationConfig, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from sonomed_vlm.active_reader.acouslic.policy import Choices
    from sonomed_vlm.active_reader.acouslic.training import episode, set_reference_to_policy
    from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy
    from sonomed_vlm.active_reader.training.grpo import grpo_loss, parameter_digest

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

        def __call__(self, texts, **kwargs):
            def encode(text):
                return [30 + int(c) if c.isdigit() else 50 + ord(c) - ord("A") for c in text]

            return {
                "input_ids": [encode(t) for t in texts]
                if isinstance(texts, list)
                else encode(texts)
            }

        def apply_chat_template(self, messages, **kwargs):
            images = [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]
            ids = [5] + [v for _ in images for v in [3, 1, 4]] + [6]
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "mm_token_type_ids": torch.tensor(
                    [[0] + [v for _ in images for v in [0, 1, 0]] + [0]]
                ),
                "pixel_values": torch.cat(
                    [torch.full((4, 24), im.getpixel((0, 0))[0] / 255) for im in images]
                ),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            }

        def save_pretrained(self, path):
            path.mkdir()

    class Data:
        changed = False

        def images(self, alias, indices, blank=False):
            result, hashes = [], []
            for i in indices:
                v = 0 if blank else (255 if self.changed and i == initial(7)[0] else i % 255)
                im = Image.new("RGB", (8, 8), (v, 0, 0))
                result.append(im)
                hashes.append({"index": i, "sha256": hashlib.sha256(im.tobytes()).hexdigest()})
            return result, hashes

        def quality(self, alias):
            q = np.zeros(840)
            q[300:700] = 1
            return q, q.astype(int)

    torch.manual_seed(42)
    model = get_peft_model(
        Qwen3VLForConditionalGeneration(cfg),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        adapter_name="policy",
    )
    original = tmp_path / "original"
    model.save_pretrained(original, selected_adapters=["policy"])
    model.load_adapter(original / "policy", adapter_name="reference", is_trainable=False)
    model.generation_config = GenerationConfig(pad_token_id=0, eos_token_id=255, bos_token_id=5)
    policy = QwenPolicy.__new__(QwenPolicy)
    policy.model, policy.processor, policy.for_training = model, Processor(), True
    policy.config = SimpleNamespace()
    policy.use_adapter("policy")
    data = Data()
    choices = Choices(policy, data)
    observed = initial(7)
    gen = choices.inspect("fixture", observed, observed, "full", 42, sample=True, retain=True)
    assert len(gen.tokens) == 3 and int(gen.text) not in observed
    current = policy.log_probs(gen)
    assert torch.allclose(current.detach().cpu(), gen.behavior_log_probs, atol=1e-5)
    with torch.no_grad():
        ref = policy.log_probs(gen, "reference").detach()
    current = policy.log_probs(gen)
    frozen = parameter_digest(model, lambda n, p: ".reference." in n or "visual" in n)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    loss, _ = grpo_loss(current, gen.behavior_log_probs, ref, torch.tensor(1.0))
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
    optimizer.step()
    assert frozen == parameter_digest(model, lambda n, p: ".reference." in n or "visual" in n)
    set_reference_to_policy(policy)
    assert torch.allclose(
        policy.log_probs(gen).detach(), policy.log_probs(gen, "reference").detach(), atol=1e-5
    )
    latest = inspect(int(gen.text), observed)
    all_seen = observed + latest
    full_before = choices.inspect("fixture", all_seen, latest, "full", 1).record
    current_before = choices.inspect("fixture", all_seen, latest, "current", 1).record
    data.changed = True
    full_after = choices.inspect("fixture", all_seen, latest, "full", 1).record
    current_after = choices.inspect("fixture", all_seen, latest, "current", 1).record
    assert full_before["processed_pixels_sha256"] != full_after["processed_pixels_sha256"]
    assert current_before["processed_pixels_sha256"] == current_after["processed_pixels_sha256"]
    row, _ = episode(choices, {"alias": "fixture", "split": "train"}, 7, "full")
    assert len(set(row["indices"])) == 12
    assert sum(t["private_quality_increment"] for t in row["turns"]) == pytest.approx(row["reward"])
