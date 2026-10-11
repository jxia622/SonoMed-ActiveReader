#!/usr/bin/env python3
"""Development-only, zero-update probe of image-relative inspection actions.

Preserves all unseen frame centers and a 12-frame budget. All conditions can
reference all observed locations; only the visible pixels differ. Nothing in
this script admits held-out subjects or changes the running flat-action pilot.
"""

import argparse
import hashlib
import json
import string
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from sonomed_vlm.active_reader.acouslic.data import Cases
from sonomed_vlm.active_reader.acouslic.environment import initial, inspect, legal_centers, outcome
from sonomed_vlm.active_reader.acouslic.policy import ChoiceGrammar, Choices
from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.io import code_revision, file_hash, read_json, write_json
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy
from sonomed_vlm.active_reader.training.grpo import parameter_digest


def actions(observed):
    mapping = {}
    unseen = set(legal_centers(observed))
    for j, anchor in enumerate(observed):
        for center in sorted(unseen):
            if center // 140 == anchor // 140:
                mapping[f"{string.ascii_uppercase[j]}{center - anchor:+04d}"] = center
    assert set(mapping.values()) == unseen, "Image-relative actions must retain full reachability"
    return mapping


class AnchoredChoices(Choices):
    def prepare(self, alias, indices, question, texts, purpose, blank=False):
        images, hashes = self.data.images(alias, indices, blank)
        content = []
        for index, image in zip(indices, images, strict=True):
            content.extend(
                [
                    {"type": "text", "text": f"Image {self.labels[index]}:"},
                    {"type": "image", "image": image},
                ]
            )
        content.append({"type": "text", "text": question})
        inputs = dict(
            self.policy.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
        grammar = ChoiceGrammar(self.policy.processor.tokenizer, texts)
        assert grammar.length == 5
        record = {
            "alias": alias,
            "purpose": purpose,
            "question": question,
            "image_indices": list(indices),
            "displayed_image_labels": [self.labels[i] for i in indices],
            "clean_images": hashes,
            "blank_images": blank,
            "input_ids": inputs["input_ids"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "legal_choices": list(texts),
            "processed_pixels_sha256": hashlib.sha256(
                inputs["pixel_values"].contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "visual_input_tokens": int(
                (inputs["input_ids"] == self.policy.model.config.image_token_id).sum()
            ),
            "image_presentations": len(images),
            "context": {"protocol": "acouslic_anchored_actions_probe_v1"},
        }
        return inputs, grammar, record

    def inspect(self, alias, observed, latest, mode, seed=0, sample=False, retain=False):
        self.labels = {index: string.ascii_uppercase[j] for j, index in enumerate(observed)}
        displayed = list(latest) if mode == "current" else list(observed)
        mapping = actions(observed)
        locations = [
            {
                "anchor": self.labels[i],
                "global_frame": i,
                "sweep": i // 140 + 1,
                "local_frame": i % 140,
            }
            for i in observed
        ]
        question = (
            "You are navigating six prenatal ultrasound sweeps to find a suitable transverse "
            "fetal abdominal plane for circumference measurement. Each sweep has 140 frames. "
            "Choose an observed image as an anchor and a signed frame offset within that sweep. "
            "Use visible anatomy to choose a promising region and decide which direction and "
            "distance to inspect. Positive offsets move later; negative offsets move earlier. "
            "The inspection returns the three nearest unseen frames around your target. "
            "You may reference ANY previous anchor, including older locations whose images "
            "are not currently displayed. Every unseen frame remains reachable.\n"
            f"All observed anchors: {json.dumps(locations)}\n"
            f"Currently displayed anchors: {json.dumps([self.labels[i] for i in displayed])}\n"
            f"Most recently observed anchors: {json.dumps([self.labels[i] for i in latest])}\n"
            f"Remaining new-frame budget: {12 - len(observed)}\n"
            "Reply with exactly an anchor letter, a + or - sign, and three offset digits. "
            "Do not target an already observed frame or leave the anchor's sweep."
        )
        generation = self.generate(
            alias,
            displayed,
            question,
            list(mapping),
            "anchored_inspection",
            sample,
            seed,
            retain,
            blank=mode == "geometry",
        )
        raw = generation.text
        center = mapping[raw]
        generation.record["action_center"] = center
        generation.record["action_mapping"] = mapping
        generation.record["context"].update(
            mode=mode,
            observed=list(observed),
            latest=list(latest),
            budget_remaining=12 - len(observed),
        )
        generation.text = str(center)
        return generation


def run(args):
    campaign = read_json(args.run / "campaign.json")
    protocol = campaign["protocol"]
    assert campaign["code_revision"] == code_revision()
    assert campaign["data_partition_sha256"] == file_hash(args.data / "partition.json")
    assert read_json(args.run / "warm/report.json")["gate_passed"]
    assert not args.output.exists(), "Preserve existing probe results"
    data = Cases(args.data)
    tune = data.subset("tune")
    assert len(tune) == 8
    config = load_config(args.config).model.model_copy(
        update={"dtype": "float32", "image_max_pixels": protocol["selector_max_pixels"]}
    )
    assert (
        file_hash(Path(config.adapter) / "adapter_model.safetensors")
        == protocol["original_adapter_sha256"]
    )
    torch.set_float32_matmul_precision("highest")
    policy = QwenPolicy(config, for_training=True)
    policy.load_policy(args.run / "warm/checkpoint")
    before = parameter_digest(policy.model, lambda name, p: ".policy." in name)
    choices = AnchoredChoices(policy, data)
    checks = []
    old = initial(7)
    latest = inspect(50, old)
    for mode in ("full", "current", "geometry"):
        gen = choices.inspect(
            data.subset("train")[0]["alias"],
            old + latest,
            latest,
            mode,
            seed=20261011,
            sample=True,
            retain=True,
        )
        current = policy.log_probs(gen)
        drift = float((current.detach().cpu() - gen.behavior_log_probs).abs().max())
        assert drift <= protocol["max_behavior_logprob_drift"]
        policy.model.zero_grad(set_to_none=True)
        (-current.sum()).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.model.parameters() if p.requires_grad], 1
        )
        assert torch.isfinite(norm) and norm > 0
        checks.append(
            {"mode": mode, "drift": drift, "gradient_norm": float(norm), "input": gen.record}
        )
    policy.model.zero_grad(set_to_none=True)
    write_json(args.output / "engineering_checks.json", checks)
    rows = []
    for mode in ("full", "current", "geometry"):
        for case in tune:
            q, labels = data.quality(case["alias"])
            for seed in protocol["evaluation_starts"]:
                observed, latest = initial(seed), initial(seed)
                initial_value = float(q[observed].max())
                turns = []
                for turn in range(2):
                    previous = float(q[observed].max())
                    gen = choices.inspect(case["alias"], observed, latest, mode, seed=turn * 31)
                    center = int(gen.text)
                    latest = inspect(center, observed)
                    observed.extend(latest)
                    turns.append(
                        {
                            "center": center,
                            "returned_indices": latest,
                            "private_quality_increment": float(q[observed].max()) - previous,
                            "input": gen.record,
                        }
                    )
                rows.append(
                    {
                        "alias": case["alias"],
                        "split": "tune",
                        "mode": mode,
                        "seed": seed,
                        "initial_wfss": initial_value,
                        "turns": turns,
                        **outcome(q, labels, observed),
                    }
                )
                write_json(args.output / "acquisition.json", rows)
            print(f"Anchored zero-update probe: {mode} {case['alias']}", flush=True)
    assert before == parameter_digest(policy.model, lambda name, p: ".policy." in name)
    report = {
        "status": "complete",
        "subjects": 8,
        "optimizer_updates": 0,
        "scope": "Exploratory development-only action-representation probe, not RL",
        "script_sha256": file_hash(Path(__file__)),
        "policy_unchanged": True,
        "warm_checkpoint_sha256": file_hash(
            args.run / "warm/checkpoint/policy/adapter_model.safetensors"
        ),
        "max_behavior_logprob_drift": max(x["drift"] for x in checks),
        "all_unseen_frames_reachable_in_all_modes": True,
        "methods": {
            m: {
                "mean_acquisition_wfss": float(
                    np.mean([r["best_wfss"] for r in rows if r["mode"] == m])
                ),
                "mean_inspection_gain": float(
                    np.mean([r["best_wfss"] - r["initial_wfss"] for r in rows if r["mode"] == m])
                ),
            }
            for m in ("full", "current", "geometry")
        },
        "heldout_opened": False,
        "breastvid_test_opened": False,
    }
    write_json(args.output / "report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/navigation_crc.yaml"))
    run(p.parse_args())
