#!/usr/bin/env python3
"""Small, frozen-checkpoint ACOUSLIC screen. No optimizer or RL training.

The private plane labels never enter a selector/reader prompt. The marked
privileged controls may use labels to construct frame sets. This is a screening
diagnostic, not a held-out clinical benchmark or challenge submission.
"""

import argparse
import hashlib
import json
import random
import string
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image

from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.io import file_hash, read_json, write_json
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory
from sonomed_vlm.active_reader.training.grpo import parameter_digest


def stratified(k):
    assert k % 6 == 0
    return [s * 140 + round((j + 1) * 139 / (k // 6 + 1)) for s in range(6) for j in range(k // 6)]


def initial(seed):
    rng = random.Random(seed)
    return [s * 140 + rng.randrange(35, 105) for s in range(6)]


def candidate_menu(observed, latest):
    """Six sweep coverage actions plus two local refinements; no hidden pixels."""
    seen = set(observed)
    choices = []
    for s in range(6):
        pool = [i for i in range(s * 140, (s + 1) * 140) if i not in seen]
        center = max(pool, key=lambda i: (min(abs(i - j) for j in seen), -i))
        indices = sorted(sorted(pool, key=lambda i: (abs(i - center), i))[:3])
        choices.append({"sweep": s + 1, "kind": "coverage", "indices": indices})
    current = latest[len(latest) // 2]
    s = current // 140
    for direction in (-1, 1):
        center = min((s + 1) * 140 - 1, max(s * 140, current + direction * 12))
        pool = [i for i in range(s * 140, (s + 1) * 140) if i not in seen]
        indices = sorted(sorted(pool, key=lambda i: (abs(i - center), i))[:3])
        if indices not in [c["indices"] for c in choices]:
            choices.append({"sweep": s + 1, "kind": "local", "indices": indices})
    return choices


class FrozenChoice:
    def __init__(self, policy, samples, output):
        self.policy, self.samples, self.output = policy, samples, output
        self.calls = 0
        self.records = []
        self.frame_hashes = read_json(samples / "frame_hashes.json")

    def choose(self, alias, indices, question, count, purpose):
        letters = string.ascii_uppercase[:count]
        tokenizer = self.policy.processor.tokenizer
        answer_ids = [tokenizer(x, add_special_tokens=False)["input_ids"] for x in letters]
        assert all(len(x) == 1 for x in answer_ids)
        answer_ids = [x[0] for x in answer_ids]
        assert len(set(answer_ids)) == count
        content = []
        hashes = []
        for ordinal, i in enumerate(indices):
            path = self.samples / alias / "frames" / f"{i:04d}.png"
            assert file_hash(path) == self.frame_hashes[alias][str(i)]
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                pixels = rgb.tobytes()
                clean = Image.frombytes("RGB", rgb.size, pixels)
            hashes.append(hashlib.sha256(pixels).hexdigest())
            content.extend(
                [
                    {"type": "text", "text": f"Image {string.ascii_uppercase[ordinal]}:"},
                    {"type": "image", "image": clean},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": question + f"\nReply with exactly one letter from {', '.join(letters)}.",
            }
        )
        messages = [{"role": "user", "content": content}]
        inputs = dict(
            self.policy.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
        model = self.policy.model
        device = next(model.parameters()).device
        dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
        values = {
            k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            logits = (
                model(**values, use_cache=False, logits_to_keep=1).logits[0, -1, answer_ids].float()
            )
            probabilities = logits.softmax(0).cpu().tolist()
        selected = int(np.argmax(probabilities))
        record = {
            "call": self.calls,
            "alias": alias,
            "purpose": purpose,
            "image_indices": indices,
            "clean_image_sha256": hashes,
            "question": question,
            "input_ids": inputs["input_ids"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "legal_answer_tokens": answer_ids,
            "probabilities": probabilities,
            "selected": selected,
            "model_calls": 1,
            "score_definition": "single-token constrained categorical decision, then deterministic termination; uncalibrated",
        }
        write_json(self.output / "inputs" / f"{self.calls:04d}.json", record)
        self.records.append({k: v for k, v in record.items() if k != "input_ids"})
        self.calls += 1
        return selected


def reader_question():
    return (
        "These are clean frames from one prenatal ultrasound examination. "
        "Choose the single image that provides the most suitable transverse fetal abdominal "
        "plane for abdominal circumference measurement. Select based only on visible anatomy "
        "and plane quality. Do not estimate a measurement or infer a clinical diagnosis. "
        "Choose the image's letter."
    )


def run(args):
    assert not (args.output / "inputs").exists(), "Use a fresh output directory"
    config = load_config(args.config)
    manifest = read_json(args.samples / "screen_manifest.json")
    assert len(manifest) == 5 and all(r["split"] == "public_train_screening" for r in manifest)
    config.model.dtype = "bfloat16"
    policy = QwenPolicy(config.model)
    assert not any(p.requires_grad for p in policy.model.parameters())
    before = parameter_digest(policy.model, lambda n, p: True)
    write_json(
        args.output / "provenance.json",
        {
            "source_sha256": file_hash(Path(__file__)),
            "manifest_sha256": file_hash(args.samples / "screen_manifest.json"),
            "model": inventory(config.model),
            "training": False,
            "breastvid_test_opened": False,
            "scope": "five preselected public training cases, seeds 7 and 19, 12-frame primary budget; 24-frame reference",
            "limitations": "Exploratory screen; all cases development-exposed; no RL benefit or clinical validity claim.",
        },
    )
    policy.processor.save_pretrained(args.output / "processor")
    scorer = FrozenChoice(policy, args.samples, args.output)
    results = []
    controls = []
    for case in manifest:
        alias = case["alias"]
        private = np.load(args.samples / alias / "private_labels.npz")
        quality = private["quality"]
        for seed in (7, 19):
            start = initial(seed)
            sets = {
                "uniform": stratified(12),
                "random": sorted(random.Random(seed).sample(range(840), 12)),
            }
            covered = list(start)
            while len(covered) < 12:
                new = max(
                    (i for i in range(840) if i not in covered),
                    key=lambda i: (min(abs(i - j) for j in covered), -i),
                )
                covered.append(new)
            sets["coverage"] = sorted(covered)
            for mode in ("current", "history"):
                observed, latest = list(start), list(start)
                for turn in range(2):
                    menu = candidate_menu(observed, latest)
                    displayed = sorted(observed) if mode == "history" else list(latest)
                    geometry = [
                        {
                            "letter": string.ascii_uppercase[i],
                            "sweep": f // 140 + 1,
                            "frame_in_sweep": f % 140,
                        }
                        for i, f in enumerate(displayed)
                    ]
                    question = (
                        "You are gathering evidence to select a fetal abdominal circumference measurement plane. "
                        "The examination contains six distinct 140-frame sweeps. Choose the next inspection "
                        "most likely to reveal a suitable plane, using the images already observed. "
                        "Candidate images have not yet been shown. Only the images above are available.\n"
                        f"Displayed image locations: {json.dumps(geometry)}\n"
                        f"All previously observed locations: {json.dumps(sorted(observed))}\n"
                        f"Remaining new-frame budget: {12 - len(observed)}\n"
                        + "\n".join(
                            f"{string.ascii_uppercase[i]}: {json.dumps(c)}"
                            for i, c in enumerate(menu)
                        )
                    )
                    index = scorer.choose(
                        alias,
                        displayed,
                        question,
                        len(menu),
                        f"selector_{mode}_seed{seed}_turn{turn}",
                    )
                    latest = menu[index]["indices"]
                    assert len(set(latest) - set(observed)) == 3
                    observed.extend(latest)
                assert len(set(observed)) == 12
                sets[f"qwen_{mode}"] = sorted(observed)
            sets["broader_uniform24"] = stratified(24)
            # Privileged diagnostic only. Selecting the set uses labels; the reader sees pixels only.
            sets["privileged12"] = sorted(sorted(range(840), key=lambda i: (-quality[i], i))[:12])
            for method, indices in sets.items():
                pick = scorer.choose(
                    alias, indices, reader_question(), len(indices), f"reader_{method}_seed{seed}"
                )
                selected = indices[pick]
                results.append(
                    {
                        "alias": alias,
                        "subject_id": case["subject_id"],
                        "seed": seed,
                        "method": method,
                        "indices": indices,
                        "selected_frame": selected,
                        "wfss": float(quality[selected]),
                        "any_annotated": bool(np.any(quality[indices] > 0)),
                        "any_optimal": bool(np.any(private["labels"][indices] == 1)),
                        "best_available_wfss": float(quality[indices].max()),
                    }
                )
                write_json(args.output / "results.json", results)
        positives = np.flatnonzero(quality == quality.max()) if quality.max() > 0 else []
        if len(positives):
            positive = int(positives[len(positives) // 2])
            negative = random.Random(142).sample(np.flatnonzero(quality == 0).tolist(), 3)
            indices = [positive, *negative]
            random.Random(142).shuffle(indices)
            for order, shown in [("original", indices), ("reversed", list(reversed(indices)))]:
                pick = scorer.choose(
                    alias, shown, reader_question(), 4, f"recognition_control_{order}"
                )
                controls.append(
                    {
                        "alias": alias,
                        "order": order,
                        "indices": shown,
                        "selected_frame": shown[pick],
                        "wfss": float(quality[shown[pick]]),
                        "picked_supported_plane": shown[pick] == positive,
                    }
                )
            write_json(args.output / "recognition_controls.json", controls)
        print(f"Completed {alias}; {scorer.calls} frozen forward calls", flush=True)
    after = parameter_digest(policy.model, lambda n, p: True)
    assert before == after and not any(p.requires_grad for p in policy.model.parameters())
    write_json(
        args.output / "report.json",
        {
            "status": "complete",
            "frozen_reader_unchanged": True,
            "reader_parameter_sha256": after,
            "model_calls": scorer.calls,
            "training": False,
            "breastvid_test_opened": False,
            "cases": 5,
            "primary_budget": 12,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/navigation_crc.yaml"))
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
