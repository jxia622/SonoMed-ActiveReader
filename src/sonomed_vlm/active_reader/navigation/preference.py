"""Small preference warm-start and fixed-state learning test; no diagnostic loss.

Each complete legal action is scored under the same finite candidate trie. The
pairwise objective uses fit-only frozen-reader preferences and a frozen original
ultrasound reference. This is preference training, not GRPO or clinical expertise.
"""

import hashlib
import json
import random
from statistics import mean

import torch
import torch.nn.functional as F

from sonomed_vlm.active_reader.io import digest, read_json, write_json
from sonomed_vlm.active_reader.models.constraints import ActionGrammar
from sonomed_vlm.active_reader.models.qwen3_adapter import Generation
from sonomed_vlm.active_reader.navigation.probe import context_observation, replay_state
from sonomed_vlm.active_reader.navigation.prompts import messages
from sonomed_vlm.active_reader.navigation.training import reset_selector
from sonomed_vlm.active_reader.training.grpo import parameter_digest


def generations(selector, obs):
    """Native inputs and complete paths, with identical public context for all actions."""
    inputs = dict(
        selector.processor.apply_chat_template(
            messages(obs),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    )
    inputs = {k: v.cpu() for k, v in inputs.items()}
    eos = selector.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    grammar = ActionGrammar(obs, selector.processor.tokenizer, eos)
    result = []
    for action in obs.legal_actions:
        text = json.dumps(action, separators=(",", ":"))
        ids = selector.processor.tokenizer(text, add_special_tokens=False)["input_ids"] + [eos]
        result.append(
            Generation(
                text, {"token_constraints": grammar.constraints_for(ids)}, inputs, torch.tensor(ids)
            )
        )
    audit = {
        "context": obs.public_dict(),
        "input_ids": inputs["input_ids"][0].tolist(),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "processed_pixels_sha256": hashlib.sha256(
            inputs["pixel_values"].contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "clean_or_counterfactual_image_sha256": [
            hashlib.sha256(im.tobytes()).hexdigest() for im in obs.images
        ],
        "candidate_tokens": [g.tokens.tolist() for g in result],
        "candidate_token_supports": [g.record["token_constraints"] for g in result],
        "prompt_loss_mask": [0] * inputs["input_ids"].shape[1],
        "candidate_loss_masks": [[1] * len(g.tokens) for g in result],
        "loss_mask_definition": "input prompt/image tokens excluded; complete action tokens only",
        "history_mode": obs.history_mode,
    }
    return result, audit


def observation_for(row, cases, root, config, mode):
    case = cases[row["video_id"]]
    if row["split"] != "train" or case.split != "train" or case.group_id != row["group_id"]:
        raise ValueError(
            "Probe scoring refuses nontraining or mismatched split/group before image access"
        )
    b = replay_state(case, root, config, row["seed"], row["budget"], row["prefix"])
    if list(b.observation().indices) != row["observed_indices"]:
        raise ValueError("Counterfactual state reconstruction changed")
    return context_observation(b, row["actions"], mode)


def score_states(selector, rows, cases, root, config, output, mode):
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "policy_weights": parameter_digest(selector.model, lambda n, p: ".policy." in n),
        "states": digest(rows),
        "mode": mode,
    }
    if (output / "provenance.json").exists() and read_json(
        output / "provenance.json"
    ) != provenance:
        raise ValueError("Saved ranking belongs to different weights or counterfactual states")
    write_json(output / "provenance.json", provenance)
    records = []
    for row in rows:
        path = output / f"{row['state_id']}.json"
        if path.exists():
            record = read_json(path)
            if record["state_id"] != row["state_id"] or record["mode"] != mode:
                raise ValueError("Saved ranking context differs")
        else:
            obs = observation_for(row, cases, root, config, mode)
            candidates, audit = generations(selector, obs)
            with torch.no_grad():
                logps = torch.stack([selector.log_probs(g).sum() for g in candidates])
            residual = float(logps.logsumexp(0))
            if not torch.isfinite(logps).all() or abs(residual) > 0.001:
                raise ValueError(
                    f"Complete constrained action probabilities do not sum to one: {residual}"
                )
            probs = logps.softmax(0).tolist()
            chosen = int(logps.argmax())
            scores = [x["log_score"] for x in row["outcomes"]]
            record = {
                "state_id": row["state_id"],
                "video_id": row["video_id"],
                "group_id": row["group_id"],
                "role": row["role"],
                "mode": mode,
                "observed_frames": len(row["observed_indices"]),
                "log_probabilities": logps.tolist(),
                "probability_sum_residual": residual,
                "selected_candidate": chosen,
                "selected_indices": row["outcomes"][chosen]["selected_indices"],
                "selected_log_score": scores[chosen],
                "expected_log_score": sum(p * q for p, q in zip(probs, scores, strict=True)),
                "best_log_score": max(scores),
                "coverage_log_score": scores[row["coverage_index"]],
                "uniform_menu_log_score": mean(scores),
                "selected_correct": row["outcomes"][chosen]["correct"],
                "best_candidate_selected": scores[chosen] >= max(scores) - 1e-6,
                "native_inputs": audit,
            }
            write_json(path, record)
        records.append(record)
        print(f"ranked {mode} {len(records)}/{len(rows)}", flush=True)
    write_json(output / "report.json", summarize_rankings(records))
    return records


def summarize_rankings(records):
    summary = {}
    for role in ("fit", "validation"):
        for depth in (4, 8, None):
            rows = [
                r
                for r in records
                if r["role"] == role and (depth is None or r["observed_frames"] == depth)
            ]
            if not rows:
                continue
            summary[f"{role}:{depth or 'all'}"] = {
                "cases": len({r["video_id"] for r in rows}),
                "states": len(rows),
                "mean_selected_log_score": mean(r["selected_log_score"] for r in rows),
                "mean_expected_log_score": mean(r["expected_log_score"] for r in rows),
                "mean_regret_to_menu_best": mean(
                    r["best_log_score"] - r["selected_log_score"] for r in rows
                ),
                "mean_gain_over_coverage": mean(
                    r["selected_log_score"] - r["coverage_log_score"] for r in rows
                ),
                "mean_gain_over_uniform_menu": mean(
                    r["selected_log_score"] - r["uniform_menu_log_score"] for r in rows
                ),
                "fraction_menu_best_selected": mean(r["best_candidate_selected"] for r in rows),
                "correct_fraction": mean(r["selected_correct"] for r in rows),
            }
    return {
        "summary": summary,
        "scope": "fixed states and restricted candidate menus; not an end-to-end deployed accuracy estimate",
        "test_opened": False,
    }


def learning_comparison(before, after):
    by = {r["state_id"]: r for r in before}
    result = {}
    for role in ("fit", "validation"):
        pairs = [(by[r["state_id"]], r) for r in after if r["role"] == role]
        result[role] = {
            "states": len(pairs),
            "changed_final_evidence_sets": sum(
                a["selected_indices"] != b["selected_indices"] for a, b in pairs
            ),
            "mean_selected_log_score_gain": mean(
                b["selected_log_score"] - a["selected_log_score"] for a, b in pairs
            ),
            "mean_expected_log_score_gain": mean(
                b["expected_log_score"] - a["expected_log_score"] for a, b in pairs
            ),
            "correct_fraction_gain": mean(
                b["selected_correct"] - a["selected_correct"] for a, b in pairs
            ),
        }
    return result


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta=1.0):
    margin = (chosen - rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin)


def train_preferences(
    selector,
    rows,
    initial_scores,
    cases,
    root,
    config,
    output,
    signature,
    mode,
    steps=64,
    learning_rate=1e-5,
    minimum_gap=0.1,
):
    """Repeated small-subset learning proof; validation rows can never enter loss."""
    if mode not in ("full", "current_frame"):
        raise ValueError("Only genuine observed contexts are trained")
    reset_selector(selector)
    initial = {r["state_id"]: r for r in initial_scores}
    examples = []
    for row in rows:
        if row["role"] != "fit":
            continue
        values = [x["log_score"] for x in row["outcomes"]]
        best, worst = (
            max(range(len(values)), key=values.__getitem__),
            min(range(len(values)), key=values.__getitem__),
        )
        if values[best] - values[worst] >= minimum_gap:
            examples.append((row, best, worst))
    if not examples:
        raise ValueError("No fit-only preference pairs exceed the prespecified gap")
    if any(initial[row["state_id"]]["mode"] != mode for row, _, _ in examples):
        raise ValueError("Preference reference probabilities use a different visual context")
    params = [(n, p) for n, p in selector.model.named_parameters() if p.requires_grad]
    if not params or any(
        ".policy." not in n or "lora_" not in n or "visual" in n for n, _ in params
    ):
        raise ValueError("Only language-side policy LoRA may train")
    names = {n for n, _ in params}

    def fixed(n, p):
        return "visual" in n or "vision" in n or ".reference." in n

    frozen_hash = parameter_digest(selector.model, fixed)
    policy_hash = parameter_digest(selector.model, lambda n, p: n in names)
    optimizer = torch.optim.AdamW([p for _, p in params], lr=learning_rate, weight_decay=0)
    checkpoints = sorted(output.glob("step-*/state.json"))
    start = 0
    if checkpoints:
        state = read_json(checkpoints[-1])
        if state["signature"] != signature or state["mode"] != mode:
            raise ValueError("Preference resume signature differs")
        selector.load_policy(checkpoints[-1].parent)
        optimizer.load_state_dict(
            torch.load(
                checkpoints[-1].parent / "optimizer.pt",
                weights_only=True,
                map_location=next(selector.model.parameters()).device,
            )
        )
        start = state["completed_steps"]
    prepared = {}
    for step in range(start, steps):
        order = list(range(len(examples)))
        random.Random(config.seed + step // len(examples)).shuffle(order)
        row, best, worst = examples[order[step % len(examples)]]
        state_id = row["state_id"]
        if state_id not in prepared:
            prepared[state_id] = generations(
                selector, observation_for(row, cases, root, config, mode)
            )
        candidate, audit = prepared[state_id]
        reference = initial[state_id]["log_probabilities"]
        optimizer.zero_grad(set_to_none=True)
        chosen = selector.log_probs(candidate[best]).sum()
        rejected = selector.log_probs(candidate[worst]).sum()
        loss = preference_loss(chosen, rejected, reference[best], reference[worst])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite preference objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite preference gradient")
        optimizer.step()
        write_json(
            output / "steps" / f"{step + 1:06d}.json",
            {
                "step": step + 1,
                "state_id": state_id,
                "role": row["role"],
                "split": row["split"],
                "chosen": best,
                "rejected": worst,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "chosen_log_probability": float(chosen.detach()),
                "rejected_log_probability": float(rejected.detach()),
                "reference_chosen": reference[best],
                "reference_rejected": reference[worst],
                "native_inputs": audit,
            },
        )
        if (step + 1) % 16 == 0 or step + 1 == steps:
            checkpoint = output / f"step-{step + 1:06d}"
            selector.save(checkpoint)
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
            write_json(
                checkpoint / "state.json",
                {"signature": signature, "mode": mode, "completed_steps": step + 1},
            )
        print(f"preference {mode} step={step + 1} loss={float(loss.detach()):.4f}", flush=True)
    final = output / f"step-{steps:06d}"
    final_hash = parameter_digest(selector.model, lambda n, p: n in names)
    if parameter_digest(selector.model, fixed) != frozen_hash:
        raise RuntimeError("Frozen vision/reference weights changed")
    selector.load_policy(final)
    if parameter_digest(selector.model, lambda n, p: n in names) != final_hash:
        raise RuntimeError("Preference checkpoint reload differs")
    report = {
        "mode": mode,
        "steps": steps,
        "fit_pairs": len(examples),
        "fit_cases": len({r["video_id"] for r, _, _ in examples}),
        "checkpoint": str(final),
        "language_adapter_changed": final_hash != policy_hash,
        "frozen_reference_and_vision_unchanged": True,
        "save_reload_identical": True,
        "objective": "pairwise complete-action preference relative to original ultrasound reference; beta=1",
        "learning_rate": learning_rate,
        "test_opened": False,
    }
    write_json(output / "report.json", report)
    return report
