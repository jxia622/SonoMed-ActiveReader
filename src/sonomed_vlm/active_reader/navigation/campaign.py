"""Prespecified development comparisons -> matched navigation pilots -> history ablations."""

from pathlib import Path

import torch

from sonomed_vlm.active_reader.cli import audited_cases, choose_cases, data_root
from sonomed_vlm.active_reader.io import (
    code_revision,
    digest,
    file_hash,
    read_json,
    versions,
    write_json,
)
from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory
from sonomed_vlm.active_reader.navigation.evaluation import paired, run_evaluation
from sonomed_vlm.active_reader.navigation.prompts import READER_HASH, SELECTOR_HASH
from sonomed_vlm.active_reader.navigation.reader import FrozenReader
from sonomed_vlm.active_reader.navigation.training import reset_selector, train_navigation


def run(config, output, steps=32, limit=None, evaluate_only=False):
    if not config.model.constrain_actions:
        raise ValueError("Navigation milestone requires audited constrained decoding")
    cases, audit = audited_cases(config)
    dev = choose_cases(cases, "dev", limit)
    train = [c for c in cases if c.split == "train"]
    config = config.model_copy(deep=True)
    config.reward.relevance_weight = 0.0
    config.training.steps = steps
    # Full precision fixes the observed BF16 cached/teacher likelihood mismatch.
    # Disable TF32 approximations for selector sampling and likelihood reconstruction.
    torch.set_float32_matmul_precision("highest")
    selector_config = config.model.model_copy(update={"dtype": config.navigation.selector_dtype})
    root = data_root(config)
    provenance = {
        "code_revision": code_revision(),
        "reader": inventory(config.model),
        "selector_initial": inventory(selector_config),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "reader_prompt": READER_HASH,
        "selector_prompt": SELECTOR_HASH,
        "data": audit["manifest_fingerprint"],
        "splits": file_hash(config.data.splits),
        "versions": versions(),
        "config": config.model_dump(mode="json"),
        "limit": limit,
        "test_policy": "test is sealed; this command has no test option",
    }
    output = Path(output)
    meta = output / "campaign.json"
    if meta.exists() and read_json(meta) != provenance:
        raise ValueError("Campaign provenance differs; use a new output directory")
    write_json(meta, provenance)
    print("Loading separate frozen reader and selector", flush=True)
    reader = FrozenReader(QwenPolicy(config.model))
    selector = QwenPolicy(selector_config, for_training=not evaluate_only)
    pre = run_evaluation(selector, reader, dev, root, config, output / "before", provenance)
    if evaluate_only:
        return pre
    initial_rows = read_json(output / "before" / "trajectories.json")
    uniform = {
        (r["video_id"], r["budget"], r["seed"]): r for r in initial_rows if r["method"] == "uniform"
    }
    score_changes = [
        abs(
            r["reader_true_label_log_score"]
            - uniform[(r["video_id"], r["budget"], r["seed"])]["reader_true_label_log_score"]
        )
        for r in initial_rows
        if r["method"] != "uniform"
    ]
    variable = max(score_changes, default=0) > 1e-6
    write_json(
        output / "selection_gate.json",
        {
            "reader_scores_vary_by_selection": bool(variable),
            "comparisons_with_score_change": sum(x > 1e-6 for x in score_changes),
            "max_absolute_logscore_change": max(score_changes, default=0),
            "development_videos": len(dev),
            "test_opened": False,
        },
    )
    if not variable:
        raise ValueError(
            "Frozen reader has no measurable selection signal; inspect development report before RL"
        )
    results = {"before": "before/report.json", "arms": {}}
    for arm, history, box_weight in (
        ("full", "full", 0.0),
        ("current_frame", "current_frame", 0.0),
        ("full_box_auxiliary", "full", 0.1),
    ):
        arm_config = config.model_copy(deep=True)
        arm_config.reward.relevance_weight = box_weight
        training_dir = output / arm / "training"
        reset_selector(selector)
        if (training_dir / "training_report.json").exists():
            training = read_json(training_dir / "training_report.json")
            selector.load_policy(Path(training["checkpoint"]))
        else:
            checkpoints = sorted(training_dir.glob("step-*/state.json"))
            training = train_navigation(
                selector,
                reader,
                train,
                root,
                arm_config,
                training_dir,
                provenance,
                history,
                resume=checkpoints[-1].parent if checkpoints else None,
            )
        post_config = config.model_copy(deep=True)
        post_config.evaluation.budgets = [config.environment.max_unique_frames]
        post_provenance = {
            **provenance,
            "selector_training_arm": arm,
            "selector_weights": file_hash(
                Path(training["checkpoint"]) / "policy" / "adapter_model.safetensors"
            ),
        }
        run_evaluation(
            selector,
            reader,
            dev,
            root,
            post_config,
            output / arm / "after",
            post_provenance,
            methods=["uniform", "random", "full", "current_batch", "current_frame"],
        )
        # Same immutable reader + prompt => nonadaptive outcomes must be unchanged.
        before_fixed = {
            (r["video_id"], r["budget"], r["seed"], r["method"]): r["reader"]["probabilities"]
            for r in initial_rows
            if r["method"] in ("uniform", "random")
        }
        post_rows = read_json(output / arm / "after" / "trajectories.json")
        learning_rows = [
            {**r, "method": "before:" + r["method"]}
            for r in initial_rows
            if r["budget"] == config.environment.max_unique_frames
        ]
        learning_rows += [{**r, "method": "after:" + r["method"]} for r in post_rows]
        write_json(
            output / arm / "learning_comparison.json",
            paired(learning_rows, "before:" + history, config.evaluation.bootstrap_samples),
        )
        for row in post_rows:
            if row["method"] in ("uniform", "random"):
                if (
                    row["reader"]["probabilities"]
                    != before_fixed[(row["video_id"], row["budget"], row["seed"], row["method"])]
                ):
                    raise RuntimeError("Frozen-reader nonadaptive control changed")
        results["arms"][arm] = {
            "training": str(training_dir / "training_report.json"),
            "evaluation": str(output / arm / "after" / "report.json"),
        }
    results.update(test_opened=False, development_only=True, experiment_id=digest(provenance))
    write_json(output / "report.json", results)
    return results
