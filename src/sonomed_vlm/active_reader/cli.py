"""Audited-data -> private replay -> frozen baselines -> small policy-gradient run."""

import argparse
import json
import os
from pathlib import Path

from sonomed_vlm.active_reader.config import load_config
from sonomed_vlm.active_reader.data.audit import audit_data, contact_sheet, write_audit
from sonomed_vlm.active_reader.data.manifest import fingerprint, load_manifest, save_manifest
from sonomed_vlm.active_reader.data.splits import attach_splits, make_splits
from sonomed_vlm.active_reader.env.replay import Replay
from sonomed_vlm.active_reader.io import (
    code_revision,
    digest,
    file_hash,
    read_json,
    versions,
    write_json,
)
from sonomed_vlm.active_reader.models.prompts import PROMPT_HASH


def data_root(config):
    if not config.data.root or not Path(config.data.root).is_dir():
        raise ValueError("Set BREASTVID_ROOT/data.root to the extracted miccai_buv directory")
    return Path(config.data.root).expanduser().resolve()


def audited_cases(config):
    cases = load_manifest(config.data.manifest)
    audit = read_json(config.data.audit)
    if not audit["automatic_checks_passed"] or audit["manifest_fingerprint"] != fingerprint(cases):
        raise ValueError("Data audit is unresolved or stale; rerun audit-data and prepare-manifest")
    # Each revealed frame is hash-verified inside Replay before it reaches the model.
    # Re-reading the entire corpus before a single-case smoke test is unnecessary.
    data_root(config)
    return attach_splits(cases, read_json(config.data.splits)), audit


def choose_cases(cases, split, limit=None):
    selected = [c for c in cases if c.split == split]
    if limit is None:
        return selected
    if limit < 1:
        raise ValueError("Case limit must be positive")
    # A feasibility subset should exercise both labels where possible. Full evaluation
    # always uses every case in its selected split.
    buckets = [[c for c in selected if c.label == label] for label in ("benign", "malignant")]
    interleaved = [
        bucket[i]
        for i in range(max(map(len, buckets), default=0))
        for bucket in buckets
        if i < len(bucket)
    ]
    return interleaved[:limit]


def smoke_environment(config, output):
    import tempfile

    from PIL import Image

    from sonomed_vlm.active_reader.data.manifest import Case, Frame
    from sonomed_vlm.active_reader.env.actions import Inspect, Stop

    with tempfile.TemporaryDirectory(prefix="active-reader-smoke-") as temporary:
        root = Path(temporary)
        frames = []
        for i in range(40):
            Image.new("RGB", (16, 16), (i * 5, i, 100)).save(root / f"{i}.png")
            frames.append(Frame(i, str(i), f"{i}.png", 16, 16, "present", [[0, 0, 2 + i / 4, 10]]))
        case = Case(
            "synthetic", "private/malignant/test", "malignant", frames, group_id="synthetic"
        )
        env = Replay(case, root, config.environment, config.reward)
        observation = env.reset(config.seed)
        initial = list(observation.indices)
        action = next(
            (
                Inspect(b, s)
                for b, s in observation.legal_inspections
                if set(
                    __import__(
                        "sonomed_vlm.active_reader.env.actions", fromlist=["window_indices"]
                    ).window_indices(40, b, s)
                )
                - set(initial)
            ),
            None,
        )
        if action:
            observation, _, _, _ = env.step(action)
        observation, _, done, info = env.step(Stop("malignant", (observation.indices[0],)))
        assert done and info["prediction"] == "malignant"
        write_json(
            output,
            {
                "status": "passed",
                "fixture": "synthetic software test; not BreastVid results",
                "initial_indices": initial,
                "final_indices": observation.indices,
                "private_components": info["reward_components"],
                "versions": versions(),
            },
        )


def smoke_model(config, cases, output):
    from sonomed_vlm.active_reader.env.actions import Inspect, parse_action
    from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory

    policy = QwenPolicy(config.model)
    case = next(c for c in cases if c.split == "train" and c.num_frames >= 8)
    env = Replay(case, data_root(config), config.environment, config.reward)
    observation = env.reset(config.seed, max(12, config.environment.max_unique_frames))
    from dataclasses import replace

    single = replace(
        observation,
        indices=observation.indices[:1],
        images=observation.images[:1],
        returned_indices=observation.indices[:1],
        legal_inspections=(),
    )
    results = [policy.generate(single, final_only=True).record]
    inv = inventory(config.model)
    write_json(output, {"status": "in_progress", "inventory": inv, "generations": results})
    try:
        single_action = parse_action(results[0]["raw_output"])
        if single_action.action != "stop" or not set(single_action.evidence_frames) <= set(
            single.indices
        ):
            raise ValueError("Single-frame answer did not cite an observed frame")
    except ValueError as exc:
        write_json(
            output,
            {"status": "failed", "inventory": inv, "generations": results, "reason": str(exc)},
        )
        raise ValueError(
            "Single-frame format gate failed; inspect the saved model-smoke report"
        ) from exc
    selected = []
    for _ in range(2):
        before = set(observation.indices)
        from sonomed_vlm.active_reader.env.actions import window_indices

        candidate = next(
            (
                (b, s)
                for b, s in observation.legal_inspections
                if set(window_indices(case.num_frames, b, s)) - before
            ),
            None,
        )
        if candidate is None:
            raise ValueError("Smoke configuration needs two distinct inspection turns")
        observation, _, _, _ = env.step(Inspect(*candidate))
        selected.append(list(observation.indices))
        result = policy.generate(observation)
        results.append(result.record)
        try:
            action = parse_action(result.text)
            if action.action == "stop" and not set(action.evidence_frames) <= set(
                observation.indices
            ):
                raise ValueError("Model cited unobserved evidence")
            if (
                action.action == "inspect"
                and (action.center_bin, action.scale) not in observation.legal_inspections
            ):
                raise ValueError("Model requested an illegal inspection")
        except ValueError as exc:
            write_json(
                output,
                {"status": "failed", "inventory": inv, "generations": results, "reason": str(exc)},
            )
            raise ValueError(
                "Multi-image action gate failed; inspect the saved model-smoke report"
            ) from exc
    inv["supports_ordered_multi_image_input"] = True
    write_json(
        output,
        {
            "status": "passed",
            "inventory": inv,
            "two_distinct_contexts": selected,
            "generations": results,
            "note": "Frozen inference gate only; RL probability/gradient gate runs during train-rl",
        },
    )


def run_baselines(config, cases, audit, args):
    from sonomed_vlm.active_reader.agents.fixed_sampling import METHODS
    from sonomed_vlm.active_reader.evaluation.baselines import fixed_read, run_case
    from sonomed_vlm.active_reader.evaluation.metrics import evaluate
    from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory

    inv = inventory(config.model)
    selected_cases = choose_cases(cases, args.split, args.limit)
    if not selected_cases:
        raise ValueError("Requested split contains no cases")
    if args.split == "test":
        if not args.open_test or not args.development_report:
            raise ValueError(
                "Test is sealed. Select a checkpoint on development data, then pass --open-test --development-report."
            )
        dev = read_json(args.development_report)
        if dev.get("split") != "dev" or dev.get("inventory") != inv:
            raise ValueError("Development report must be for this exact selected model/processor")
    methods = args.methods or config.evaluation.methods
    if not set(methods) <= METHODS:
        raise ValueError("Unknown baseline method")
    run_dir = args.output or config.output / "baselines"
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "code_revision": code_revision(),
        "config": config.model_dump(mode="json"),
        "inventory": inv,
        "manifest": audit["manifest_fingerprint"],
        "splits": file_hash(config.data.splits),
        "prompt_template_hash": PROMPT_HASH,
        "versions": versions(),
        "split": args.split,
        "methods": methods,
        "limit": args.limit,
        "selection_source": file_hash(args.selection_from) if args.selection_from else None,
    }
    run_hash = digest(provenance)
    meta_path = run_dir / "provenance.json"
    if meta_path.exists() and read_json(meta_path) != provenance:
        raise ValueError("Output belongs to another experiment; choose a fresh --output directory")
    write_json(meta_path, provenance)
    policy, rows = None, []
    selections = read_json(args.selection_from) if args.selection_from else None
    if selections is not None and isinstance(selections, dict):
        selections = selections["trajectories"]
    for case in selected_cases:
        jobs = [
            (m, b, s, None)
            for m in methods
            for b in config.evaluation.budgets
            for s in config.evaluation.seeds
            if (m != "dense" or b == max(config.evaluation.budgets))
            and (
                m not in ("first_frame", "center_frame")
                or (b == min(config.evaluation.budgets) and s == config.evaluation.seeds[0])
            )
        ]
        if selections is not None:
            jobs = [
                (f"fixed_reader:{r['method']}", r["budget"], r["seed"], r["selected_indices"])
                for r in selections
                if r["video_id"] == case.video_id
            ]
        for method, budget, seed, indices in jobs:
            path = (
                run_dir
                / "episodes"
                / (digest([run_hash, case.video_id, method, budget, seed]) + ".json")
            )
            if path.exists():
                record = read_json(path)
            else:
                if policy is None:
                    policy = QwenPolicy(config.model)
                if indices is None:
                    record = run_case(case, data_root(config), policy, config, method, seed, budget)
                else:
                    record = fixed_read(
                        case,
                        data_root(config),
                        policy,
                        config,
                        method,
                        seed,
                        budget,
                        selected=indices,
                    )
                write_json(path, record)
                print(f"{method} budget={budget} completed={len(rows) + 1}", flush=True)
            rows.append(record)
    report = {
        **evaluate(rows, config.evaluation.bootstrap_samples, config.seed),
        "split": args.split,
        "inventory": inv,
        "data_fingerprint": audit["manifest_fingerprint"],
        "contamination": audit["contamination"],
        "visual_shortcut_review": audit["visual_shortcut_review"],
        "provenance_hash": run_hash,
    }
    write_json(run_dir / "trajectories.json", rows)
    write_json(run_dir / "report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("configs/active_reader.yaml"))
    sub = result.add_subparsers(dest="command", required=True)
    for name in (
        "audit-data",
        "prepare-manifest",
        "make-splits",
        "smoke-test-environment",
        "smoke-test-model",
        "inventory-model",
    ):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path)
        if name in ("audit-data", "prepare-manifest"):
            p.add_argument("--contact-sheets", action="store_true")
    for name in ("run-baselines", "collect-rollouts"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path)
        p.add_argument(
            "--split",
            choices=("train", "dev", "test"),
            default="dev" if name == "run-baselines" else "train",
        )
        p.add_argument("--methods", nargs="+")
        p.add_argument("--limit", type=int)
        p.add_argument("--open-test", action="store_true")
        p.add_argument("--development-report", type=Path)
        p.add_argument("--selection-from", type=Path)
    p = sub.add_parser("train-rl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-smoke-report", type=Path, required=True)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--no-relevance", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("navigation-campaign")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--evaluate-only", action="store_true")
    p = sub.add_parser("review-trajectories")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    os.umask(0o077)
    if args.command == "smoke-test-environment":
        smoke_environment(config, args.output or config.output / "environment_smoke.json")
    elif args.command == "navigation-campaign":
        from sonomed_vlm.active_reader.navigation.campaign import run

        if args.steps < 1:
            raise ValueError("Navigation pilot requires at least one update")
        run(config, args.output, args.steps, args.limit, args.evaluate_only)
    elif args.command in ("audit-data", "prepare-manifest"):
        cases, report = audit_data(
            data_root(config), config.data.annotation_files, config.data.sft_manifests
        )
        write_audit(args.output or config.data.audit, report)
        write_json(config.data.audit.parent / "corrections.json", report["corrections"])
        write_json(
            config.data.audit.parent / "quarantine.json",
            {"quarantine": report["quarantine"], "label_conflicts": report["label_conflicts"]},
        )
        if args.contact_sheets:
            for case in cases:
                if case.frames:
                    contact_sheet(
                        case,
                        data_root(config),
                        config.data.audit.parent / "human_audit_only" / f"{case.video_id}.png",
                    )
        if not report["automatic_checks_passed"]:
            raise ValueError(
                "Data audit found unresolved issues; inspect the private audit report before creating a manifest"
            )
        if args.command == "prepare-manifest":
            save_manifest(config.data.manifest, cases)
        print(
            json.dumps(
                {
                    "sequences": report["corrected_sequences"],
                    "frames": report["corrected_frames"],
                    "class_balance": report["class_balance"],
                }
            )
        )
    elif args.command == "make-splits":
        cases = load_manifest(config.data.manifest)
        audit = read_json(config.data.audit)
        if not audit["automatic_checks_passed"] or audit["manifest_fingerprint"] != fingerprint(
            cases
        ):
            raise ValueError("Manifest lacks a matching successful audit")
        write_json(args.output or config.data.splits, make_splits(cases, config.seed))
    elif args.command == "inventory-model":
        from sonomed_vlm.active_reader.models.qwen3_adapter import inventory

        write_json(args.output or config.output / "model_inventory.json", inventory(config.model))
    elif args.command == "evaluate":
        from sonomed_vlm.active_reader.evaluation.metrics import evaluate

        write_json(
            args.output,
            evaluate(read_json(args.predictions), config.evaluation.bootstrap_samples, config.seed),
        )
    elif args.command in (
        "smoke-test-model",
        "run-baselines",
        "collect-rollouts",
        "train-rl",
        "review-trajectories",
    ):
        cases, audit = audited_cases(config)
        if args.command == "smoke-test-model":
            smoke_model(config, cases, args.output or config.output / "model_smoke.json")
        elif args.command == "run-baselines":
            run_baselines(config, cases, audit, args)
        elif args.command == "collect-rollouts":
            from sonomed_vlm.active_reader.env.rewards import class_weights
            from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory
            from sonomed_vlm.active_reader.training.rollouts import grouped_rollouts

            if args.split != "train":
                raise ValueError("Training rollout collection uses train cases only")
            selected = [c for c in cases if c.split == "train"]
            weights = class_weights(selected, config.reward.class_balance)
            policy = QwenPolicy(config.model)
            directory = args.output or config.output / "rollouts"
            provenance = {
                "code_revision": code_revision(),
                "model": inventory(config.model),
                "data": audit["manifest_fingerprint"],
                "config": config.model_dump(mode="json"),
            }
            if (directory / "provenance.json").exists() and read_json(
                directory / "provenance.json"
            ) != provenance:
                raise ValueError("Rollout directory belongs to another experiment")
            write_json(directory / "provenance.json", provenance)
            for i, case in enumerate(selected[: args.limit] if args.limit else selected):
                destination = directory / f"{case.video_id}.json"
                if destination.exists():
                    continue
                group = grouped_rollouts(
                    lambda case=case: Replay(
                        case, data_root(config), config.environment, config.reward, weights
                    ),
                    policy,
                    config.seed + i,
                    config.seed + i * 100003,
                    config.environment.max_unique_frames,
                    config.training.group_size,
                )
                write_json(destination, [t.record for t in group])
        elif args.command == "train-rl":
            from sonomed_vlm.active_reader.models.qwen3_adapter import QwenPolicy, inventory
            from sonomed_vlm.active_reader.training.grpo import train

            smoke, baseline = read_json(args.model_smoke_report), read_json(args.baseline_report)
            inv = inventory(config.model)
            if (
                smoke.get("status") != "passed"
                or baseline.get("split") != "dev"
                or baseline.get("data_fingerprint") != audit["manifest_fingerprint"]
                or baseline.get("inventory") != inv
            ):
                raise ValueError(
                    "RL requires matching real-model smoke and development baseline reports"
                )
            tested = dict(smoke.get("inventory", {}))
            tested["supports_ordered_multi_image_input"] = None
            if tested != inv:
                raise ValueError("Model smoke report is for a different checkpoint/processor")
            if args.no_relevance:
                config.reward.relevance_weight = 0.0
            if args.output.exists() and any(args.output.iterdir()) and not args.resume:
                raise ValueError(
                    "RL output already exists; resume explicitly or choose a fresh directory"
                )
            train(
                QwenPolicy(config.model, for_training=True),
                [c for c in cases if c.split == "train"],
                data_root(config),
                config,
                args.output,
                {
                    "code_revision": code_revision(),
                    "inventory": inv,
                    "audit": audit["manifest_fingerprint"],
                    "splits": file_hash(config.data.splits),
                },
                args.resume,
            )
        else:
            by_id = {c.video_id: c for c in cases}
            rows = read_json(args.predictions)
            for i, row in enumerate(rows):
                case = by_id[row["video_id"]]
                contact_sheet(
                    case,
                    data_root(config),
                    args.output / f"trajectory-{i:05d}.png",
                    row["selected_indices"],
                )
            write_json(
                args.output / "index.json",
                [
                    {
                        k: r.get(k)
                        for k in ("video_id", "method", "prediction", "label", "selected_indices")
                    }
                    for r in rows
                ],
            )
    print(f"{args.command}: complete", flush=True)


if __name__ == "__main__":
    main()
