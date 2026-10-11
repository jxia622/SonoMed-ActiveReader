#!/usr/bin/env python3
"""Independently verify frozen-reader audit records and seal boundaries."""

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from PIL import Image
from tokenizers import Tokenizer

from sonomed_vlm.active_reader.config import Config
from sonomed_vlm.active_reader.data.manifest import source_path
from sonomed_vlm.active_reader.io import digest, file_hash, read_json, write_json
from sonomed_vlm.active_reader.navigation.browser import NavigationBrowser, coverage_action
from sonomed_vlm.active_reader.navigation.probe import candidate_menu
from sonomed_vlm.active_reader.suitability import allowed_cases, development_plan


def audit(args):
    provenance = read_json(args.directory / "provenance.json")
    assert provenance["manifest_sha256"] == file_hash(args.manifest)
    assert provenance["splits_sha256"] == file_hash(args.splits)
    cases, _ = allowed_cases(args.manifest, args.splits)
    cases = {c.video_id: c for c in cases}
    tokenizer = Tokenizer.from_file(str(args.directory / "processor/tokenizer.json"))
    text_by_count = {}
    pixels = {}
    counts = Counter()
    reads = {}
    for path in sorted((args.directory / "reads").glob("*.json")):
        row = read_json(path)
        assert row["video_id"] in cases, "Sealed/unadmitted case in reads; refusing pixel access"
        case = cases[row["video_id"]]
        assert row["split"] == case.split and row["group_id"] == case.group_id
        if provenance["phase"] == "development":
            assert case.split == "dev"
        elif provenance["phase"] == "history":
            assert case.split == "train"
        indices = row["indices"]
        assert indices == sorted(set(indices))
        assert path.stem == digest([case.video_id, indices])
        hashes = []
        for i in indices:
            key = (case.video_id, i)
            if key not in pixels:
                with Image.open(source_path(args.data_root, case.frames[i].relative_path)) as im:
                    pixels[key] = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
            hashes.append(pixels[key])
        r = row["reader"]
        assert r["selected_indices"] == indices and r["clean_image_sha256"] == hashes
        assert len(r["image_grid_thw"]) == len(indices)
        text = tokenizer.decode(r["input_ids"], skip_special_tokens=True)
        assert text_by_count.setdefault(len(indices), text) == text
        assert all(
            k not in text
            for k in (
                case.video_id,
                "boxes_xyxy",
                "observed_indices",
                "remaining_frames",
                "reward_components",
            )
        )
        assert r["prediction"] == max(r["probabilities"], key=r["probabilities"].get)
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
        reads[path.stem] = row
        counts["reader_inputs"] += 1
    if (args.directory / "development.json").exists():
        dev = read_json(args.directory / "development.json")
        assert {r["video_id"] for r in dev} == {v for v, c in cases.items() if c.split == "dev"}
        for r in dev:
            c = cases[r["video_id"]]
            assert c.split == r["split"] == "dev" and c.label == r["label"]
            plan = development_plan(c)
            assert len(plan) == len(r["evaluations"])
            for expected, e in zip(plan, r["evaluations"], strict=True):
                assert all(e[k] == v for k, v in expected.items())
                source = reads[e["read_key"]]
                assert source["video_id"] == c.video_id and source["indices"] == e["indices"]
                assert source["reader"]["probabilities"] == e["probabilities"]
                assert source["reader"]["prediction"] == e["prediction"]
                counts["development_evaluations"] += 1
            counts["development_cases"] += 1
    if (args.directory / "history.json").exists():
        history = read_json(args.directory / "history.json")
        roles = read_json(args.directory / "history_partition.json")
        assert len(history) == 2 * sum(len(v) for v in roles.values())
        assert {(r["video_id"], r["seed"]) for r in history} == {
            (v, seed) for members in roles.values() for v in members for seed in (7, 19)
        }
        assert {cases[v].group_id for v in roles["fit"]}.isdisjoint(
            cases[v].group_id for v in roles["validation"]
        )
        config = Config.model_validate(provenance["config"])
        for row in history:
            c = cases[row["video_id"]]
            assert c.split == row["split"] == "train"
            assert c.video_id in roles[row["role"]]
            b = NavigationBrowser(c, args.data_root, config)
            b.reset(row["seed"], 12)
            assert row["prefix"] == coverage_action(b)
            b.step(row["prefix"])
            assert row["context"]["observed_indices"] == list(b.observation().indices)
            assert row["context"]["returned_indices"] == list(b.observation().returned_indices)
            assert row["context"]["remaining_frames"] == 4
            menu = candidate_menu(b, 8, 42 + row["seed"])
            assert menu == [o["action"] for o in row["outcomes"]]
            for o in row["outcomes"]:
                b2 = NavigationBrowser(c, args.data_root, config)
                b2.reset(row["seed"], 12)
                b2.step(row["prefix"])
                b2.step(o["action"])
                assert b2.done and len(b2.env.visited) == 12
                assert o["indices"] == sorted(b2.env.visited)
                assert o["new_indices"] == sorted(
                    b2.env.visited - set(row["context"]["observed_indices"])
                )
                saved = reads[o["read_key"]]
                assert saved["video_id"] == c.video_id and saved["indices"] == o["indices"]
                assert (
                    abs(
                        o["log_score"]
                        - math.log(max(1e-8, saved["reader"]["probabilities"][c.label]))
                    )
                    < 1e-8
                )
                assert o["correct"] == (saved["reader"]["prediction"] == c.label)
                counts["history_branches"] += 1
            counts["history_states"] += 1
    report = read_json(args.directory / "report.json")
    assert report["status"] == "complete" and report["frozen_reader_unchanged"]
    return {
        "status": "passed",
        "counts": dict(counts),
        "reader_text_frame_counts": sorted(text_by_count),
        "source_frames_checked": len(pixels),
        "test_opened": False,
        "scope": "authoritative split gates before pixels; clean source inputs, identical reader text, exact budgets, candidate replay and saved scores",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/active_reader/manifest.jsonl"))
    p.add_argument("--splits", type=Path, default=Path("artifacts/active_reader/splits.json"))
    p.add_argument("--data-root", type=Path, default=Path("data/raw/breastvid/miccai_buv"))
    args = p.parse_args()
    result = audit(args)
    write_json(args.directory / "input_audit.json", result)
    print(result)
