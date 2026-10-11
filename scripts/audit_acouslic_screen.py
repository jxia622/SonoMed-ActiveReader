#!/usr/bin/env python3
"""Audit the frozen public-training screen; emit anonymous aggregate results."""

import argparse
import hashlib
import json
import random
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from screen_acouslic_qwen import candidate_menu, initial, reader_question, stratified
from tokenizers import Tokenizer


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, output):
    samples, run = root / "samples/acouslic", root / "acouslic-screen-v1"
    manifest = read(samples / "screen_manifest.json")
    report, provenance = read(run / "report.json"), read(run / "provenance.json")
    assert report["status"] == "complete" and report["frozen_reader_unchanged"]
    assert not report["training"] and not report["breastvid_test_opened"]
    assert provenance["source_sha256"] == sha(Path(__file__).with_name("screen_acouslic_qwen.py"))
    assert provenance["manifest_sha256"] == sha(samples / "screen_manifest.json")
    assert len(manifest) == len({r["subject_id"] for r in manifest}) == 5
    assert all(r["split"] == "public_train_screening" for r in manifest)
    aliases = {r["alias"] for r in manifest}
    labels = {a: np.load(samples / a / "private_labels.npz") for a in aliases}
    hashes = read(samples / "frame_hashes.json")
    tokenizer = Tokenizer.from_file(str(run / "processor/tokenizer.json"))
    records = [read(p) for p in sorted((run / "inputs").glob("*.json"))]
    assert len(records) == report["model_calls"] == 120
    assert [r["call"] for r in records] == list(range(120))
    cache, by_key = {}, {}
    for r in records:
        alias, indices = r["alias"], r["image_indices"]
        assert alias in aliases and len(set(indices)) == len(indices)
        assert len(indices) == len(r["clean_image_sha256"]) == len(r["image_grid_thw"])
        native = "<|im_start|>user\n"
        for j, (index, grid, digest) in enumerate(
            zip(indices, r["image_grid_thw"], r["clean_image_sha256"], strict=True)
        ):
            assert 0 <= index < 840
            key = (alias, index)
            if key not in cache:
                path = samples / alias / "frames" / f"{index:04d}.png"
                assert sha(path) == hashes[alias][str(index)]
                with Image.open(path) as im:
                    assert im.size == (744, 562) and not im.info
                    cache[key] = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
            assert digest == cache[key]
            assert grid == [1, 36, 46]
            native += f"Image {string.ascii_uppercase[j]}:<|vision_start|>"
            native += "<|image_pad|>" * (int(np.prod(grid)) // 4) + "<|vision_end|>"
        count = len(r["legal_answer_tokens"])
        answer_ids = [
            tokenizer.encode(x, add_special_tokens=False).ids
            for x in string.ascii_uppercase[:count]
        ]
        assert all(len(x) == 1 for x in answer_ids)
        assert [x[0] for x in answer_ids] == r["legal_answer_tokens"]
        native += r["question"] + "\nReply with exactly one letter from "
        native += ", ".join(string.ascii_uppercase[:count]) + ".<|im_end|>\n<|im_start|>assistant\n"
        assert tokenizer.encode(native, add_special_tokens=False).ids == r["input_ids"]
        p = np.array(r["probabilities"])
        assert p.shape == (count,) and np.isfinite(p).all() and np.all(p >= 0)
        assert np.isclose(p.sum(), 1) and r["selected"] == int(p.argmax())
        if not r["purpose"].startswith("selector_"):
            assert r["question"] == reader_question()
            assert count == len(indices)
        by_key[(alias, r["purpose"])] = r
    results = read(run / "results.json")
    assert len(results) == 70
    method_names = {
        "uniform",
        "random",
        "coverage",
        "qwen_current",
        "qwen_history",
        "broader_uniform24",
        "privileged12",
    }
    assert len({(r["alias"], r["seed"], r["method"]) for r in results}) == 70
    paths, histories = [], []
    for case in manifest:
        alias, q = case["alias"], labels[case["alias"]]["quality"]
        for seed in (7, 19):
            start = initial(seed)
            expected = {
                "uniform": stratified(12),
                "random": sorted(random.Random(seed).sample(range(840), 12)),
                "broader_uniform24": stratified(24),
                "privileged12": sorted(sorted(range(840), key=lambda i: (-q[i], i))[:12]),
            }
            coverage = start.copy()
            while len(coverage) < 12:
                coverage.append(
                    max(
                        (i for i in range(840) if i not in coverage),
                        key=lambda i: (min(abs(i - j) for j in coverage), -i),
                    )
                )
            expected["coverage"] = sorted(coverage)
            for mode in ("current", "history"):
                observed, latest = start.copy(), start.copy()
                for turn in range(2):
                    r = by_key[(alias, f"selector_{mode}_seed{seed}_turn{turn}")]
                    displayed = sorted(observed) if mode == "history" else latest
                    assert r["image_indices"] == displayed
                    menu = candidate_menu(observed, latest)
                    lines = r["question"].splitlines()
                    geometry = json.loads(
                        next(
                            x.split(": ", 1)[1]
                            for x in lines
                            if x.startswith("Displayed image locations:")
                        )
                    )
                    assert geometry == [
                        {
                            "letter": string.ascii_uppercase[j],
                            "sweep": i // 140 + 1,
                            "frame_in_sweep": i % 140,
                        }
                        for j, i in enumerate(displayed)
                    ]
                    assert (
                        f"All previously observed locations: {json.dumps(sorted(observed))}"
                        in lines
                    )
                    assert f"Remaining new-frame budget: {12 - len(observed)}" in lines
                    actual_menu = [json.loads(x[3:]) for x in lines if re.match(r"^[A-Z]: \{", x)]
                    assert actual_menu == menu and len(menu) == len(r["probabilities"])
                    latest = menu[r["selected"]]["indices"]
                    assert len(set(latest) - set(observed)) == 3
                    assert len({i // 140 for i in latest}) == 1
                    observed.extend(latest)
                expected[f"qwen_{mode}"] = sorted(observed)
            c0, h0 = [
                by_key[(alias, f"selector_{m}_seed{seed}_turn0")] for m in ("current", "history")
            ]
            assert (
                c0["input_ids"] == h0["input_ids"]
                and c0["clean_image_sha256"] == h0["clean_image_sha256"]
            )
            assert c0["selected"] == h0["selected"]
            histories.append(expected["qwen_current"] != expected["qwen_history"])
            oracle_scores = []
            for first in candidate_menu(start, start):
                for second in candidate_menu(start + first["indices"], first["indices"]):
                    ix = start + first["indices"] + second["indices"]
                    assert len(set(ix)) == 12
                    oracle_scores.append(float(q[ix].max()))
            paths.append(
                {
                    "case": alias,
                    "seed": seed,
                    "initial_best_wfss": float(q[start].max()),
                    "oracle_reachable_wfss": max(oracle_scores),
                    "positive_paths": sum(v > 0 for v in oracle_scores),
                    "paths": len(oracle_scores),
                }
            )
            rows = [r for r in results if r["alias"] == alias and r["seed"] == seed]
            assert {r["method"] for r in rows} == method_names
            for row in rows:
                assert (
                    row["subject_id"] == case["subject_id"]
                    and row["indices"] == expected[row["method"]]
                )
                r = by_key[(alias, f"reader_{row['method']}_seed{seed}")]
                assert r["image_indices"] == row["indices"]
                selected = row["indices"][r["selected"]]
                assert selected == row["selected_frame"] and row["wfss"] == float(q[selected])
                assert row["any_annotated"] == bool((q[row["indices"]] > 0).any())
                assert row["best_available_wfss"] == float(q[row["indices"]].max())
    controls = read(run / "recognition_controls.json")
    assert len(controls) == 10
    for r in controls:
        c = by_key[(r["alias"], f"recognition_control_{r['order']}")]
        assert c["image_indices"] == r["indices"]
        chosen = c["image_indices"][c["selected"]]
        assert r["selected_frame"] == chosen
        assert r["wfss"] == float(labels[r["alias"]]["quality"][chosen])
        assert r["picked_supported_plane"] == (r["wfss"] > 0)
    methods = []
    for method in sorted(method_names):
        rows = [r for r in results if r["method"] == method]
        methods.append(
            {
                "method": method,
                "subjects": 5,
                "case_start_pairs": 10,
                "frames": len(rows[0]["indices"]),
                "mean_final_wfss": float(np.mean([r["wfss"] for r in rows])),
                "annotated_capture_count": sum(r["any_annotated"] for r in rows),
                "mean_best_available_wfss": float(
                    np.mean([r["best_available_wfss"] for r in rows])
                ),
            }
        )
    summary = {
        "status": "passed",
        "cases": 5,
        "model_calls": 120,
        "unique_source_frames_verified": len(cache),
        "checks": [
            "frozen checkpoint and exact deployed source",
            "native token sequence and legal single-token decisions",
            "clean source file and RGB hashes",
            "identical reader question",
            "legal action replay and 12 unique frames",
            "current/history identical first state",
            "private label score recomputation",
        ],
        "methods": methods,
        "recognition_correct": sum(r["picked_supported_plane"] for r in controls),
        "recognition_presentations": 10,
        "recognition_correct_both_orders_subjects": sum(
            all(x["picked_supported_plane"] for x in controls if x["alias"] == a) for a in aliases
        ),
        "history_changed_final_set_pairs": sum(histories),
        "reachable_oracle": paths,
        "selector_action_counts": dict(
            Counter(
                string.ascii_uppercase[r["selected"]]
                for r in records
                if r["purpose"].startswith("selector")
            )
        ),
        "frozen_reader_parameter_sha256": report["reader_parameter_sha256"],
        "training": False,
        "breastvid_test_opened": False,
        "limitations": "Five development-exposed subjects. Two starts are not independent subjects. Restricted menu has poor reachability; unannotated does not imply anatomically unusable. No RL or clinical accuracy claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"reachable_oracle", "checks"}}, indent=2
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    audit(args.root, args.output)
