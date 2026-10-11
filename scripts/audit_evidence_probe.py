#!/usr/bin/env python3
"""Independently replay saved train-only branches and check native input boundaries."""

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from tokenizers import Tokenizer

from sonomed_vlm.active_reader.config import Config
from sonomed_vlm.active_reader.data.manifest import fingerprint, load_manifest
from sonomed_vlm.active_reader.data.splits import attach_splits
from sonomed_vlm.active_reader.io import file_hash, read_json, write_json
from sonomed_vlm.active_reader.models.constraints import ActionGrammar
from sonomed_vlm.active_reader.navigation.browser import coverage_action
from sonomed_vlm.active_reader.navigation.preference import observation_for
from sonomed_vlm.active_reader.navigation.probe import replay_state


def audit(args):
    provenance = read_json(args.directory / "provenance.json")
    config = Config.model_validate(provenance["config"])
    manifest = load_manifest(args.manifest)
    assert fingerprint(manifest) == provenance["data"], "Manifest changed"
    assert file_hash(args.splits) == provenance["splits"], "Split assignments changed"
    cases = {c.video_id: c for c in attach_splits(manifest, read_json(args.splits))}
    states = read_json(args.directory / "probe/states.json")
    # Authoritative membership checks precede every image read.
    assert all(r["split"] == "train" and cases[r["video_id"]].split == "train" for r in states)
    expected = provenance["partition"]
    assert set(r["video_id"] for r in states if r["role"] == "fit") == set(expected["fit"])
    assert set(r["video_id"] for r in states if r["role"] == "validation") == set(
        expected["validation"]
    )
    assert {cases[v].group_id for v in expected["fit"]}.isdisjoint(
        cases[v].group_id for v in expected["validation"]
    )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    class Tokens:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [tokenizer.encode(t, add_special_tokens=False).ids for t in texts]}

    counts = Counter()
    reader_text = {}
    same_evidence = {}
    by_id = {r["state_id"]: r for r in states}
    contexts = {}

    def check_reader(record, obs):
        hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in obs.images]
        assert hashes == record["clean_image_sha256"], "Reader pixels differ from source"
        assert list(obs.indices) == record["selected_indices"]
        assert len(record["image_grid_thw"]) == len(obs.images)
        text = tokenizer.decode(record["input_ids"], skip_special_tokens=True)
        n = len(obs.images)
        assert reader_text.setdefault(n, text) == text, "Reader text differs at matched count"
        key = tuple(zip(record["selected_indices"], hashes, strict=True))
        evidence = (record["input_ids"], record["processed_pixels_sha256"], record["probabilities"])
        assert same_evidence.setdefault(key, evidence) == evidence
        counts["reader_inputs"] += 1

    for row in states:
        case = cases[row["video_id"]]
        assert row["group_id"] == case.group_id
        b = replay_state(case, args.data_root, config, row["seed"], 12, row["prefix"])
        assert list(b.observation().indices) == row["observed_indices"]
        assert row["actions"][row["coverage_index"]] == coverage_action(b)
        check_reader(row["initial_reader"], b.observation(reader=True))
        assert len(row["actions"]) == len(row["outcomes"])
        for action, outcome in zip(row["actions"], row["outcomes"], strict=True):
            assert action == outcome["action"]
            branch = replay_state(case, args.data_root, config, row["seed"], 12, row["prefix"])
            obs = branch.step(action)
            assert len(obs.returned_indices) == 4
            for follow in outcome["completion"]:
                assert follow == coverage_action(branch), "Completion changed between branches"
                branch.step(follow)
            assert branch.done and len(branch.env.visited) == 12
            assert outcome["selected_indices"] == list(branch.observation().indices)
            check_reader(outcome["reader"], branch.observation(reader=True))
            assert (
                abs(
                    outcome["log_score"]
                    - math.log(max(1e-8, outcome["reader"]["probabilities"][case.label]))
                )
                < 1e-7
            )
            counts["counterfactual_branches"] += 1
        counts["states"] += 1

    def check_native(row, native, mode):
        key = (row["state_id"], mode)
        if key not in contexts:
            obs = observation_for(row, cases, args.data_root, config, mode)
            hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in obs.images]
            contexts[key] = (obs, hashes)
        obs, hashes = contexts[key]
        # JSON normalizes tuples into lists in the stored public context.
        import json

        assert native["context"] == json.loads(json.dumps(obs.public_dict()))
        assert native["clean_or_counterfactual_image_sha256"] == hashes
        assert len(native["image_grid_thw"]) == len(obs.images)
        assert len(native["prompt_loss_mask"]) == len(native["input_ids"])
        assert not any(native["prompt_loss_mask"])
        assert len(native["candidate_tokens"]) == len(row["actions"])
        eos = native["candidate_tokens"][0][-1]
        grammar = ActionGrammar(obs, Tokens(), eos)
        for ids, support, mask, action in zip(
            native["candidate_tokens"],
            native["candidate_token_supports"],
            native["candidate_loss_masks"],
            row["actions"],
            strict=True,
        ):
            assert grammar.constraints_for(ids) == support
            assert mask == [1] * len(ids)
            assert json.loads(tokenizer.decode(ids, skip_special_tokens=True)) == action
        actual_text = tokenizer.decode(native["input_ids"], skip_special_tokens=True)
        assert row["video_id"] not in actual_text
        counts["native_selector_inputs"] += 1

    for arm in ("full", "current_frame"):
        for path in sorted((args.directory / arm).rglob("*.json")):
            if path.parent.name in ("policy", "processor"):
                continue
            record = read_json(path)
            if "native_inputs" not in record:
                continue
            row = by_id[record["state_id"]]
            mode = record.get("mode", arm)
            check_native(row, record["native_inputs"], mode)
            if "selected_candidate" in record:
                i = record["selected_candidate"]
                assert record["selected_indices"] == row["outcomes"][i]["selected_indices"]
                assert abs(sum(math.exp(x) for x in record["log_probabilities"]) - 1) < 0.001
                counts["ranked_states"] += 1
            else:
                assert record["role"] == row["role"] == "fit"
                assert record["split"] == "train"
                chosen, rejected = record["chosen"], record["rejected"]
                assert (
                    row["outcomes"][chosen]["log_score"] - row["outcomes"][rejected]["log_score"]
                    >= 0.1
                )
                counts["fit_updates"] += 1
    return {
        "status": "passed",
        "checks": dict(counts),
        "reader_text_counts": sorted(reader_text),
        "test_opened": False,
        "scope": "source pixels, clean reader inputs, train partitions, matched branches, constrained supports, and fit-only losses; no efficacy claim",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args)
    write_json(args.output, result)
    print(result)
