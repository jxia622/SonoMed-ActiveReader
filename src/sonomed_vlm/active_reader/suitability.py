"""Evaluation-only BreastVid suitability audit; sealed cases never reach image I/O."""

import hashlib
import math
import random
from collections import Counter
from itertools import pairwise

import numpy as np
from PIL import Image

from sonomed_vlm.active_reader.data.manifest import load_manifest, source_path
from sonomed_vlm.active_reader.data.splits import attach_splits
from sonomed_vlm.active_reader.env.observation import Observation
from sonomed_vlm.active_reader.io import digest, file_hash, read_json, write_json
from sonomed_vlm.active_reader.navigation.browser import NavigationBrowser, coverage_action
from sonomed_vlm.active_reader.navigation.probe import candidate_menu, partition_cases


def allowed_cases(manifest, splits):
    """Read the existing inventory, then exclude sealed records before any analysis."""
    assignments = read_json(splits)
    cases = attach_splits(load_manifest(manifest), assignments)
    inventory = Counter(c.split for c in cases)
    return [c for c in cases if c.split in ("train", "dev")], dict(inventory)


def require_allowed(case):
    if case.split not in ("train", "dev"):
        raise ValueError("Suitability audit refuses sealed/test image or annotation access")


def uniform_indices(n, budget):
    if not 1 <= budget <= n:
        raise ValueError("Invalid exact frame budget")
    if budget == 1:
        return [n // 2]
    return np.rint(np.linspace(0, n - 1, budget)).astype(int).tolist()


def union_area(boxes):
    """Exact rectangle union, without double-counting multiple lesions."""
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0.0
    for left, right in pairwise(xs):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] <= left and b[2] >= right)
        height, end = 0.0, -math.inf
        for lo, hi in intervals:
            height += max(0, hi - max(end, lo))
            end = max(end, hi)
        total += (right - left) * height
    return total


def development_plan(case):
    require_allowed(case)
    if case.split != "dev":
        raise ValueError("Reader headroom comparison is development-only")
    n = case.num_frames
    plan = [{"method": "single", "candidate": i, "indices": [i]} for i in range(n)]
    for budget in (4, 8, 12):
        plan.append({"method": "uniform", "budget": budget, "indices": uniform_indices(n, budget)})
        for seed in (7, 19, 43):
            indices = sorted(random.Random(seed).sample(range(n), budget))
            plan.append({"method": "random", "budget": budget, "seed": seed, "indices": indices})
        for fraction in (0.1, 0.35, 0.65, 0.9):
            start = min(n - budget, max(0, round(fraction * (n - 1)) - budget // 2))
            plan.append(
                {
                    "method": "window",
                    "budget": budget,
                    "center": fraction,
                    "indices": list(range(start, start + budget)),
                }
            )
        # Explicit privileged annotation diagnostic. Unknown visibility ranks last.
        areas = [
            union_area(f.boxes_xyxy) / (f.width * f.height)
            if f.annotation_status == "present"
            else -1
            for f in case.frames
        ]
        top = sorted(sorted(range(n), key=lambda i: (-areas[i], i))[:budget])
        plan.append({"method": "privileged_area", "budget": budget, "indices": top})
    for start in range(0, n, 12):
        plan.append(
            {
                "method": "dense_chunk",
                "chunk": start // 12,
                "indices": list(range(start, min(n, start + 12))),
            }
        )
    return plan


class AuditReader:
    """Durable evaluation cache, with native inputs retained for independent checks."""

    def __init__(self, reader, root, output):
        self.reader, self.root, self.output = reader, root, output

    def score(self, case, indices):
        require_allowed(case)
        indices = sorted(indices)
        if (
            not indices
            or len(indices) != len(set(indices))
            or not 0 <= min(indices) <= max(indices) < case.num_frames
        ):
            raise ValueError("Invalid selected frame set")
        key = digest([case.video_id, indices])
        path = self.output / "reads" / f"{key}.json"
        if path.exists():
            saved = read_json(path)
            if (
                saved["video_id"] != case.video_id
                or saved["indices"] != indices
                or saved["split"] != case.split
            ):
                raise ValueError("Cached selection mismatch")
            return key, saved["reader"]
        images = []
        for i in indices:
            frame = case.frames[i]
            source = source_path(self.root, frame.relative_path)
            if frame.sha256 and file_hash(source) != frame.sha256:
                raise ValueError("Source pixels changed since the data audit")
            with Image.open(source) as im:
                rgb = im.convert("RGB")
                if rgb.size != (frame.width, frame.height):
                    raise ValueError("Source dimensions changed since the data audit")
                pixels = rgb.tobytes()
                pixel_hash = hashlib.sha256(
                    f"{frame.width}x{frame.height}:".encode() + pixels
                ).hexdigest()
                if frame.pixel_sha256 and pixel_hash != frame.pixel_sha256:
                    raise ValueError("Decoded source pixels changed since the data audit")
                # Preserve Replay's boundary: no filename or PNG/EXIF metadata.
                images.append(Image.frombytes("RGB", rgb.size, pixels))
        obs = Observation(indices=tuple(indices), images=tuple(images), protocol="reader")
        result = self.reader.score(obs)
        write_json(
            path,
            {
                "video_id": case.video_id,
                "group_id": case.group_id,
                "split": case.split,
                "indices": indices,
                "reader": result,
            },
        )
        # The durable cache avoids retaining thousands of native token arrays in RAM.
        self.reader.cache.clear()
        return key, result


def evaluate_development(cases, scorer, output):
    cases = [c for c in cases if c.split == "dev"]
    records = []
    for number, case in enumerate(cases, 1):
        path = output / "development" / f"{digest(case.video_id)}.json"
        rows = []
        for spec in development_plan(case):
            key, result = scorer.score(case, spec["indices"])
            rows.append(
                {
                    **spec,
                    "read_key": key,
                    "probabilities": result["probabilities"],
                    "prediction": result["prediction"],
                }
            )
        record = {
            "video_id": case.video_id,
            "group_id": case.group_id,
            "split": case.split,
            "label": case.label,
            "num_frames": case.num_frames,
            "evaluations": rows,
        }
        write_json(path, record)
        records.append(record)
        print(f"Development complete {number}/{len(cases)}: {len(rows)} planned reads", flush=True)
    write_json(output / "development.json", records)


def evaluate_history(cases, scorer, config, root, output, count=32):
    partition = partition_cases([c for c in cases if c.split == "train"], count=count, seed=142)
    write_json(
        output / "history_partition.json",
        {k: [c.video_id for c in v] for k, v in partition.items()},
    )
    rows = []
    for role, subset in partition.items():
        for number, case in enumerate(subset, 1):
            for seed in (7, 19):
                key = digest([case.video_id, seed, "history-suitability-v1"])
                path = output / "history" / f"{key}.json"
                if path.exists():
                    rows.append(read_json(path))
                    continue
                browser = NavigationBrowser(case, root, config)
                browser.reset(seed, 12)
                prefix = coverage_action(browser)
                browser.step(prefix)
                obs = browser.observation()
                assert len(obs.indices) == 8 and len(obs.returned_indices) == 4
                actions = candidate_menu(browser, 8, seed=42 + seed)
                outcomes = []
                for action in actions:
                    branch = NavigationBrowser(case, root, config)
                    branch.reset(seed, 12)
                    branch.step(prefix)
                    branch.step(action)
                    indices = sorted(branch.env.visited)
                    assert branch.done and len(indices) == 12
                    read_key, result = scorer.score(case, indices)
                    outcomes.append(
                        {
                            "action": action,
                            "new_indices": sorted(set(indices) - set(obs.indices)),
                            "indices": indices,
                            "read_key": read_key,
                            "log_score": math.log(max(1e-8, result["probabilities"][case.label])),
                            "correct": result["prediction"] == case.label,
                        }
                    )
                row = {
                    "state_id": key,
                    "video_id": case.video_id,
                    "group_id": case.group_id,
                    "split": case.split,
                    "role": role,
                    "seed": seed,
                    "label": case.label,
                    "prefix": prefix,
                    "context": obs.public_dict(),
                    "outcomes": outcomes,
                }
                write_json(path, row)
                rows.append(row)
            print(f"History values complete {role} {number}/{len(subset)}", flush=True)
    write_json(output / "history.json", rows)
