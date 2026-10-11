"""Connected patient/duplicate groups and deterministic stratified allocation."""

import random
from collections import Counter, defaultdict

from sonomed_vlm.active_reader.data.manifest import fingerprint
from sonomed_vlm.active_reader.io import digest


def connect_groups(cases):
    parent = {c.video_id: c.video_id for c in cases}

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        parent[find(b)] = find(a)

    seen, overlap = {}, []
    for case in cases:
        keys = [("group", case.group_id)]
        if case.patient_id:
            keys.append(("patient", case.patient_id))
        keys += [("pixels", f.pixel_sha256) for f in case.frames if f.pixel_sha256]
        for key in keys:
            if key in seen and seen[key] != case.video_id:
                join(case.video_id, seen[key])
                if key[0] == "pixels":
                    overlap.append(sorted([case.video_id, seen[key]]))
            else:
                seen[key] = case.video_id
    components = defaultdict(list)
    for case in cases:
        components[find(case.video_id)].append(case)
    for group in components.values():
        gid = "group_" + digest(sorted(c.video_id for c in group))[:16]
        basis = "+".join(
            sorted(
                {c.group_basis for c in group}
                | (
                    {"exact_duplicate"}
                    if any(set(pair) <= {c.video_id for c in group} for pair in overlap)
                    else set()
                )
            )
        )
        for case in group:
            case.group_id, case.group_basis = gid, basis
    return [list(x) for x in sorted({tuple(x) for x in overlap})]


def make_splits(cases, seed=42, ratios=(0.70, 0.15, 0.15)):
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Need three positive train/dev/test fractions summing to one")
    groups = defaultdict(list)
    for case in cases:
        case.validate()
        groups[case.group_id].append(case)
    if len(groups) < 3:
        raise ValueError("Need at least three independent groups for train/dev/test")
    totals = Counter(c.label for c in cases)
    if set(totals) != {"benign", "malignant"}:
        raise ValueError("Both corrected classes are required")
    names = ("train", "dev", "test")
    targets = {
        s: {label: totals[label] * ratio for label in totals}
        for s, ratio in zip(names, ratios, strict=True)
    }
    # Seeded restarts minimize class-count deviation while retaining entire components.
    rng, best = random.Random(seed), None
    for _ in range(128):
        order = list(groups)
        rng.shuffle(order)
        order.sort(key=lambda g: -len(groups[g]))
        counts, allocation = {s: Counter() for s in names}, {}
        for i, gid in enumerate(order):
            vector = Counter(c.label for c in groups[gid])
            empty = [s for s in names if s not in allocation.values()]
            candidates = empty if len(order) - i == len(empty) else list(names)
            rng.shuffle(candidates)

            def objective(s, counts=counts, vector=vector):
                return sum(
                    ((counts[t][label] + (vector[label] if t == s else 0) - targets[t][label]) ** 2)
                    / max(targets[t][label], 1)
                    for t in names
                    for label in totals
                )

            chosen = min(candidates, key=objective)
            allocation[gid] = chosen
            counts[chosen].update(vector)
        score = sum(
            (counts[s][label] - targets[s][label]) ** 2 / max(targets[s][label], 1)
            for s in names
            for label in totals
        )
        if best is None or score < best[0]:
            best = score, allocation, counts
    _, allocation, counts = best
    result = {
        "seed": seed,
        "ratios": ratios,
        "manifest_fingerprint": fingerprint(cases),
        "groups": allocation,
        "cases": {c.video_id: allocation[c.group_id] for c in cases},
        "counts": {s: dict(counts[s]) for s in names},
        "independence": "group-disjoint; residual patient-overlap risk where IDs are missing",
        "warnings": [
            f"{s} lacks {label}" for s in names for label in totals if not counts[s][label]
        ],
    }
    return result


def attach_splits(cases, split_manifest):
    if fingerprint(cases) != split_manifest["manifest_fingerprint"]:
        raise ValueError("Manifest changed after split creation; regenerate and re-audit splits")
    for case in cases:
        case.split = split_manifest["cases"][case.video_id]
        if case.split != split_manifest["groups"][case.group_id]:
            raise ValueError("Split leaks a connected group")
    return cases
