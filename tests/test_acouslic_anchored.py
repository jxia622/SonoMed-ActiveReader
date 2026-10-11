"""Reachability safeguards for the development-only image-relative action probe."""

import importlib.util
import sys
from pathlib import Path

from sonomed_vlm.active_reader.acouslic.environment import initial, inspect, legal_centers


def test_relative_actions_keep_every_unseen_center_and_can_refine_an_old_region():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "anchored_probe", scripts / "probe_acouslic_anchored_actions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed in (7, 19, 42):
        first = initial(seed)
        # The most recent observation is in a different sweep from anchor A.
        latest = inspect(839, first)
        for observed in (first, first + latest):
            mapping = module.actions(observed)
            assert set(mapping.values()) == set(legal_centers(observed))
            for action, center in mapping.items():
                anchor = observed[ord(action[0]) - ord("A")]
                assert len(action) == 5 and action[1] in "+-"
                assert center == anchor + int(action[1:])
                assert center // 140 == anchor // 140
            assert mapping["A+003"] == first[0] + 3
        # Current-frame mode is allowed to reference older, non-displayed anchors.
        old_region_center = module.actions(first + latest)["A+003"]
        acquired = inspect(old_region_center, first + latest)
        assert len(set(first + latest + acquired)) == 12
        assert all(i // 140 == first[0] // 140 for i in acquired)
