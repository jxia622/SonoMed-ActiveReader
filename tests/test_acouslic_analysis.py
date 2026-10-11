"""Scientific safeguards for paired subject uncertainty, without clinical fixtures."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def analysis_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "acouslic_analysis", scripts / "summarize_acouslic_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_patient_difficulty_cancels_in_paired_effects():
    # Different subjects are difficult for both methods. Independent method
    # resampling would incorrectly add that variation to the treatment effect.
    baseline = np.linspace(0.05, 0.65, 20)
    treated = baseline + 0.2
    sample = np.random.default_rng(7).integers(0, 20, size=(10000, 20))
    contrasts, radius = analysis_module().paired_subject_contrasts(
        {"rl": treated, "baseline": baseline}, [("rl", "baseline")], sample
    )
    assert contrasts[0]["difference"] == pytest.approx(0.2)
    assert contrasts[0]["paired_percentile_95"] == pytest.approx([0.2, 0.2])
    assert radius < 1e-12


def test_uncertain_null_is_not_a_gain_and_adding_comparisons_does_not_narrow_bounds():
    # Half the subjects improve and half worsen; the mean is exactly zero.
    common = np.full(20, 0.5)
    mixed = common + np.tile([-0.4, 0.4], 10)
    sample = np.random.default_rng(19).integers(0, 20, size=(10000, 20))
    module = analysis_module()
    matrix = {"rl": mixed, "baseline": common, "same": mixed.copy()}
    single, radius = module.paired_subject_contrasts(matrix, [("rl", "baseline")], sample)
    multiple, expanded_radius = module.paired_subject_contrasts(
        matrix, [("rl", "baseline"), ("rl", "same")], sample
    )
    assert single[0]["difference"] == pytest.approx(0)
    lo, hi = single[0]["simultaneous_basic_95"]
    assert lo < 0 < hi
    assert expanded_radius >= radius
    assert multiple[1]["difference"] == pytest.approx(0)
    with pytest.raises(AssertionError):
        module.paired_subject_contrasts(matrix, [("rl", "baseline")], sample[:, :10])
