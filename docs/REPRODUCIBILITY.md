# Reproducibility and release scope

This release supports **running the software tests, exercising synthetic replay, regenerating public figures, and inspecting the exact research implementations**. It does not include the ultrasound checkpoint or original admission manifests needed to reproduce the numerical model results.

## What is executable from a fresh clone

| Task | Included requirements | Status |
|---|---|---|
| Synthetic acquisition demo | Python + NumPy | Runnable without data or weights |
| Source tests | Project dependencies + `dev` extra | Synthetic images and tiny random model fixtures |
| Public result figures | `viz` extra + aggregate JSON | Runnable; no source images or held-out records |
| BreastVid replay smoke | Project dependencies + included config | Synthetic fixture; no dataset download |
| Dataset/model experiments | Separately obtained data, approved local model assets, admitted split manifests | Source included; exact assets not released |
| Historical v2/v2b campaign replay | Original frozen protocol, checkpoints, partition, source/config hashes | Cannot be reproduced exactly from this public repository alone |

## Install and verify

Use Python 3.11–3.13. The complete dependency set includes PyTorch; choose an appropriate CUDA build separately for GPU research. The public CPU verification environment is recorded in [`requirements/verified-local.txt`](../requirements/verified-local.txt), as a version record rather than a universal lockfile.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz]'
python -m pytest -q
python scripts/demo_acquisition.py
python scripts/plot_public_results.py
python scripts/active_reader.py --config configs/active_reader.yaml smoke-test-environment --output artifacts/replay-smoke.json
```

The synthetic demo validates the existing ACOUSLIC environment: six initial indices, two three-frame inspections, no duplicates, and a 12-frame final set. Its toy scores are explicitly synthetic and must never be used as a medical or RL performance result.

The tests check budgets, reachability, replay privacy, split guards, causal history, constrained rollout/teacher-forcing probabilities, and conditional reward credit. Tests instantiate tiny Qwen models with random weights; passing them does not validate the original checkpoint's clinical performance.

## Bring your own model and data

[`configs/navigation.example.yaml`](../configs/navigation.example.yaml) accepts locally supplied assets through `ULTRASOUND_QWEN_BASE`, `ULTRASOUND_QWEN_ADAPTER`, `ULTRASOUND_QWEN_PROCESSOR`, and `BREASTVID_ROOT`. Copy it to an ignored `configs/navigation.local.yaml`; set the paths and provenance before any real-data command. Leave the test split sealed.

The reported experiments used Qwen3-VL-4B-Instruct at base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, plus the existing ultrasound adapter. The adapter SHA-256 is recorded in the v1 protocol. **A base Qwen model or a different adapter is a new experiment**, not replication of these scores. SFT overlap checks require a local training manifest; the empty example list is not evidence of no overlap.

Dataset downloads must come from their original sources. BreastVid preparation must apply author corrections, quarantine conflicting records, group known linked sequences, and complete the audit before admitting pixels. `prepare_acouslic_pilot.py` consumes the earlier screening directory (source CSV, archive-range metadata, ZIP entries, and screened-case manifest). Those local admission artifacts are not included; this is not a one-command public ACOUSLIC download package.

## Experiment entry points

| Entry point | Role |
|---|---|
| `scripts/active_reader.py` | BreastVid admission, replay smoke, baselines, legacy GRPO, navigation |
| `scripts/evaluate_suitability.py` | Frozen-reader evidence-selection audit |
| `scripts/analyze_suitability_history.py` | Conditional history probe |
| `scripts/prepare_acouslic_pilot.py` | Deterministic admission from existing screening artifacts |
| `scripts/run_acouslic_pilot.py` | v1 smoke, recognition warm start, policy arms, explicit evaluation stages |
| `scripts/probe_acouslic_history.py` | Older-image intervention with fixed policy and first action |
| `scripts/probe_acouslic_anchored_actions.py` | Zero-update image-relative action probe |
| `scripts/train_acouslic_refinement_v2b.py` | Conditional-credit refinement with training-only auxiliary |
| `scripts/audit_acouslic_*.py` | Reconstruct inputs, source provenance, action probabilities, rewards |

Use `--help` for exact arguments. Some original runners default to private campaign filenames that are deliberately absent here; pass an explicit local `--config`, `--protocol`, and artifact paths where required. Cluster-specific launchers were excluded.

Historical protocols are **records, not ready-to-launch public configs**. Their guards bind source files, model hashes, admitted partitions, and original configuration. Do not replace recorded hashes to make an old run appear reproducible. A new campaign requires a separately named, predeclared protocol bound to its own assets and a fresh output directory; retain the checks and document changed provenance. The archived record explains why a guard may reject this portable export. No held-out evaluation command is part of the quick start.

## Public evidence and omissions

[`results/`](../results/README.md) stores aggregate means, published paired intervals, audit summaries, checkpoint fingerprints, and source-summary hashes. Per-subject arrays, medical images, mask files, patient manifests, model weights, generated responses, and native multimodal traces remain outside Git. Fingerprints link the report to retained local evidence but cannot replace independent access to the source artifacts.

The source snapshot preserves the `sonomed_vlm` package namespace and shared model/configuration helpers from SonoMed-VLM. The active-reader core is exported unchanged. Publication adds documentation, aggregate exports, plots, portable examples, and a synthetic demo. Upstream SFT benchmark results are not claimed as active-reader results.
