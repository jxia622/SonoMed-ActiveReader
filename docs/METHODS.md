# Methods and safeguards

The policy chooses observations under a fixed budget. The reader is held fixed so a change in downstream performance can be attributed to the evidence presented rather than simultaneous classifier training. Acquisition and reader outcomes are logged separately.

## Observation interface

The replay environment reveals only selected clean source frames and public bookkeeping: observed indices, sweep or location information, action history, and remaining budget. Native model inputs are retained privately for audit. Annotations, source diagnosis labels, hidden frames, and observations from alternative rollouts are unavailable to the policy.

In the ACOUSLIC pilot, each scan has 840 frames arranged in six 140-frame sweeps. The episode starts with six observations, one per sweep, then permits two inspections of three fresh frames each: **12 unique frames total**. Each action requests a previously unseen center and returns its nearest three unseen neighbors in the same sweep. All frames are reachable, including regions near older observations.

v1 represents an action as a global frame index. v2b uses an observed-image anchor and signed three-digit offset; the legal mapping preserves full reachability. The reference and policy likelihoods use the same constrained action support. Invalid or duplicate observations do not silently increase the budget.

## Visual conditions

| Condition | Visual input | Bookkeeping |
|---|---|---|
| `full` | All previously acquired images | Same observed locations and budget |
| `current` | Latest observation batch; older images omitted | Same observed locations and budget |
| `geometry` | Blank image slots | Same observed locations and budget |

At the first decision, full and current both see the initial six images. At the second, full can use all nine and current can see only the latest three. The current arm omits older image slots, so native token counts and prompts can differ across trained conditions. Those arm comparisons do not isolate visual memory by themselves. The separate older-image intervention preserves all nine slots and prompt tokens, replacing only older pixels with black while holding the full checkpoint and first action fixed.

## Training

The Qwen3-VL language-side policy LoRA trains; the vision encoder and reference adapter stay frozen. A 64-update supervised plane-recognition warm start precedes ACOUSLIC RL. This recognition warm start is separate from the original ultrasound reader, which remains fixed.

v1 uses 32 group-relative policy optimization updates with eight trajectories per group. Reward is the increase in the best observed annotation WFSS. Both inspection actions receive trajectory-return credit; incremental gains are separately logged and must sum to terminal gain.

v2b uses 64 updates, four first-action branches, and two second-action continuations per branch. First-action advantages compare mean terminal return across branches. Second-action advantages compare only post-first-action gains from the same history. Alternative branches do not share visual evidence.

The auxiliary in v2b adds **0.2 × incremental maximum Gaussian proximity** to supported-plane annotations, with a width of 12 frames and matching sweep required. Only newly inspected frames enter this auxiliary; initial frames are excluded and its starting value is zero. It is a training heuristic, not a clinical annotation, and it is not asserted to preserve the optimal policy. Reported evaluation uses only the original WFSS acquisition score. BreastVid box relevance remains optional legacy machinery; it is not the primary ACOUSLIC objective.

Neither ACOUSLIC campaign trains adaptive stopping. Both use learning rate 0.00002, KL weight 0.02, a fixed final checkpoint, and one training seed. Historical values are recorded in [`protocols/`](../protocols/README.md).

## Reader and sampling controls

The existing ultrasound-adapted Qwen checkpoint evaluates selected clean images with an identical prompt across selection methods. Reader parameters do not change during comparison. BreastVid's reader produces constrained benign/malignant probabilities; dense controls pool probabilities over individual frames or batches rather than attending jointly to a complete video.

ACOUSLIC baselines include uniform sampling, random frames, random legal inspection actions, coverage, and a prior learned solely from training annotations. Uniform ACOUSLIC sampling selects two evenly spaced interior frames per sweep. Observation counts match, but starting-frame constraints and compute costs differ across sampling and policy methods. Random-action episodes provide the comparator with the same inspection interface. A privileged label-based ceiling measures available acquisition quality without representing a deployable method.

The policy uses FP32 and a 196,608-pixel image cap in ACOUSLIC. The original frozen reader uses BF16 and a 1,048,576-pixel cap. Matching unique frames does not match model calls, token counts, repeated image presentations, or latency; these are separate costs.

## Audit chain

1. **Admission:** verify source hashes and author corrections; group linked observations before partitioning.
2. **Seal:** training and development paths exclude test cases; any final evaluation requires explicit admission and fixed checkpoint records.
3. **Pixels and history:** replay chosen source frames, remove overlays/annotation channels, verify image slots, and inspect representative processor roundtrips.
4. **Probability reconstruction:** compare generation-time constrained token probabilities with teacher-forced replay under the identical legal support; maximum allowed drift is 0.001.
5. **Credit:** reconstruct rewards and action gains; verify conditional branch isolation in v2b.
6. **Parameters:** verify trainable adapter changes, frozen vision/reference stability, and save/reload equality.

The v1 development audit replayed 768 training episodes and 1,888 native inputs, with 12 representative processor roundtrips. The expanded audit including the anchored action probe covered 1,987 inputs and 18 roundtrips. Each completed v2b arm replayed 832 native inputs and two representative roundtrips. These are engineering validations, not independent measures of model quality.
