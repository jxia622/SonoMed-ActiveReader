# Current findings

**As of October 11, 2026, there is no confirmed advantage over strong sampling baselines and no demonstrated benefit from visual history.** The initial RL pilot improved mean acquisition quality over its starting policy. A subsequent refinement increased reward variation but failed the primary development gate. This report preserves both outcomes.

[Overview](../README.md) · [Datasets](DATASETS.md) · [Methods](METHODS.md) · [Aggregate data](../results/README.md)

## What the experiments measure

There are two different tasks in this project. BreastVid measures benign/malignant balanced accuracy using a fixed ultrasound reader. ACOUSLIC measures acquisition of annotation-supported fetal abdominal measurement planes. Their numerical scores are not comparable.

For ACOUSLIC, the primary score is the **best available weighted frame-selection score (WFSS) among 12 acquired clean frames**. Per-frame weights follow the official quality convention: 1 for an optimal frame; 0.6 for a suboptimal frame when the scan contains an optimal frame; 1 for a suboptimal frame when no optimal frame exists. Unannotated frames score zero by convention, without establishing anatomical absence. The evaluator, not the policy, uses these annotations.

The official challenge scores a submitted single frame and other task components. Our best-of-set score measures candidate acquisition, not final frame selection, circumference estimation, diagnostic accuracy, or a challenge leaderboard result. The original frozen-reader stage remains separate; a completed downstream held-out reader improvement is not claimed.

## ACOUSLIC: initial pilot (v1)

The subject partition is 32 training, eight tuning, and 20 sealed internal evaluation subjects from the public training release. The challenge test was excluded. Each subject contributes one scan. Policies use two fixed starts per tuning subject; random baselines use 100 starts per subject. Repeats are averaged within subjects, giving **eight independent subject units**, not hundreds of independent trials.

All policy arms start from the same 64-update supervised recognition checkpoint and receive 32 GRPO updates. The final checkpoint is used without selecting a tuning-best iteration.

| Policy condition | Before RL | After RL | Change |
|---|---:|---:|---:|
| Full visual history | 0.2250 | 0.3500 | +0.1250 |
| Current observations + bookkeeping | 0.2250 | 0.2250 | 0.0000 |
| Blank images + bookkeeping | 0.2250 | 0.2875 | +0.0625 |

Reference means: uniform **0.4500**, training-only location prior **0.4250**, coverage **0.3500**, random actions **0.2830**, random frames **0.23925**. The annotation-privileged ceiling is 1.0000 and is not an achievable label-free comparator.

| Full-history comparison | Mean difference | Exploratory paired 95% interval |
|---|---:|---:|
| Versus matched pre-RL | +0.1250 | [0.0000, +0.3125] |
| Versus random frames | +0.11075 | [−0.0825, +0.3023] |
| Versus uniform | −0.1000 | [−0.5375, +0.3000] |

Intervals resample subjects jointly across methods with 10,000 bootstrap draws. These are marginal percentile intervals; the stored simultaneous intervals are wider and also include zero. Small-sample tuning intervals are exploratory, not confirmatory evidence.

The full-history gain comes entirely from two subjects whose scores rose from 0 to 0.5. Six subjects were unchanged. All acquisition gains arose at the first inspection; the second inspection added no mean quality in any trained arm. A difference between the full and current models therefore cannot alone be attributed to useful older visual history.

### Recognition improved; navigation remained weak

Before RL, supervised adaptation increased tuning recognition from **17/32 to 28/32 presentations (53.1% to 87.5%)**. Correct selection in both original and reversed image order increased from **6/16 to 12/16 pairs (37.5% to 75%)**. These are repeated controls within eight subjects. This is a supervised recognition result, not an RL navigation gain.

Only 5/32 training groups had different returns in each of the full and current arms; the blank-image arm had 11/32. In the full arm, 4/9 positively reinforced second actions added no new acquisition quality under shared terminal-return credit. This motivated a conditional-credit experiment; it does not establish that credit assignment was the sole cause of weak performance.

### Direct history intervention

Using the final full-history checkpoint, we held the first action, prompt tokens, image slots, bookkeeping, latest three images, and remaining budget fixed, then hid only the six older images. **8/16 second actions changed, but every paired acquisition score stayed the same**: mean 0.3500 in both conditions.

The model responds to older pixels. This intervention found no useful acquisition benefit from those pixels at this checkpoint. Eight subjects and a sparse outcome cannot establish that history never helps.

## ACOUSLIC: completed refinement (v2b)

This development campaign uses the same subjects and warm-start weights, with an image-relative action representation, conditional credit, a training-only proximity auxiliary, and 64 updates per arm. All changes were bundled, so differences cannot isolate a single component. The final update is fixed in advance; the primary model remains full history.

| Policy condition | Matched pre-RL | Final, update 64 | Change | First groups with signal | Second groups with signal |
|---|---:|---:|---:|---:|---:|
| Full visual history | 0.2625 | 0.2625 | 0.0000 | 53/64 | 90/256 |
| Current observations + bookkeeping | 0.2250 | 0.3250 | +0.1000 | 48/64 | 64/256 |
| Blank images + bookkeeping | 0.2250 | 0.2625 | +0.0375 | 50/64 | 95/256 |

The full arm's pre-RL 0.2625 differs from v1's 0.2250 because its action representation changed **before training**. Treating that difference as an RL gain would be incorrect.

All three runs and independent audits completed. Adapters changed, frozen vision/reference weights stayed unchanged, save/reload matched, and behavior-versus-teacher-forcing log-probability drift stayed below 0.000050 (threshold 0.001). Each audit replayed 832 native inputs, including conditional credit and source provenance; two representative processor roundtrips per arm passed. These checks validate execution, not efficacy.

**Interpretation:** the revised signal reached more training groups, but the primary full-history result did not improve. Current observations had the highest mean among the refined arms, still below uniform (0.4500), the location prior (0.4250), and coverage (0.3500). The development gate failed. These are point estimates on reused tuning subjects; no new significance claim is made for v2b.

### The failed v2 smoke is retained

The first implementation included starting observations in the proximity maximum. Its two-update smoke had no reward differences and unchanged weights, so dependent training jobs were canceled before starting. v2b excluded starting frames from the auxiliary potential and initialized it at zero. The failed protocol and code remain available for audit; it is not silently removed from the experiment history.

## BreastVid: why the task changed

Among 145 permitted training/development sequences, **97.68% of frames have valid lesion boxes**, with strong adjacent-frame redundancy. This makes merely finding a lesion a weak target for navigation. However, reader outputs do vary with frame choice.

On 26 development sequences, with the same frozen reader and prompt, full-span uniform 12-frame sampling achieved **60.63% balanced accuracy**, random 12-frame sampling **63.54%**, and a retrospective best-of-eight candidate selector **85.00%**. The last method uses true diagnoses to choose a set; it demonstrates oracle headroom, not a deployable policy.

The earlier navigation RL experiment had unchanged full-history balanced accuracy (**62.50% before and after**) and random sampling at **65.63%** under that experiment's initial-observation protocol. Those values are not interchangeable with the later full-span audit. An early uniform baseline truncated coverage; its old score is invalid and excluded from valid comparisons here.

A training-only, group-disjoint history probe (32 fit / 32 validation groups) found a full-history versus latest-observation evidence-log-score difference of **+0.0021**, with interval **[−0.0667, +0.0939]**. Correctness was 65.63% for both. This probe did not demonstrate predictable incremental value from older observations. See the [dataset report](DATASETS.md) for lengths, coverage, redundancy, reader budgets, and timing limitations.

## What can be claimed

| Question | Supported conclusion |
|---|---|
| Did training execute correctly? | Yes: native-input replay, probability reconstruction, adapter changes, frozen-weight checks, and checkpoint roundtrips passed. |
| Did an RL policy improve over its start? | Some exploratory means improved: v1 full and v2b current. A reliable broad improvement is not established. |
| Does it beat a standard sampling method? | v1 full has a higher mean than random, with uncertainty including no advantage. It trails uniform and the location prior. No confirmed baseline win. |
| Did previous visual observations help? | Not demonstrated. The direct v1 history intervention changed actions without improving quality. |
| Is ACOUSLIC a better research target? | Its sparse supported-plane target creates a clearer acquisition problem. Learnable conditional navigation still needs evidence. |
| Have test outcomes been opened? | No. Twenty internal ACOUSLIC evaluation subjects and 26 BreastVid test sequences remain sealed; the ACOUSLIC challenge test was never admitted. |

The next milestone is a repeatable development improvement over strong nonadaptive baselines, followed by a controlled visual-history comparison. Evaluation must not redefine success after looking at held-out outcomes. The current release is a transparent research snapshot, not a validated performance claim.
