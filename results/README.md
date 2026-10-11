# Aggregate results

All empirical outcomes in this directory come from training/development work. **No test outcomes are included.**

| File | Contents |
|---|---|
| [`acouslic_v1.json`](acouslic_v1.json) | Method means, subject-bootstrap intervals, paired and simultaneous contrasts, training-signal and history-intervention summaries |
| [`acouslic_v2b.json`](acouslic_v2b.json) | All three completed refinement arms, matched pre-RL means, update counts, checkpoint hashes, and independent audits |
| [`acouslic_summary.csv`](acouslic_summary.csv) | Compact cross-campaign table used by the public report |
| [`breastvid_reader.json`](breastvid_reader.json) | Frozen-reader balanced accuracy by selected-frame budget and dense controls |
| [`source_fingerprints.json`](source_fingerprints.json) | Hashes of the retained private summaries used for this export |

Acquisition WFSS and diagnostic balanced accuracy are different endpoints. v1 and v2b differ in action representation, credit, and training reward. Reused baseline means do not represent a new independent evaluation. For assumptions and uncertainty, read [Current findings](../docs/RESULTS.md).

Run `python scripts/plot_public_results.py` from the repository root to regenerate the figures and CSV from these aggregate records. Bootstrap intervals stored in the v1 file cannot be recomputed from means alone; individual subject records are not redistributed. The figure displays descriptive point estimates and explicitly identifies this limitation.
