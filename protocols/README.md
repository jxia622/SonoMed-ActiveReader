# Historical protocol records

These files describe the completed research, including the failed intermediate smoke. They are not a public launch configuration for another checkpoint or partition.

| Record | Outcome |
|---|---|
| [`acouslic_pilot_protocol.json`](acouslic_pilot_protocol.json) | v1: three 32-update arms completed; development gate failed |
| [`acouslic_refinement_protocol.json`](acouslic_refinement_protocol.json) | v2: two-update smoke failed its weight-change gate; dependent arms did not train |
| [`acouslic_refinement_v2b_protocol.json`](acouslic_refinement_v2b_protocol.json) | v2b: three 64-update arms completed and audited; development gate failed |

The refinement records retain historical source, model, and partition hashes. Their source list includes original cluster launch/configuration files that are intentionally not distributed. The v2b engineering note was redacted to remove a cluster job identifier, so this public JSON file's own hash differs from the original protocol hash retained in result records. Scientific fields and historical digests are preserved.

See [reproducibility](../docs/REPRODUCIBILITY.md) before using the runners. No safeguard should be disabled or historical hash relabeled to bypass a mismatch. Twenty internal ACOUSLIC evaluation subjects and the BreastVid test split remain sealed.
