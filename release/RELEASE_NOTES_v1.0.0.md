# DamageACT-U — paper-v1.0.0 release notes

This release freezes the reproducibility package for the final event-held-out
DamageACT-U experiment.

## Frozen scope

- Phase 7C event/split audit
- Phase 7D-A building manifests and wrong-PRE donor construction
- Phase 7D-B POST and Siamese expert artifacts
- Phase 7D-C frozen router/control artifacts
- Phase 7D-D sealed held-out evaluation
- G3 checkpoint-inference parity evidence
- G4 deterministic router/intervention parity
- G5 repository/external-asset release split
- G6 release-readiness and fresh-clone reproduction

## Headline held-out results

- POST clean Macro-F1: 0.456562921272
- Neural router clean Macro-F1: 0.459673702532
- Neural minus POST: +0.003110781260
- Clean gain scene-bootstrap 95% CI: [-0.000611909452, 0.007171176847]
- Neural wrong-PRE Macro-F1: 0.453240019462
- Wrong-PRE gain vs matched POST: -0.003070980525
- Utility AUROC: 0.772838348490
- Utility AUPRC: 0.719635385987

## Precommitted interpretation

- Clean effectiveness: **FAIL**
- Class safety: **PASS**
- Wrong-PRE safety: **PASS**
- Gate suppression: **PASS**
- Utility generalization: **PASS**
- `ALL_PRECOMMITTED_CRITERIA_PASS`: **False**

## Important limitations

The final event-track execution is single-initialization (Seed 42). Random-seed
robustness is not established. G3 establishes checkpoint-inference parity, not
training-from-scratch reproduction. The clean held-out neural-router gain has a
95% scene-bootstrap interval spanning zero, so this release does not claim
statistically significant clean superiority of the neural router.

Large frozen assets are distributed separately and are verified through
`release/external_assets_manifest.json`.
