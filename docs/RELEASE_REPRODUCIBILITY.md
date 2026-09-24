# DamageACT-U release reproducibility

This document describes the **frozen artifact reproduction path** for the final
event-held-out DamageACT-U evidence.

## Scope

The canonical paper pipeline is:

Phase 7C data/split audit -> Phase 7D-A building manifests -> Phase 7D-B experts ->
Phase 7D-C routers -> Phase 7D-D sealed held-out evaluation.

The final event-track evidence is **single-initialization (Seed 42)**. Random-seed
robustness was not established.

G3 establishes checkpoint-to-saved-expert-prediction parity on the deterministic
selected Kaggle T4 replay. It does **not** establish training-from-scratch
reproduction.

G4 establishes deterministic parity from saved expert predictions and frozen
router artifacts through clean/wrong-PRE route metrics, gate behavior, and utility
statistics.

## External assets

Large frozen artifacts are distributed separately. Their exact paths, byte sizes,
and SHA256 values are stored in:

- `release/external_assets_manifest.json`
- `release/SHA256SUMS.external.txt`

Install a downloaded asset package into a clone with:

```bash
python scripts/install_external_assets.py --assets-root <ASSET_PACKAGE_ROOT>
```

## One-command frozen-result reproduction

After assets are available:

```bash
python scripts/reproduce_release.py
```

Or install from an external asset package and reproduce in one command:

```bash
python scripts/reproduce_release.py --assets-root <ASSET_PACKAGE_ROOT>
```

This runs the external-asset verifier, tests, canonical evidence audit,
deterministic core reproduction, and full bootstrap reproduction.

## Frozen headline results

- POST clean Macro-F1: 0.456562921272
- Neural router clean Macro-F1: 0.459673702532
- Neural minus POST: +0.003110781260
- 95% scene-bootstrap CI for clean gain: [-0.000611909452, 0.007171176847]
- Neural wrong-PRE Macro-F1: 0.453240019462
- Wrong-PRE gain vs matched POST: -0.003070980525
- 95% scene-bootstrap CI for wrong-PRE gain: [-0.006597738161, 0.000701304682]
- Utility AUROC: 0.772838348490
- Utility AUPRC: 0.719635385987
- Gate–deltaCE Spearman (local deterministic reconstruction): 0.103607093389

The pre-specified clean-effectiveness criterion **failed**. Class safety,
wrong-PRE safety, gate suppression, and utility generalization passed.
`ALL_PRECOMMITTED_CRITERIA_PASS` remains **False**.

These results must not be rewritten as a claim that the neural router is
statistically superior on clean held-out events.
