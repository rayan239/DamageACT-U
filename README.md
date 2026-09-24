# DamageACT-U

Learning when to trust PRE-disaster imagery for reliable building-damage assessment under disaster shift.

## Project status

The **research code, frozen artifacts, audit trail, and final event-held-out results are complete**. The manuscript has **not yet been written**, so author/citation metadata is intentionally deferred rather than guessed.

Frozen tested release:

- tag: `paper-v1.0.0`
- commit: `618ce84790c7e96efd0a7df3dea98f8b37b36cd9`
- final event-track execution: **Seed 42 only**
- training-from-scratch reproduction: **not claimed**
- clean precommitted effectiveness criterion: **FAIL**
- `ALL_PRECOMMITTED_CRITERIA_PASS`: **False**

Do not move or rewrite the frozen tag. Post-tag commits on `master` are documentation/release-hygiene updates only.

## Canonical scientific pipeline

```text
Phase 7C  event/split integrity audit
    ->
Phase 7D-A  building manifests + PRE-referenced paired crops + wrong-PRE donors
    ->
Phase 7D-B  capacity-matched POST-only and Siamese ResNet-18 experts
    ->
Phase 7D-C  temporal-utility routers and controls
    ->
Phase 7D-D  one-shot sealed six-event evaluation
    ->
G3/G4 parity checks + deterministic/statistical reproduction
```

The implementation-level methodology is documented in:

- `docs/METHODOLOGY_FROM_CODE.md` — exact method reconstructed from the frozen code/configs.
- `docs/PAPER_WRITING_MAP.md` — future manuscript Methods/Experiments section mapped to code and evidence.
- `docs/FINAL_CLAIM_EVIDENCE.md` — final claim boundaries.
- `PAPER_RESULTS.md` — frozen numerical interpretation.
- `docs/RELEASE_REPRODUCIBILITY.md` — public reproduction route.

## Core method in one paragraph

For each xBD building, the same PRE-defined crop window is applied to PRE and POST imagery. A capacity-matched POST-only expert and a Siamese PRE/POST expert share the same ImageNet-pretrained ResNet-18 encoder/head design. The Siamese representation is `[z_pre, z_post, |z_post-z_pre|]`; the POST control uses `[z_post, z_post, z_post]` so the head capacity remains matched. A temporal-utility router receives 26 inference-time features derived only from the two experts' output distributions and diagnostics. Its sigmoid gate `u` blends logits as `L = L_post + u (L_pair - L_post)`, so `u=0` recovers POST-only and `u=1` recovers the paired expert. Wrong-PRE intervention replaces PRE with a different same-scene building crop to test correspondence sensitivity. Final conclusions are based on a precommitted six-event held-out TEST with no post-test tuning.

## Exact reproduction

Large frozen assets are kept outside ordinary Git. After obtaining the external asset package and restoring/verifying it:

```bash
python scripts/reproduce_release.py
```

or:

```bash
python scripts/reproduce_release.py --assets-root <ASSET_PACKAGE_ROOT>
```

This verifies the external assets, runs the test suite, audits canonical evidence, reproduces deterministic results, and reproduces the scene-bootstrap intervals.

## Important evidence boundary

The release supports a **POST-anchored temporal-utility routing and reliability framework**. It does **not** support claims of statistically significant clean superiority of the neural router, random-seed robustness, training-from-scratch reproduction, or state-of-the-art accuracy.

## Data and asset redistribution

Raw xBD imagery is not included. The repository's derived external assets are currently held back from public archival redistribution until the applicable xBD/third-party terms are reviewed. See `DATA_LICENSE.md` and `release/ASSET_REDISTRIBUTION_REVIEW.md`.

## Citation status

A final `CITATION.cff` will be added only after manuscript authorship and affiliations are finalized. See `docs/CITATION_STATUS.md`.
