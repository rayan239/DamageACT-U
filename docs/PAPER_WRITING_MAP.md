# Future paper-writing map

The manuscript has not yet been written. This file maps a future paper to the **already frozen implementation and evidence** so that Methods/Experiments can be drafted without changing the experiment after seeing TEST.

## Recommended Methods structure

### 3.1 Problem formulation

Define building-level four-class damage classification from PRE/POST imagery and introduce a POST-only expert, paired expert, and a gate `u`.

Use the routing equation:

```text
L_route = L_post + u (L_pair - L_post)
```

Source: `src/damageactu/routing/blending.py`.

### 3.2 Event-held-out study design

Describe 13 development disasters, 6 held-out disasters, seed-321 event-stratified 90/10 development split, and the sealed TEST policy.

Sources:

- `configs/protocols/phase7c_event_split.json`
- `src/damageactu/data/event_split.py`
- Phase 7C integrity evidence

### 3.3 Building extraction and paired crops

State that PRE geometry defines a 2x-context crop, minimum side 32 px, resized to 192x192, with the same window applied to PRE and POST. Describe paired geometric augmentation and ImageNet normalization.

Sources:

- `configs/protocols/phase7d_a_build.json`
- `vendor/frozen_phase7d_source/src/data/crop_utils.py`
- `vendor/frozen_phase7d_source/src/data/dataset.py`

### 3.4 Capacity-matched experts

POST representation:

`[z_post, z_post, z_post]`

Siamese representation:

`[z_pre, z_post, |z_post-z_pre|]`

Both use the same 1536->512->4 head with ReLU and dropout 0.30.

Sources:

- `vendor/frozen_phase7d_source/src/models/damage_models.py`
- expert YAML files

### 3.5 Temporal-utility routing

Describe the 26 inference-time features, correctness-discordant utility target, logistic/HGB controls, and neural 26->32->16->1 gate.

Sources:

- `src/damageactu/routing/features.py`
- `src/damageactu/evaluation/utility.py`
- `src/damageactu/routing/neural_router.py`
- `configs/routers/`

### 3.6 Wrong-PRE intervention

Describe same-scene non-self donor construction, low crop-overlap preference, footprint-size matching, and that the intervention is a stress test of correspondence dependence rather than an estimate of deployment error prevalence.

Sources:

- `src/damageactu/interventions/donor_selection.py`
- `src/damageactu/interventions/wrong_pre.py`
- frozen donor maps

### 3.7 Evaluation and statistics

Primary metric: four-class Macro-F1.

Secondary/descriptive metrics: accuracy, ordinal MAE, per-class F1, per-event Macro-F1.

Uncertainty: paired scene bootstrap.

Temporal-utility statistics: discordant AUROC/AUPRC and gate-vs-deltaCE Spearman.

Sources:

- `src/damageactu/evaluation/metrics.py`
- `utility.py`
- `bootstrap.py`
- `event_analysis.py`

### 3.8 Precommitted decision criteria

Copy the thresholds from `configs/protocols/phase7d_d_final_test.json`; do not rewrite them based on the outcome.

## Recommended Experiments structure

1. Data/split integrity.
2. POST-only vs Siamese expert behavior.
3. Comparison of all router controls.
4. Clean held-out event endpoint.
5. Wrong-PRE correspondence stress test.
6. Temporal-utility diagnostic.
7. Per-class/per-event analysis.
8. Scene-bootstrap uncertainty.
9. Reproducibility/parity validation.

## Results wording lock

The clean neural gain is positive but its 95% scene-bootstrap interval crosses zero and the precommitted +0.01 effectiveness threshold was not met. Therefore the manuscript must **not** describe the neural router as significantly superior on clean TEST.

The strongest defensible result is that the learned gate carries useful temporal-utility signal and the POST-anchored routing design is substantially less sensitive to deliberately incorrect PRE correspondence than unconditional temporal fusion.

## Required limitation paragraph topics

The final paper must explicitly disclose:

- Seed 42 only for the event-track experts;
- no random-seed robustness claim;
- no training-from-scratch reproduction claim from G3;
- clean effectiveness criterion failed;
- wrong-PRE is a synthetic controlled intervention;
- results are specific to the frozen xBD split, ResNet-18 experts, crop policy and routing design;
- no post-TEST tuning was performed.

## Before manuscript submission

When authorship is known, add final citation metadata on `master` and in the manuscript. Do not rewrite `paper-v1.0.0`; create a later documentation/release tag if needed.
