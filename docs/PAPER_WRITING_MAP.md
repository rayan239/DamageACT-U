# Manuscript revision and evidence map

A full manuscript draft now exists. This file maps manuscript wording to the **already frozen implementation and evidence** so revisions cannot silently change the experiment after TEST was opened.

The scientific snapshot remains `paper-v1.0.0`. Nothing in this document changes a model, split, checkpoint, router, threshold, intervention, metric, or reported result.

## Methods wording locks

### 3.1 Problem formulation

Define building-level four-class damage classification from PRE/POST imagery and introduce a POST-only expert, paired expert, and gate `u`.

Use the routing equation:

```text
L_route = L_post + u (L_pair - L_post)
```

Source: `src/damageactu/routing/blending.py`.

### 3.2 Event-held-out study design

State explicitly that the final event assignment follows the Hafner/DisasterAdaptiveNet event protocol verified in Phase 7C: 13 development disasters and 6 completely held-out TEST disasters. Within the development events, the patch split is event-stratified 90/10 with seed 321.

Do not draw the six TEST events as if they are downstream of the 13 development events. They are parallel branches of the 19-event assignment.

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

Both use the same 1536->512->4 head with ReLU and dropout 0.30. Describe this as matching trainable architecture/head parameterization, not as matching inference compute.

Sources:

- `vendor/frozen_phase7d_source/src/models/damage_models.py`
- expert YAML files

### 3.5 Temporal-utility routing

Describe the 26 inference-time features, correctness-discordant utility target, logistic/HGB controls, and neural 26->32->16->1 gate.

Router-development wording must preserve the actual data roles:

- router-selection and router-diagnostic scenes are disjoint;
- diagnostic scenes are opened once after selection;
- diagnostic results do not alter selected hyperparameters;
- static alpha = 0.45 is selected on router-selection data;
- the neural router's selected epoch is 1 under the frozen selection rule.

Sources:

- `src/damageactu/routing/features.py`
- `src/damageactu/evaluation/utility.py`
- `src/damageactu/routing/neural_router.py`
- `configs/routers/`

### 3.6 Wrong-PRE intervention

Describe same-scene non-self donor construction, low crop-overlap preference, and footprint-size matching. The target POST crop and target label remain unchanged. Damage labels are not used to choose the donor.

The intervention is a controlled stress test of correspondence dependence, not an estimate of deployment mismatch prevalence.

Sources:

- `src/damageactu/interventions/donor_selection.py`
- `src/damageactu/interventions/wrong_pre.py`
- frozen donor maps

### 3.7 Evaluation and statistics

Primary metric: fixed four-class Macro-F1.

Secondary/descriptive metrics: accuracy, ordinal MAE, per-class F1, per-event Macro-F1.

Uncertainty: paired scene bootstrap.

Frozen bootstrap seeds:

- clean performance differences: **20260917**
- wrong-PRE performance differences: **20260918**
- temporal-utility statistics: **20260917**

Replicate counts:

- clean performance: 2,000
- wrong-PRE performance: 2,000
- temporal-utility statistics: 1,000

Temporal-utility statistics: discordant AUROC/AUPRC and gate-vs-deltaCE Spearman.

Inferential scope must be stated narrowly: scene-bootstrap intervals quantify sampling uncertainty **conditional on the six held-out disasters**. They are not confidence intervals over the population of all future disasters.

Sources:

- `src/damageactu/evaluation/metrics.py`
- `src/damageactu/evaluation/utility.py`
- `src/damageactu/evaluation/bootstrap.py`
- `src/damageactu/evaluation/event_analysis.py`
- `scripts/reproduce_results_v2_bootstrap.py`

### 3.8 Prespecified decision criteria

Copy thresholds exactly from `configs/protocols/phase7d_d_final_test.json`; never rewrite them based on the outcome.

The manuscript should call them prespecified study decision margins and should not imply that they are externally validated disaster-response operating tolerances unless separate validation evidence is provided.

## Results wording lock

The clean neural gain is positive but its 95% paired scene-bootstrap interval crosses zero and the precommitted +0.010 clean-effectiveness requirement was not met. Therefore the manuscript must **not** describe the neural router as significantly or conclusively superior on clean TEST.

The unconditional-fusion observation must also remain architecture-specific. Prefer:

> For the evaluated Siamese expert, valid PRE pairing did not improve pooled held-out Macro-F1.

Avoid universal claims that correctly paired PRE imagery cannot improve building-damage assessment.

The strongest defensible result is that the learned gate carries useful correctness-relative temporal-utility signal and the POST-anchored routing design is substantially less sensitive to deliberately incorrect PRE correspondence than unconditional Siamese fusion.

## Figure provenance and quantitative-graphics rules

- Quantitative figures must be rendered deterministically from frozen CSV/JSON/prediction evidence or from saved deterministic curve coordinates.
- Do not use generative image editing to reconstruct ROC, PR, forest, robustness, or event-effect geometry.
- Conceptual protocol/intervention diagrams should be vector-only unless verified real imagery with explicit provenance and redistribution permission is supplied.
- Do not imply that decorative or synthetic imagery is an xBD experimental sample.
- Figure 2 must show the 19 disasters branching independently into 13 development events and 6 held-out TEST events.
- Leave-one-event-out sensitivity should remain available as supplementary evidence for event heterogeneity.

## Required limitations

The manuscript must explicitly disclose:

- Seed 42 only for the final event-track experts/router;
- no random-seed robustness claim;
- no training-from-scratch reproduction claim from G3;
- the clean-effectiveness criterion failed;
- wrong-PRE is a synthetic controlled intervention;
- scene-bootstrap uncertainty is conditional on the six held-out disasters;
- results are specific to the frozen xBD split, ResNet-18 experts, crop policy, and routing design;
- no post-TEST tuning was performed;
- static alpha remains a competitive clean-performance baseline.

## Before submission

Human metadata must be supplied rather than inferred: final author list, affiliations, contribution roles, ORCIDs, funding, competing interests, acknowledgments, and corresponding-author details.

The external-asset access/redistribution statement must describe the actual access route available at submission time. Do not promise a future archive that does not yet exist.

Do not rewrite or move `paper-v1.0.0`. If a later manuscript/release tag is needed, create a new tag after review.
