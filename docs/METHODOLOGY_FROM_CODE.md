# DamageACT-U methodology reconstructed from the frozen code

This document is the implementation-grounded methodology specification for the completed DamageACT-U experiment. It is intended to make future paper writing straightforward **without inventing methods that were not executed**.

The authoritative frozen scientific release is tag `paper-v1.0.0` at commit `618ce84790c7e96efd0a7df3dea98f8b37b36cd9`. Where the cleaned `src/damageactu/` package and the historical execution code differ in role, the executed Phase 7D source under `vendor/frozen_phase7d_source/` and the frozen artifacts/results are the evidence of what actually ran.

## 1. Task and labels

DamageACT-U studies building-level post-disaster damage classification from paired PRE- and POST-disaster xBD imagery.

The four ordered damage classes are:

0. no-damage
1. minor-damage
2. major-damage
3. destroyed

The primary metric is fixed four-class Macro-F1.

## 2. Event-held-out protocol

The event split is frozen in `configs/protocols/phase7c_event_split.json`.

Development uses 13 disasters:

`lower-puna-volcano, palu-tsunami, mexico-earthquake, socal-fire, woolsey-fire, portugal-wildfire, pinery-bushfire, midwest-flooding, moore-tornado, joplin-tornado, hurricane-harvey, hurricane-michael, hurricane-florence`.

Held-out TEST uses 6 disasters:

`nepal-flooding, guatemala-volcano, sunda-tsunami, santa-rosa-wildfire, hurricane-matthew, tuscaloosa-tornado`.

Within the 13 development events, 10% of patches are assigned to validation with seed 321 and event-stratified sampling. Expected frozen patch counts are:

- total: 11,034
- train: 8,202
- validation: 912
- held-out test: 1,920

The code explicitly rejects development/TEST event overlap and duplicate `pair_key` values. Held-out event performance was not opened during development.

## 3. Building manifest and crop construction

The building-level construction is frozen by `configs/protocols/phase7d_a_build.json` and the executed source under `vendor/frozen_phase7d_source/src/data/`.

Key choices:

- PRE geometry defines the crop.
- The **same pixel crop window** is applied to PRE and POST.
- context scale: 2.0
- minimum crop side: 32 px
- model input size: 192 x 192
- ImageNet normalization
- training-only paired geometric augmentation; PRE and POST receive the same rotation/flips

The canonical event-development building counts are:

- train buildings: 265,273
- validation buildings: 30,735
- no train/validation building overlap
- no train/validation scene overlap

Event-train class counts are:

- class 0: 212,466
- class 1: 15,518
- class 2: 19,528
- class 3: 17,761

## 4. Capacity-matched expert models

Two experts are compared using the same ResNet-18 encoder and the same classifier-head capacity.

Let `E(.)` be an ImageNet-pretrained ResNet-18 encoder producing a 512-dimensional vector.

### POST-only expert

```text
z_post = E(I_post)
h_post = [z_post, z_post, z_post]
```

The repeated representation contains no additional information; it exists to keep the downstream head dimension and trainable capacity matched to the Siamese model.

### Siamese temporal expert

```text
z_pre  = E(I_pre)
z_post = E(I_post)
d      = |z_post - z_pre|
h_pair = [z_pre, z_post, d]
```

Both representations are 1536-dimensional and use the same head:

```text
Linear(1536, 512)
ReLU
Dropout(0.30)
Linear(512, 4)
```

This design isolates the effect of supplying temporal PRE information more cleanly than comparing heads of different capacity.

## 5. Expert training

The frozen training protocol is `configs/protocols/phase7d_b_training.json`.

Executed expert settings:

- seed: 42
- epochs: 8
- optimizer: AdamW
- learning rate: 1e-4
- weight decay: 1e-4
- train batch size: 64
- evaluation batch size: 96
- AMP enabled
- no scheduler
- no early stopping
- no focal loss
- no oversampling
- checkpoint metric: validation four-class Macro-F1
- exact tie: earlier epoch
- wrong-PRE was not used for checkpoint selection
- TEST was not used during expert training/selection

Cross-entropy class weights:

- class 0: 0.3491195254016762
- class 1: 1.2918169707179104
- class 2: 1.1515690289737834
- class 3: 1.20749447490663

Three seeds (42, 1337, 2026) were originally planned, but only Seed 42 was executed because of GPU/resource limitations. Therefore the final event-track evidence is **single-initialization evidence only**.

## 6. Wrong-PRE correspondence intervention

The purpose of wrong-PRE is to test sensitivity to **incorrect temporal correspondence**, not to estimate how often such errors occur in deployment.

For a target building, donor selection:

1. restricts candidates to the **same scene**;
2. forbids self-donation;
3. prefers donors whose crop IoU with the target is at most 0.01;
4. among preferred donors, minimizes absolute log footprint-area difference, then IoU, then building ID;
5. if no preferred donor exists, minimizes IoU, then area difference, then building ID.

The validation donor map contains 30,698 targets, with coverage about 0.9987961607.

The wrong-PRE intervention keeps the target POST image/label but substitutes a PRE crop from the selected other building. This creates a controlled correspondence failure while keeping scene-level appearance closer than an arbitrary cross-scene donor would.

## 7. Router input features

Routing uses **only inference-available expert-output diagnostics**. The 26 frozen features are created by `src/damageactu/routing/features.py`.

For each class `k in {0,1,2,3}`:

- POST probability `p_post[k]`
- paired probability `p_pair[k]`
- signed difference `p_pair[k] - p_post[k]`
- absolute difference `|p_pair[k] - p_post[k]|`

This contributes 16 features.

The remaining 10 are:

- POST confidence
- paired confidence
- POST entropy
- paired entropy
- POST expected severity
- paired expected severity
- confidence difference
- entropy difference
- severity difference
- expert prediction-disagreement indicator

No raw image pixels or ground-truth labels are router inputs at inference time.

## 8. Temporal-utility target

For router learning/evaluation, the important subset is where the POST and paired experts differ in correctness.

Let:

```text
post_correct = (pred_post == y)
pair_correct = (pred_pair == y)
discordant   = post_correct != pair_correct
```

On discordant cases, the binary utility target is:

```text
q = 1 if the paired expert is correct
q = 0 if the POST expert is correct
```

Thus the router is trained/interpreted as estimating whether temporal evidence is useful relative to the safer POST anchor, rather than merely predicting pair validity.

## 9. Routing mechanisms and controls

The frozen control set contains:

1. POST-only
2. Siamese
3. equal 0.5 logit blend
4. static-alpha blend
5. max-confidence selector
6. min-entropy selector
7. logistic soft gate
8. HGB hard router
9. neural temporal-utility router
10. oracle expert selector

The selected static alpha is 0.45.

### Soft routing equation

If `L_post` and `L_pair` are four-class expert logits and `u in [0,1]` is a gate:

```text
L_route = L_post + u (L_pair - L_post)
```

Therefore:

- `u = 0` gives the POST expert exactly;
- `u = 1` gives the Siamese expert exactly;
- intermediate values continuously interpolate in logit space.

### Logistic router

The logistic model predicts paired utility from the 26 frozen features. Its predicted probability is used directly as a soft gate.

### HGB router

The histogram-gradient-boosting model predicts paired utility from the same features. In the final comparison it is a hard selector:

```text
use Siamese if p_utility >= 0.5
otherwise use POST
```

### Neural router

Architecture:

```text
26 -> 32 -> ReLU -> Dropout(0.10) -> 16 -> ReLU -> 1
```

A sigmoid converts its scalar output to `u`.

Frozen training configuration:

- seed: 42
- batch size: 256
- maximum epochs: 100
- optimizer: AdamW
- learning rate: 1e-3
- weight decay: 1e-4
- selected epoch: 1

The configuration records utility, wrong-gate, wrong-KL and sparsity loss coefficients of 0.5, 0.5, 0.5 and 0.01 respectively. The exact executed router is preserved as a frozen artifact; exact paper-number reproduction restores that artifact rather than refitting after TEST.

## 10. TEST sealing and precommitted decision rules

The six held-out events are evaluated only after expert and router choices are frozen.

`configs/protocols/phase7d_d_final_test.json` records:

- primary metric: four-class Macro-F1
- clean neural gain requirement: at least +0.01
- clean 95% CI lower bound must exceed 0
- wrong-PRE neural gain floor: -0.005
- wrong-PRE 95% CI lower floor: -0.01
- utility AUROC 95% CI lower bound must exceed 0.5
- utility Spearman 95% CI lower bound must exceed 0
- post-test tuning: forbidden

The release evaluator also contains an explicit `TRAINING_ALLOWED = False` guard for held-out evaluation.

## 11. Metrics

Primary performance:

- fixed four-class Macro-F1

Additional descriptive metrics:

- accuracy
- ordinal MAE `mean(|y - y_hat|)`
- per-class F1
- per-event Macro-F1
- unweighted mean across event Macro-F1 values

### Utility metrics

On correctness-discordant samples:

- AUROC of the gate for whether the paired expert is the correct expert
- AUPRC for the same target

For all clean TEST buildings, define:

```text
deltaCE = log p_pair(y) - log p_post(y)
```

The Spearman correlation between gate value and `deltaCE` measures whether larger gates align with larger true-label log-probability advantage from temporal fusion.

## 12. Scene-level uncertainty estimation

Performance differences use paired **scene bootstrap**, not building-wise bootstrap.

For each replicate, scenes are sampled with replacement and all buildings from each drawn scene are included. The final performance bootstrap uses:

- seed: 20260917
- 2,000 replicates for clean/wrong-PRE performance differences
- 1,000 replicates for utility statistics

This preserves within-scene dependence better than independently resampling buildings.

## 13. Frozen final TEST population and results

The final clean TEST contains:

- 1,920 patches
- 115,349 buildings
- 6 held-out disasters

Headline results:

- POST clean Macro-F1: 0.456562921272
- neural clean Macro-F1: 0.459673702532
- neural minus POST: +0.003110781260
- clean gain 95% scene-bootstrap CI: [-0.000611909452, 0.007171176847]
- neural wrong-PRE Macro-F1: 0.453240019462
- wrong-PRE neural gain vs matched POST: -0.003070980525
- wrong-PRE gain 95% CI: [-0.006597738161, 0.000701304682]
- utility AUROC: 0.772838348490
- utility AUPRC: 0.719635385987
- utility AUROC 95% CI: [0.745558470519, 0.796601864796]
- gate-deltaCE Spearman: approximately 0.103607
- utility Spearman 95% CI: [0.058445288351, 0.149023936958]
- valid matched mean neural gate: 0.175213195858
- wrong-PRE matched mean neural gate: 0.165126243399

Precommitted outcomes:

- clean effectiveness: **FAIL**
- class safety: **PASS**
- wrong-PRE safety: **PASS**
- gate suppression: **PASS**
- utility generalization: **PASS**
- `ALL_PRECOMMITTED_CRITERIA_PASS`: **False**

## 14. Reproducibility evidence layers

The repository separates different meanings of reproducibility.

### G3: checkpoint -> prediction parity

G3 demonstrates deterministic parity from the frozen expert checkpoints to selected saved expert predictions on the canonical replay setup. It does **not** demonstrate training-from-scratch reproduction.

### G4: predictions + routers -> final statistics parity

G4 restores saved expert predictions and frozen router artifacts, reconstructs routing/interventions/statistics, and verifies parity with the frozen paper record.

### G6: public fresh-clone reproduction

A fresh Git clone was tested with the separately packaged 12 external assets. Tests, canonical evidence audit, deterministic reproduction, and full bootstrap reproduction passed while the fresh clone remained Git-clean.

## 15. What the paper may and may not claim

Supported framing:

> DamageACT-U is a POST-anchored temporal-utility routing and reliability framework that learns when paired temporal evidence is useful and evaluates its behavior under deliberately incorrect PRE correspondence.

Do not claim:

- statistically significant clean superiority of the neural router;
- neural-router superiority over all simpler controls;
- random-seed robustness;
- training-from-scratch reproducibility;
- state-of-the-art accuracy;
- real-world prevalence of wrong PRE correspondence;
- an end-to-end building localization system.

## 16. Code-to-method index

| Method component | Frozen/clean implementation |
|---|---|
| event split | `configs/protocols/phase7c_event_split.json`, `src/damageactu/data/event_split.py` |
| crop/building data | `configs/protocols/phase7d_a_build.json`, `vendor/frozen_phase7d_source/src/data/` |
| expert architectures | `vendor/frozen_phase7d_source/src/models/damage_models.py`, `src/damageactu/models/resnet_backbone.py` |
| expert configs | `configs/experts/post_seed42.yaml`, `configs/experts/siamese_seed42.yaml` |
| wrong-PRE donor logic | `src/damageactu/interventions/donor_selection.py`, `wrong_pre.py` |
| router features | `src/damageactu/routing/features.py` |
| blend/selector logic | `src/damageactu/routing/blending.py` |
| neural router | `src/damageactu/routing/neural_router.py`, frozen checkpoint |
| metrics | `src/damageactu/evaluation/metrics.py` |
| utility target/statistics | `src/damageactu/evaluation/utility.py` |
| scene bootstrap | `src/damageactu/evaluation/bootstrap.py` |
| deterministic reconstruction | `scripts/reproduce_results_v2_core.py` |
| bootstrap reconstruction | `scripts/reproduce_results_v2_bootstrap.py` |
| final claim boundary | `docs/FINAL_CLAIM_EVIDENCE.md`, `PAPER_RESULTS.md` |
