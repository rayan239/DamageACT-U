# DamageACT-U frozen architecture specification

## 1. Expert design

The final system uses two **separately trained but capacity-matched** experts.

### POST-only expert

Input: POST crop, 192×192.

Encoder:
- ImageNet-pretrained ResNet-18;
- final classification layer removed;
- output embedding `z_post ∈ R^512`.

Capacity-matched representation:

`h_POST = [z_post, z_post, z_post] ∈ R^1536`.

The three copies add no information; they keep the downstream head identical in size to the temporal expert.

### Siamese temporal expert

Inputs: PRE and POST crops, both 192×192.

Within the Siamese expert, **one ResNet-18 encoder with shared weights** is applied to both inputs:

`z_pre = E(I_pre) ∈ R^512`

`z_post = E(I_post) ∈ R^512`

`d = |z_post − z_pre| ∈ R^512`

Fusion:

`h_pair = [z_pre, z_post, d] ∈ R^1536`.

### Shared classifier-head design

Both experts use:

`Linear(1536,512) → ReLU → Dropout(0.30) → Linear(512,4)`.

The four outputs correspond to no-damage, minor-damage, major-damage, and destroyed.

Trainable parameters per expert: **11,965,508**.

## 2. Router input

Each expert's logits are converted to probabilities. The router receives exactly **26 inference-time features**:

- for each of 4 classes: POST probability, Pair probability, signed difference, absolute difference = 16;
- POST and Pair confidence = 2;
- POST and Pair entropy = 2;
- POST and Pair expected severity = 2;
- confidence difference, entropy difference, severity difference = 3;
- expert prediction disagreement indicator = 1.

Total: `16 + 10 = 26`.

Before neural routing, the 26 features are standardized with the **frozen mean and standard deviation stored in the router artifact**.

## 3. Neural temporal-utility router

`Linear(26,32) → ReLU → Dropout(0.10) → Linear(32,16) → ReLU → Linear(16,1) → Sigmoid`.

Trainable router parameters: **1,409**.

The sigmoid output is `u ∈ [0,1]`.

## 4. POST-anchored logit routing

`L_route = L_post + u(L_pair − L_post)`

equivalently,

`L_route = (1−u)L_post + u L_pair`.

Therefore:
- `u=0` reproduces POST-only exactly;
- `u=1` reproduces the Siamese expert exactly;
- intermediate `u` values give soft temporal influence.

## 5. Important interpretation constraint

`u` is a **temporal-utility gate**, not a binary PRE-validity detector. The final evidence supports useful ranking of relative temporal benefit and a population-level decrease in gate value under wrong PRE; it does not support reliable per-sample mismatch detection.
