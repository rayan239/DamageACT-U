# Candidate paper captions

## Figure 1 — DamageACT-U model architecture
Two independently trained experts have the same trainable architecture and parameter count. The POST-only expert uses a separate ImageNet-pretrained ResNet-18 and repeats its 512-D POST embedding three times only to match the 1536-D head input. The Siamese expert applies one weight-shared ImageNet-pretrained ResNet-18 to PRE and POST and concatenates PRE, POST, and absolute feature-difference embeddings. Twenty-six inference-time expert-output diagnostics are frozen-z-score normalized and passed to a 26→32→16→1 temporal-utility router. Its sigmoid gate u interpolates the two experts in logit space as L_route=L_post+u(L_pair−L_post).

## Figure 1b — Wrong-PRE intervention
Controlled correspondence stress test. A target building keeps its own POST crop and true label while its PRE crop is replaced by a non-self donor from the same scene. Donor construction preferentially enforces PRE-referenced crop IoU ≤ 0.01 and matches footprint scale. This is a stress test of correspondence dependence, not an estimate of real-world mismatch prevalence.

## Figure 2 — Final event-held-out protocol
The 19 disasters are separated into 13 development events and six completely held-out TEST events. Expert and router decisions are frozen before the six TEST disasters are opened. The final TEST contains 1,920 patches and 115,349 buildings. Performance uncertainty is estimated by scene-level rather than building-level bootstrap, and post-TEST tuning is forbidden.

## Figure 3 — Clean held-out performance
Clean pooled four-class Macro-F1 across all frozen controls on 115,349 buildings from six unseen disasters. The static α=0.45 blend has the highest non-oracle Macro-F1 (0.460218), followed closely by the neural router (0.459674); POST-only is 0.456563.

## Figure 4 — Clean gain relative to POST
Observed clean Macro-F1 differences relative to POST-only with paired scene-bootstrap 95% intervals. The neural point estimate is +0.003111, but its interval [−0.000612,+0.007171] crosses zero and the precommitted +0.010 clean-effectiveness requirement is not met.

## Figure 5 — Wrong-PRE robustness
Matched valid-PRE and deliberately wrong-PRE Macro-F1. Unconditional Siamese fusion drops from 0.442718 to 0.415270, whereas neural routing drops from 0.459524 to 0.453240. The neural wrong-PRE result remains slightly below matched POST (−0.003071) but satisfies the precommitted safety floor.

## Figure 6a — Temporal-utility ROC
On 14,340 correctness-discordant TEST cases, the neural gate ranks pair-useful cases with AUROC 0.772838 (scene-bootstrap 95% CI 0.745558–0.796602).

## Figure 6b — Temporal-utility precision–recall
On the same correctness-discordant population, the neural gate achieves AUPRC 0.719635 (scene-bootstrap 95% CI 0.676572–0.762623).

## Figure 6c — Gate association with continuous temporal benefit
The neural gate is positively but weakly associated with ΔCE=log p_pair(y)−log p_POST(y): Spearman ρ=0.103607 with scene-bootstrap 95% CI [0.058445,0.149024]. This is evidence of weak monotonic utility association, not a calibrated probability interpretation.

## Figure 6d — Population-level gate suppression
On donor-matched TEST buildings, the mean gate decreases from 0.175213 with valid PRE to 0.165126 with wrong PRE. This is a population-level suppression effect and must not be described as reliable per-sample mismatch detection.

## Figure 7 — Event heterogeneity
Neural-minus-POST Macro-F1 is shown separately for each of the six held-out disasters. Positive effects on some events and negative effects on others explain why pooled improvement does not imply consistent event-wise superiority.

## Supplementary figures
The generator produces expert validation history, static-alpha sweep, per-class clean and wrong-PRE F1, leave-one-event-out sensitivity, normalized neural confusion matrices, wrong-PRE donor-overlap distribution, TEST class distribution, development-only router diagnostics, and the exact router-training objective diagram.
