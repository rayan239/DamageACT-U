# Frozen paper-result interpretation


## Primary clean held-out endpoint


POST-only Macro-F1: 0.456563
Neural router Macro-F1: 0.459674
Neural minus POST: +0.003111
Approximate scene-bootstrap 95 percent interval: [-0.00061, +0.00717]
Pre-specified requirement: gain at least +0.010 and lower 95 percent bound greater than 0.


Decision: PRIMARY CLEAN EFFECTIVENESS HYPOTHESIS NOT CONFIRMED.


## Correspondence robustness


Approximate donor-matched held-out values:
Siamese valid PRE: 0.442718
Siamese wrong PRE: 0.415270
Neural valid PRE: 0.459524
Neural wrong PRE: 0.453240


Interpretation: the neural router was substantially less sensitive to deliberately wrong PRE than unconditional Siamese fusion.


## Temporal utility


Utility AUROC: about 0.7728
95 percent scene-bootstrap interval: about [0.7456, 0.7966]
Gate versus delta-CE Spearman: about 0.1036
95 percent interval: about [0.0584, 0.1490]
Valid-PRE mean gate: about 0.1752
Wrong-PRE mean gate: about 0.1651


Decision: TEMPORAL-UTILITY GENERALIZATION SUPPORTED under the frozen terminology rule.


## Claims that are not allowed


Do not claim state-of-the-art accuracy.
Do not claim that neural routing significantly beats all baselines.
Do not claim random-seed robustness.
Do not treat the synthetic wrong-PRE intervention as an estimate of real mismatch prevalence.
Do not claim an end-to-end building localization and damage-mapping solution.
