# DamageACT-U paper-asset QA report

**Status: PASS for the paper-asset freeze.**

The generated assets were built from the frozen Phase 7D-D final result package
plus the frozen Phase 7D-C neural-router artifact. No model, threshold, split,
test population, or precommitted decision was changed.

## Hard scientific guards checked during generation

- Final clean TEST population: 115,349 buildings.
- Final wrong-PRE matched population: 115,269 buildings.
- Expert initialization: Seed 42 only.
- Neural router selected epoch: 1.
- POST clean Macro-F1: 0.456562921272.
- Neural clean Macro-F1: 0.459673702532.
- Neural−POST clean gain: +0.003110781260.
- Neural wrong-PRE Macro-F1: 0.453240019462.
- Neural−POST wrong-PRE gain: −0.003070980525.
- Utility AUROC and AUPRC reconstructed from per-building predictions.
- Gate–ΔCE Spearman reconstructed from per-building predictions.
- Frozen clean-effectiveness decision remains FAIL.
- `ALL_PRECOMMITTED_CRITERIA_PASS` remains False.
- Clean performance bootstrap: 2,000 scene replicates, seed 20260917.
- Wrong-PRE performance bootstrap: 2,000 scene replicates, seed 20260918.
- Temporal-utility bootstrap: 1,000 scene replicates, seed 20260917.

## Editorial QA choices

- The clean gain forest plot is preferred over a raw ranking bar chart in the
  manuscript because it displays the uncertainty that makes the clean endpoint
  inconclusive.
- Static α=0.45 remains visible because it slightly exceeds the neural router
  in clean pooled Macro-F1.
- Event-level heterogeneity is explicitly visualized.
- Gate suppression is described only as a population-level shift.
- Wrong-PRE is described as a controlled intervention, not a real-world
  mismatch prevalence estimate.
- Historical Phase 1–6 results are excluded from the primary final-TEST
  evidence.
- Conceptual protocol/intervention diagrams are vector-only; they do not use synthetic satellite thumbnails that could be mistaken for experimental xBD samples.
- Quantitative paper graphics must be rendered deterministically from frozen evidence or saved deterministic curve coordinates; generative image editing is not an acceptable source for quantitative geometry.
- Qualitative xBD images are intentionally not hand-picked. If added later, their selection rule must be frozen before viewing candidates and their provenance/redistribution status must be explicit.

## Remaining non-scientific item

Final journal/conference formatting (font size, column width, caption length,
figure numbering) should be adapted only after the target venue is chosen.
That is a layout step, not a change to the evidence.

## Additional devil's-advocate completeness check

The package now also documents the exact router-training loss and the deployed
compute structure. It does not report measured latency/FLOPs because those were
not frozen experimental endpoints. A qualitative image panel is optional and
has a predefined selection protocol to prevent post-hoc cherry-picking.

## Reviewer-facing uncertainty scope

Scene-bootstrap intervals quantify sampling uncertainty conditional on the six held-out TEST disasters. They must not be presented as confidence intervals over the population of all future disasters. Event-level and leave-one-event-out analyses remain important complementary evidence for heterogeneity.
