# DamageACT-U Paper Assets Freeze v1

## Recommended main-paper set

1. **Figure 1:** exact DamageACT-U architecture.
2. **Figure 1b:** wrong-PRE intervention schematic (use as a panel with Figure 1 if space allows).
3. **Figure 2:** event-held-out experimental protocol.
4. **Figure 4:** scene-bootstrap clean gain-vs-POST forest plot.
5. **Figure 5:** valid-vs-wrong-PRE robustness.
6. **Figure 6a/6b/6c:** temporal-utility evidence as three LaTeX subfigures.
7. **Figure 7:** event-level heterogeneity.
8. **Tables 1–6:** split/integrity, architecture/training, clean results, correspondence stress test, precommitted decisions, utility diagnostics.

`fig03_clean_test_macro_f1` is useful for presentations and supplement, but the forest plot is scientifically stronger for the paper because clean differences are small.

## Recommended supplement

- training curves;
- alpha sweep;
- per-class results;
- normalized confusion matrices;
- leave-one-event-out sensitivity;
- donor-overlap distribution;
- TEST class distribution;
- development-only router diagnostic evidence;
- full all-method/all-condition tables.

## Devil's-advocate exclusions

- Do **not** use historical Phase 1–6 numbers as if they were final held-out TEST results.
- Do **not** present the neural router as the top clean-Macro-F1 method: static α=0.45 is slightly higher.
- Do **not** describe gate suppression as per-sample mismatch detection.
- Do **not** call the clean neural improvement statistically established.
- Do **not** hide the failed clean-effectiveness criterion or `ALL_PRECOMMITTED_CRITERIA_PASS=False`.
- Do **not** use bootstrap intervals as evidence of random-seed robustness; final experts are Seed 42 only.
- Do **not** hand-pick qualitative satellite examples. If qualitative examples are later added, freeze a deterministic selection rule first to avoid cherry-picking.
- Do **not** add historical calibration/conformal plots to the main paper unless the manuscript explicitly studies that historical branch; they are not part of the final Phase 7 event-held-out claim.

## Additional methodology completeness assets

- `figS11_neural_router_training_objective`: exact executed router-loss diagram.
- `tableS12_model_compute_structure`: parameter and encoder-evaluation structure; no unmeasured latency/FLOPs claim.
- `tableS13_neural_router_loss`: exact loss terms and coefficients.
- `ROUTER_TRAINING_OBJECTIVE_SPEC.md`: exact mathematical/implementation interpretation of the router loss.
- `QUALITATIVE_FIGURE_PROTOCOL.md`: deterministic anti-cherry-picking protocol if raw xBD examples are later added.

Measured wall-clock inference latency and FLOPs were not part of the frozen experiment, so this package deliberately does not invent an efficiency claim.
