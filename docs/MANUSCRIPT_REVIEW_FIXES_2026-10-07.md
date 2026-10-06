# Manuscript review fixes — 2026-10-07

This document records the reviewer-style manuscript and release audit performed after the scientific result was frozen.

## Non-negotiable scientific boundary

No result was re-selected or improved during this review.

The following remain unchanged:

- frozen tag: `paper-v1.0.0`
- frozen tested commit: `618ce84790c7e96efd0a7df3dea98f8b37b36cd9`
- final event-track initialization: Seed 42 only
- POST clean Macro-F1: 0.456562921272
- neural clean Macro-F1: 0.459673702532
- clean neural-minus-POST gain: +0.003110781260
- neural wrong-PRE Macro-F1: 0.453240019462
- neural wrong-PRE minus matched POST: -0.003070980525
- clean-effectiveness decision: FAIL
- `ALL_PRECOMMITTED_CRITERIA_PASS`: False

No split, crop rule, model, checkpoint, router, alpha, threshold, donor map, metric, or decision criterion was changed.

## Corrections and clarifications applied

### Event split

The manuscript/release wording now attributes the event assignment to the Hafner/DisasterAdaptiveNet protocol verified in Phase 7C.

The protocol diagram is corrected so the 19 disasters branch into two disjoint roles:

- 13 development events
- 6 held-out TEST events

The TEST branch is not drawn as downstream of the development pool.

### Router-development roles

The revision map now states explicitly that router-selection and router-diagnostic scenes are disjoint. Router-diagnostic scenes are opened once after selection and do not alter selected hyperparameters. Static alpha = 0.45 is selected on router-selection data.

### Wrong-PRE intervention

The figure/caption now states that damage labels are not used to select wrong-PRE donors. The intervention remains a synthetic same-scene non-self correspondence stress test, not an estimate of real-world mismatch prevalence.

### Bootstrap documentation

The exact frozen seeds are:

- clean performance scene bootstrap: 20260917
- wrong-PRE performance scene bootstrap: 20260918
- temporal-utility scene bootstrap: 20260917

Replicate counts remain 2,000 / 2,000 / 1,000 respectively.

The previous shorthand that could be read as assigning seed 20260917 to both clean and wrong-PRE performance bootstraps is not exact and should not be used.

### Inferential scope

Scene-bootstrap intervals quantify sampling uncertainty conditional on the six held-out disasters. They are not confidence intervals over a population of all future disasters.

Per-event and leave-one-event-out analyses remain complementary evidence for disaster-level heterogeneity.

### Decision margins

The prespecified thresholds remain frozen exactly as executed. Manuscript wording should not imply that these study decision margins are externally validated operational disaster-response tolerances.

### Quantitative figure provenance

Quantitative figures must be rendered deterministically from frozen evidence or saved deterministic curve coordinates.

Generative image editing must not be used to reconstruct quantitative geometry.

Conceptual protocol/intervention diagrams should be vector-only unless verified source imagery with explicit provenance and redistribution permission is supplied.

### Data availability

The manuscript should describe the current state, not promise a future archive. Large derived assets required for exact artifact reconstruction remain outside ordinary Git and are not publicly redistributed while applicable xBD/third-party terms are reviewed.

### Bibliography

The manuscript bibliography should identify:

- Sebastian Gerard, Paul Borne-Pons, and Josephine Sullivan, “A simple, strong baseline for building damage detection on the xBD dataset,” arXiv:2401.17271, 2024.
- Thomas Goudemant, Benjamin Francesconi, Marjorie Bellizzi, and Adrien Dorise, “Embedded Bi-Temporal Building Damage Assessment for On-Board Data Reduction,” arXiv:2609.37013; accepted at OBPDC 2026.

## Manuscript-source items still requiring source-level handling

The repository does not currently contain the authoritative full manuscript source used to compile the reviewed 14-page PDF. Therefore the following should be handled in that source rather than by fragile binary-PDF editing:

1. Narrow any universal-sounding RQ1 wording. Preferred wording: “For the evaluated Siamese expert, valid PRE pairing did not improve pooled held-out Macro-F1.”
2. Keep figure floats in numerical reading order where the target venue permits.
3. Add mixture-of-experts / dynamic ensemble-selection context if the final Related Work revision requires a stronger novelty defense.
4. Finalize authorship, affiliations, ORCIDs, contribution roles, funding, competing interests, acknowledgments, and corresponding-author metadata.
5. Finalize the external-asset access/redistribution statement using the actual submission-time access route.

## Empirical limitation that remains unresolved

The final event-track evidence uses only Seed 42. Scene bootstrapping does not replace independent retraining across random initializations.

Multi-seed replication would strengthen the evidence, but it is **not** silently substituted into the current paper and is not required to preserve the integrity of the frozen result.

## Reviewer-facing conclusion

The manuscript should be framed as a reliability and conditional-temporal-use study, not as a state-of-the-art clean-accuracy claim.

The strongest supported conclusion is:

> Temporal evidence is useful for some buildings and harmful for others. The frozen gate carries correctness-relative temporal-utility signal and POST anchoring substantially reduces degradation under deliberately broken PRE correspondence, while clean predictive superiority over POST-only remains unconfirmed.
