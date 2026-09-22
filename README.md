# DamageACT-U


Learning when to trust PRE-disaster imagery for reliable building-damage assessment under disaster shift.


## Canonical paper path


Phase7C split/data audit -> Phase7D-A buildings -> Phase7D-B experts -> Phase7D-C routers -> Phase7D-D sealed evaluation -> paper tables/figures.


Phases 1-6 and Phase7A/B are retained as development and provenance history. They are not required for the shortest reproduction of the final event-held-out claims.


## Main result


The final six-event held-out experiment did not confirm the pre-specified clean-effectiveness criterion for the neural router. The router nevertheless generalized as a temporal-utility estimator and reduced sensitivity to deliberately incorrect PRE correspondence relative to unconditional Siamese fusion.


See PAPER_RESULTS.md for the exact claim boundary.


## Three reproduction levels


A. Fast statistical reproduction:
    python scripts/reproduce_results.py


B. Frozen-model inference reproduction:
    python scripts/reproduce_inference.py --xbd-root PATH_TO_XBD


C. Full experimental replication:
    python scripts/reproduce_all.py --xbd-root PATH_TO_XBD


The full replication route is expensive. Exact paper-number reproduction should use the released frozen prediction/evidence artifacts.


## Critical limitation


The event-track expert protocol originally planned seeds 42, 1337 and 2026. Only Seed 42 was completed because of GPU/resource limitations after Seed-42 execution. Therefore the event-track evidence is single-initialization evidence only.


## Repository roles


configs/      frozen scientific decisions
src/          reusable scientific implementation
scripts/      experiment entry points
manifests/    frozen split/sample/intervention metadata
checkpoints/  frozen model artifacts distributed separately
predictions/  frozen predictions distributed separately
results/      development and held-out evidence
tests/        scientific-integrity tests
audit/        protocol history and execution-only amendments
notebooks/    paper-facing analyses and historical notebooks
