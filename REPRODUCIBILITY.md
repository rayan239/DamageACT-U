# Reproducibility


The cleaned implementation must reproduce the already-frozen artifact before it is treated as canonical.


## Canonical sequence


1. Verify environment and pretrained ResNet18 identity.
2. Audit xBD.
3. Verify the official event split.
4. Build or restore the frozen building manifests.
5. Train POST-only and Siamese experts for Seed 42.
6. Export clean expert predictions.
7. Build and evaluate hard wrong-PRE correspondence.
8. Fit static, heuristic, logistic, HGB and neural routing controls.
9. Run the scene-disjoint router diagnostic audit.
10. Refit frozen router choices on all allowed development data.
11. Evaluate the six held-out events once.
12. Regenerate paper tables and figures from saved predictions.
13. Generate artifact hashes and a reproduction report.


## Frozen event protocol


Development events: 13.
Held-out events: 6.
Official split seed: 321.
Development validation fraction: 0.10.


Expected patch roles:
total 11034
train 8202
validation 912
test 1920


## Event-building regression checks


event-train usable buildings: 265273
event-val buildings: 30735
wrong-PRE event-val donor coverage: 0.9987961607288108


Event-train class counts:
0: 212466
1: 15518
2: 19528
3: 17761


Frozen CE weights:
0: 0.3491195254016762
1: 1.2918169707179104
2: 1.1515690289737834
3: 1.20749447490663


## Artifact identity


Always distinguish:
1. raw file SHA-256;
2. logical uncompressed CSV SHA-256;
3. semantic identity based on schema, primary keys and expected roles.


## Seed amendment


Planned expert seeds: 42, 1337, 2026.
Executed expert seeds: 42.
Reason: GPU/resource limitation after Seed-42 execution.
Allowed interpretation: single-initialization evidence only.


Bootstrap confidence intervals quantify sampling uncertainty across scenes. They do not replace training-seed replication.
