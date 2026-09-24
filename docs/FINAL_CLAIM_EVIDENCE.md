# DamageACT-U final claim-to-evidence map

| Claim / limitation | Frozen evidence | Release evidence path |
|---|---|---|
| Event split is leakage-controlled at the event level | 11,034 patches; 8,202 train, 912 val, 1,920 test; development/test event sets disjoint | `results/development/phase7c_*`, `manifests/event_split/` |
| Building manifests are train/val scene-disjoint | 265,273 train buildings; 30,735 validation buildings; zero building/scene overlap | `results/development/phase7d_a_*` |
| Final experts are Seed 42 only | Phase 7D-B expert seed 42, 8 epochs, best epoch 8 | `results/development/phase7d_b_*` |
| Router was frozen before TEST | Neural epoch 1; static alpha 0.45; TEST sealed during development | `results/development/phase7d_c_*`, router checkpoints |
| Clean neural gain is small and not bootstrap-significant | +0.003110781260; 95% CI [-0.000611909452, 0.007171176847] | `results/reproduction_core_v2.json`, `results/reproduction_bootstrap_v2.json` |
| Wrong-PRE safety criterion passes | Neural wrong-PRE gain vs matched POST -0.003070980525; CI [-0.006597738161, 0.000701304682] | bootstrap reproduction + held-out results |
| Neural gate suppresses paired use under wrong PRE at population level | matched valid mean 0.175213195858; wrong-PRE mean 0.165126243399 | G4 parity report + deterministic reproduction |
| Temporal-utility signal generalizes | AUROC 0.772838348490; AUROC CI [0.745558470519, 0.796601864796] | `results/heldout_events/test_neural_utility.json`, bootstrap report |
| Neural-router superiority over simpler routers is not established | Development neural did not outperform HGB reliably; final clean-effectiveness criterion failed | Phase 7B/7D-C evidence + precommitted interpretation |
| Training-from-scratch reproducibility is not established by G3 | G3 checks frozen checkpoint inference on selected deterministic replay | G3 audit/report |
| Final evidence is not a multi-seed robustness claim | Event-track final execution used Seed 42 only | canonical evidence audit |

## Mandatory interpretation

DamageACT-U should be described as a POST-anchored temporal-utility routing and
reliability framework. The release supports selective use of temporal evidence and
wrong-correspondence safety analysis. It does not support a claim of statistically
significant clean neural-router superiority, random-seed robustness, or
training-from-scratch reproduction.
