# Claim-to-code matrix


| Claim | Canonical code | Evidence |
| --- | --- | --- |
| Event-held-out split is disjoint | data/event_split.py | frozen patch-role manifest |
| Unconditional temporal fusion is wrong-PRE sensitive | interventions/wrong_pre.py | wrong-PRE metrics |
| Router input is inference-available only | routing/features.py | 26-feature schema |
| Gate learns temporal utility | evaluation/utility.py | utility AUROC/AUPRC/Spearman |
| Neural router suppresses corrupted PRE influence | routing/neural_router.py | valid-vs-wrong gate evidence |
| Clean primary endpoint was not confirmed | evaluation/event_analysis.py | final endpoint record |
| Random-seed robustness was not evaluated | audit/planned_vs_executed.json | Seed-42-only execution record |
| TEST was frozen/no retuning | phase7d_d_final_test.json | final results index and protocol SHA |
