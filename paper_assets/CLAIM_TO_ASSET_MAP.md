# Claim-to-asset map

| Research question / claim | Primary evidence | Recommended asset | Guardrail |
|---|---|---|---|
| RQ1: Is unconditional temporal fusion reliable on unseen disasters? | POST 0.456563 vs Siamese 0.442854 | Table 3, Figures 3–4 | Do not imply temporal fusion is generally superior. |
| RQ2: Is temporal utility learnable? | AUROC 0.772838; AUPRC 0.719635; positive gate–ΔCE Spearman | Table 6, Figures 6a–6c | Utility is relative usefulness, not PRE validity. |
| RQ3: Does POST anchoring limit correspondence harm? | Siamese valid→wrong drop ≈−0.02745; neural valid→wrong drop ≈−0.00628 | Table 4, Figure 5 | Neural wrong-PRE remains slightly below matched POST. |
| RQ4: Does this behavior generalize to held-out disasters? | Six fully unseen events; utility/safety criteria pass but clean effectiveness fails | Figures 2, 4, 7; Table 5 | Do not claim consistent event-wise improvement. |
| Class-level safety | Worst neural−POST class F1 difference −0.006047 | Table 5; per-class supplement | Safety threshold pass is not class superiority. |
| Gate suppression | mean u: valid 0.175213 vs wrong 0.165126 | Figure 6d, Table 6 | Population-level shift only. |
| Reproducibility | frozen tag, G3/G4/G6 parity | Supplementary reproducibility table/text | Not training-from-scratch reproduction. |
