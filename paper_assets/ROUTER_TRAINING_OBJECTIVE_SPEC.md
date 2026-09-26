# Exact neural-router training objective

The executed Phase 7D-C router code defines:

`L = L_clean + 0.5 L_utility + 0.5 L_wrong_gate + 0.5 L_wrong_KL + 0.01 L_sparsity`.

The terms are:

- `L_clean`: cross-entropy of the valid-PRE routed logits and the true damage label.
- `L_utility`: positive-class-weighted binary cross-entropy on correctness-discordant valid-PRE cases only. The target is `q=1` when Siamese is correct and POST is wrong, otherwise `q=0` when POST is correct and Siamese is wrong.
- `L_wrong_gate`: BCEWithLogits between the wrong-PRE gate logit and zero target.
- `L_wrong_KL`: PyTorch `kl_div(log_softmax(L_routed_wrong), softmax(L_post), reduction="batchmean")`, i.e. `KL(p_POST || p_routed_wrong)` under PyTorch's target/input convention.
- `L_sparsity`: mean valid-PRE gate `u_valid`.

The positive utility-class weight is computed from the discordant router-training cases as `n_negative / n_positive`.

This loss is a training regularization design. It does not justify interpreting the learned gate as a direct probability that PRE correspondence is valid.
