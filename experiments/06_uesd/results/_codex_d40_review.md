**Review Verdict: No CRITICAL blockers found.** D40 is launchable from a correctness/shape standpoint, with a few warnings worth fixing or documenting.

**WARNING** [exp_d40_extended_convergence.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d40_extended_convergence.py:263): multi-T evaluation uses fresh random batches for each `T`. That still estimates aggregate convergence at each depth, but it is not a paired trajectory over identical examples, so T-to-T residual/accuracy comparisons include sampling noise. The separate `convergence_trajectory()` does use a fixed batch, but only 256 samples.

**WARNING** [exp_d40_extended_convergence.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d40_extended_convergence.py:520): resume is run-granular only. It skips completed `(seed, lambda_sc)` entries from the JSON, but cannot resume a partially completed run/phase from the `.pt` checkpoint, and it does not verify that an existing JSON matches the current config. Safe enough for crash-after-run, not safe for mid-run restart or changed sweep constants.

**WARNING** [exp_d40_extended_convergence.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d40_extended_convergence.py:212): recovery loss runs `model.dynamics_step` in train mode, so Transformer dropout is active during recovery training. That may be acceptable as extra stochastic regularization, but if recovery is meant to train deterministic basin return, it should mirror SC and temporarily set `model.dynamics.eval()`.

**INFO** SC computation is correct for the D39 issue: [margin_gated_sc](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d40_extended_convergence.py:156) sets `model.dynamics.eval()` for the SC one-step residual and restores training state afterward. Gradients still flow through the eval-mode dynamics; `eval()` disables dropout, not autograd.

**INFO** Shape smoke test passes. `src/tgt=(B,16)`, `context/s=(B,16,128)`, `logits=(B,16,64)`, result loss slices to `(B*8,64)` vs `(B*8,)`. Addition targets only first half, matching `_ce_result_only`, margin, SC-gated positions, and evaluation accuracy.

**INFO** Multi-T evaluation really does test extended iteration convergence: [evaluate_at_T](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d40_extended_convergence.py:269) reinitializes `s` and applies exactly `T_eval` dynamics steps before measuring logits and one-step residual. [convergence_trajectory](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d40_extended_convergence.py:340) logs residual after T without accidentally advancing the tracked state, because `s_next` is not assigned back to `s`.

**INFO** T=200 evaluation should not OOM under the current code. Evaluation is `@torch.no_grad()`, chunked at `bs=512`, and does not retain per-step activations. It may be slow, but memory should be bounded.

Parameter check: intentional confirmed. D40 base `UESDModel` has `702,208` params. D39 `BasinCoupledUESD` has `903,856` params, so D40 is smaller by `201,648`, exactly the removed flow head path.