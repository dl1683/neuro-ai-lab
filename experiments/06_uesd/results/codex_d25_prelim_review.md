### CRITICAL
- None found in this script/config that would deterministically crash or corrupt metrics.

### WARNING
- Variable-T + fixed-T recovery target mismatch can bias results:
  - Training adds recovery after a sampled `T ∈ {4,6,8,10,12,14,16}` in `[train_recovery]`.
  - Recovery evaluation always starts from `T=10` in `[evaluate_recovery]` and `evaluate_noise_robustness]`.
  - This is meaningful only if you intentionally define “recovery@10.” If you want recovery quality to align with variable-T training, evaluate recovery after the same `T` distribution (or at least report by each T).  
  - References: [train_recovery](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L154), [evaluate_recovery](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L269), [evaluate_noise_robustness](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L320)

- Checkpoint resume integrity lacks config/version guard:
  - Resume logic merges prior `runs` without validating that current experiment hyperparameters/config match the saved file.
  - If you rerun with changed defaults or edited schedule/variants, cross-run contamination can silently corrupt comparative results.
  - Add a minimal schema/version hash in `RESULTS_PATH` and verify before resume.  
  - References: [run/load checkpoint](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L76), [resume gate](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L388)

### NOTE
- Q1 (gradient flow) is mostly correct:
  - `task_loss` flows through `s -> readout_logits(s)` as expected.
  - `recovery_loss` flows through `s_perturbed -> dynamics_step` and all extra steps to logits.
  - `s.detach()` is only used on `state_norm`; it does **not** stop grad into `recovery_loss` via `s_perturbed = s + noise` since `s` is still in graph and directly added to constant noise.  
  - References: [task/recovery losses](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L166), [perturb/recover path](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L173), [dynamics step](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py#L41)

- Q2 (noise detach): keeping noise detached is the right default for this objective.
  - If noise were not detached, gradients could partly optimize for shrinking perturbation scale via `state_norm` effects (an undesirable shortcut), not true robustness.  
  - Current formulation learns robustness via state-to-output sensitivity in extra dynamics/unrolled readout, while treating corruption as exogenous.  
  - Reference: [noise construction](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L174)

- Q3 (small `T` meaning): meaningful but potentially ambiguous.
  - For low `T` (e.g., 4), state is likely pre-converged; “recovery” from perturbed `s_T` can become “local stabilization of transient trajectories,” which may not match the intended “recover final converged attractor” behavior.
  - If you want both, split two losses: one on early-`T` stability, one on late-`T` recovery (`T` near upper range).  
  - References: [T sampling](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L61), [loop](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L154)

- Q4 (GPU memory): likely **not an immediate OOM** with current shapes.
  - Worst graph depth is `(T + K)` with max `26` steps at batch 256, seq_len 8, d_model 128, single-layer dynamics.
  - This is non-trivial but usually within 32GB; still linear in unroll length, so increasing K, T, seq_len, or batch can become the first actual OOM trigger.
  - Reference: [unroll in training](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L162)

- Q5 (sigma schedule): linear ramp is fine and already correctly bounded.
  - It reaches exactly `sigma_end` at final step (`min(step/total_steps,1.0)`).
  - If learning is unstable with high `sigma_end`, use warmup-then-plateau manually; current code never plateaus before the end.
  - References: [sigma_schedule](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L95), [usage](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L173)

- Q6 (checkpointing): atomicity mostly okay.
  - Temp-file + replace is a good write pattern.
  - Minor fragility: no guarantee `results/` exists before first save; add `mkdir(parents=True, exist_ok=True)` in `save_checkpoint` to avoid runtime crash on clean directories.
  - Reference: [save_checkpoint](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L77)

- Q7 (eval protocol): as noted above, fixed-T eval is a protocol choice, not a bug.
  - It gives a clean, comparable metric at T=10, but it does not measure variable-T recovery in training distribution.
  - Add per-T recovery curves for full interpretation.  
  - Reference: [evaluate_recovery fixed T=10](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L271), [train T range](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L61)