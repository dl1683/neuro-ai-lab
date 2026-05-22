Integrity verdict: **Not ready for a definitive claim that SC is unnecessary or that dynamics are proven superior; this pass is a useful sanity check but not a falsifiable validation of the dynamics hypothesis at this scale.**

### 1) LAMBDA=0 finding (`lam=0.0`) — does CE alone imply implicit convergence?

The claim is **not justified as stated**.

- In `training.py`, `_e5_step` defines  
  `loss = eff_lam * sc_loss + ce_loss`, and for `lambda_1 = 0.0`, `eff_lam = 0.0`, so SC is absent from the objective.  
  [training.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py)
- In `exp_b_reversal.json`, the same run shows `ce_loss` collapsing (`0.00245` by step `20000`) while `sc_loss` happens to shrink (`40.23 -> 5.91`) even though it is unpenalized.  
  [exp_b_reversal.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_b_reversal.json)
- But diagnostics contradict “convergence pressure”: for `lam=0.0`, `D2` residual mean is `0.1963`, and `D4` shows `converged_frac=0.0`, `converged_correct_frac=0.0`, so the attractor coupling is not demonstrated.  
  [diagnostics.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/diagnostics.py)  
  [exp_b_reversal.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_b_reversal.json)

`D4` returns `wrong_attractor_rate=0.0` when no examples converge, so this is not evidence of “no wrong attractors.” It is a vacuous zero.

**Interpretation:** CE alone can solve token mapping on this toy task, but not via demonstrated dynamical convergence. This is consistent with the proof: convergence (residual small) and correctness are only coupled under additional conditions, not implied by CE alone.  
[convergence_correctness.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/convergence_correctness.md)

### 2) Encoder-only confound

This is a **real and severe confound**.

- Encoder-only architecture is a full `TransformerEncoder` with positional embeddings and direct token readout, so it can do position-dependent token permutation in one pass.  
  [model.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py)
- It reaches `token_acc=1.0`, `seq_acc=1.0` on reversal in 10k held-out samples.  
  [exp_b_reversal.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_b_reversal.json)

For fixed `(V=64, L=8)` reversal, this is expected and does **not** isolate dynamics.  
It does not invalidate training code or accuracy numbers, but it does invalidate “reversal requires dynamics” claims.

### 3) Statistical rigor (single seed, no CI)

Current evidence is **underpowered for robustness claims**.

- Single seed per condition is recorded; no seed list in config; no reproducibility seed controls in runner.  
  [exp_b_reversal.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_b_reversal.py)  
  [training.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py)
- 0% WA with 10k eval examples gives only an upper bound, not proof of zero-rate:
  - 95% Clopper–Pearson upper bound after 0 failures at n=10000 is about `3.0e-4` (≈0.03%).
- With nonconvex training, seed variance can flip rare-event metrics strongly.  
  Practically, you need at least **5 seeds** for variability checks, preferably **10 seeds** for all four lambda settings + encoder-only + E1/AR. Keep evaluation size fixed or increased.

### 4) D5 invalidity impact

Material but scoped.

- `basin_perturbation` currently uses the corrected normalization (`sigma_frac * ||s|| / sqrt(n_elements)`), and the run notes that earlier runs were bugged.  
  [diagnostics.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/diagnostics.py)  
  [EXPERIMENTS.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/EXPERIMENTS.md)
- Therefore, any `basin_perturbation.stability_frac` values in `exp_b_reversal.json` are likely not usable as evidence (`0.197`, `0.247`, `0.0013`, `0.0`, etc.).  
  [exp_b_reversal.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_b_reversal.json)
- It weakens any claims about robustness and basin geometry, but does **not** directly invalidate token accuracy, margin, or the D4/D6 methodology (aside from the lam0.0 no-convergence caveat).

### 5) Non-monotonic spectral radius vs lambda

Not theoretically surprising; it is expected in a constrained optimization setting.

- Reported mean rho: `lam0.0=0.9808`, `lam0.1=0.9624`, `lam1.0=0.9755`, `lam10.0=0.9887`; max rho also non-monotonic (`0.9956` at `0.1` to `1.0105` at `10`).
- Lambda is ramped via warmup (except lam0 branch sets `warmup_steps=0`), so optimization trajectory differs by schedule, not just final weighting.  
  [exp_b_reversal.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_b_reversal.py)  
  [training.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py)
- Theoretical takeaway from the proofs: smaller rho and positive margin are both desirable, but no proof says rho is monotonic in lambda.  
  [convergence_correctness.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/convergence_correctness.md)

So “non-monotonicity” is **expected**, but the practical message is clear: this sweep shows a **dynamics-accuracy-convergence tradeoff with a sweet spot around lam≈0.1 for stability**.

### 6) Decision table verdict (“E5 viable, proceed to harder tasks.”)

**Partially justified, but premature.**

- “Proceed” is acceptable only for a diagnostic follow-up plan, not as evidence of necessity over trivial encoders.
- Current decision logic picks `best_lambda=0.0` based on accuracy tie-breaking only; this hides the no-convergence result for `lam0.0`.  
  [exp_b_reversal.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_b_reversal.py)  
  [exp_b_reversal.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_b_reversal.json)
- Harder tasks should be chosen to defeat encoder-only and expose iterative refinement:
  - longer lengths (L=16,32,64) with same V
  - variable-length output (|y|≠|x|)
  - many-to-one mappings (e.g., parity, classification from sequence)
  - multi-step denoising/iterative correction tasks
  - perturbation-robustness after adversarial/noisy input
  - non-aligned or latent-constraint tasks where encoder-only cannot trivially align positions  
  [model.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py)

### 7) Overclaim check

**Supported by evidence:**
- All models reach 100% token/seq accuracy on this task setting.
- lam0.1 has best reported contraction statistics among plausible convergent E5 settings (`mean_rho=0.9624`, `converged_frac=0.9999`).
- `D4` can be near-zero under this task when convergence is achieved.

**Not supported (or only weakly supported):**
- “CE optimization alone creates implicit convergence pressure” (lam0 confound plus zero-convergence gate).
- “E5 viable” on the basis of `lam0.0` wrong-attractor=0 without convergence.
- “Dynamics necessary” on reversal (encoder confound).
- Any robust robustness claims from D5 with current run artifacts.

### Required minimum next evidence set before claiming generality

1. Rerun all lambdas across seeds (≥5, preferably 10), with fixed seeds and seed reporting.
2. Recompute `D5` after bug-fixed code and confirm against baseline; do not use old D5 values.
3. Rework gate logic: use secondary criteria (e.g., converged_frac, residual, spectral bounds) before best-λ selection.
4. Add at least one non-encoder-solvable task before calling this framework validated beyond toy-scale transformation.