1. FOUNDATIONAL ASSUMPTION AUDIT

- Core assumptions behind Variable-T gain:
  - The model benefits from learning an operator family valid across horizons, not a single horizon-specific trajectory.
  - Evaluation assumes fixed, low-entropy latent task family (addition, V=64, seq_len=8) and that near-perfect seq_acc across horizons implies useful “widened compute window.”
  - Training budget equivalence is assumed. In this code, `T` is sampled from `{4,6,8,10,12,14,16}` with uniform probability (`mean=10`), so total step-budget per batch is approximately matched to baseline T=10, i.e., not more average computation per update.
- Could the improvement be a data-amount artifact?
  - Unlikely as “more total data,” because `TRAINING_STEPS` remains 20k and batch count is fixed.
  - It is an intervention artifact candidate: model is being trained under depth-domain augmentation, so it learns a different recursion under shorter/longer horizons.
  - Strong evidence of genuine horizon-invariance effect: baseline T=32 is 0.8853 mean vs Variable-T 0.9992 mean, while T=10 stays ~1.0.
- Important assumption that is not tested: this does not prove adaptive stopping is solved; it proves robustness to longer unroll lengths under CE-like readout objective.

2. STATISTICAL METHODOLOGY (3 seeds)

- 3 seeds is a weak signal for a nontrivial effect; it is enough for an engineering signal check, not for high-confidence inference.
- `variable_t` variance is not suspiciously “too good” by itself, but the run file does not store per-condition sample std; user-provided `std=0.0001` appears externally computed.
- Additional fragility in uncertainty quantification:
  - `evaluate_step_ablation` uses one fixed eval sample via `set_seed(9999)` and single 4096-sample batch; low Monte Carlo variance by design.
  - Recovery is averaged over 16 points/run (4 sigmas × 4 horizons) and then meaned across 3 seeds in code, which can hide tail behavior.
  - With 3 seeds and rounded metrics to 4 decimals, tiny sd claims can be partly a reporting-resolution artifact.
- Practical read:
  - `T10` and `T32` near-deterministic in Variable-T are plausible but not yet statistically hardened.
  - Need larger seed count + repeated eval draws to rule out seed-specific optimization luck.

3. DENOISING FAILURE ANALYSIS

- The code is not “just noisy,” it is likely too destructive:
  - `DENOISE_SIGMA_FRAC=0.3` and `noise_scale = sigma_frac * ||s||` in [train_denoising](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d22_robust_dynamics.py).
  - Injection happens every batch at random step 3–7 for exactly 10-step unroll in denoising variant and 4–16 in combined.
  - If `||s||` is moderate-large, absolute perturbation is huge; CE objective receives only final-state token supervision, which can collapse learning to random-chance (`~ln(64)=4.158` per D22 notes), exactly what you observed.
- Is sigma=0.3 “genuinely” too aggressive?
  - Given immediate total failure at all seeds and both denoising/combined, yes.
  - It is most consistent with gradient starvation via over-noisy targets, not a deeper modeling impossibility.
- Better sigma schedule:
  - Use curriculum: start `sigma_frac ∈ {0.01,0.02,0.05}` then anneal or warm up to `0.1–0.2` by training half-point and hold, not end at 0.3.
  - Inject noise probabilistically (`p` small, e.g., 0.1→0.3) rather than every batch.
  - Normalize-noise schedule: `sigma_abs = σ_frac * running_median(||s||_batch)` and clip by percentile.
  - Couple with explicit denoising loss (not pure CE) and/or E3-style denoising objective (cf. D18 regime), so gradients target denoising directly.

4. RECOVERY GAP (0 → almost 0)

- Variable-T improves recovery from -0.0246 to -0.0001 but stays non-positive; this is still consistent with “stable but not self-correcting.”
- Mechanistic reason:
  - No recovery objective in Variable-T (`train_variable_t` only samples T). It enforces horizon robustness, not basin recovery.
  - Recovery protocol perturbs final state and asks the same CE-trained dynamics to return to original basin; if the dynamics learned mostly “readout-friendly alignment by horizon,” they can stay stable but not reconstruct from nearby corruptions.
  - D21 and D17 are consistent: CE/E5 have negative/no recovery to perturbation despite decent task accuracy.
- Is this fundamental?
  - Not fundamental to UESD in general.
  - It is likely fundamental to this specific objective stack (CE-only final readout + fixed architecture + large, unstructured noise tests).

5. THESIS CONFIDENCE UPDATE (evidence-linked)

- T1: Dynamics essential for computation — **9/10**
  - Supported by D19 ratio tests (CE 0.015, E5 0.000) and T1 collapse at T=1.
  - Reinforced by D20 bottleneck sweep step-dependence (0.818→0.992 as V grows) and D22 Variable-T preserving high T performance without losing T=10.
- T4: E5 advantage over CE — **3.5/10**
  - Not supported on raw task accuracy: D18 shows CE 0.9999/0.9998, E5 1.0/1.0, E3 1.0/1.0.
  - Internal-metric differences exist (D5 Dyna regimes, D18 Lyapunov/alignment, D11 fixed-point behavior), but no decisive practical edge in core task metrics.
  - D24 zero-shot transfer failure and marginal transfer further weakens “stronger computation regime” claims.
- T5: Parallel computation — **9/10**
  - Strong support from D7 (all positions converge around same step), D10 (halting spread ≈0), D16 (parallel accumulation/no backtracking for CE), and D8/D17 carry-causality findings.
  - D22 does not contradict this; it likely improves this profile by reducing over-iteration harm.
- T6: Causal repair / recovery — **2/10**
  - D17 max recovery remains low (~16.8% at +20 for E5, 9.7% CE at 1/4 corrupted), D21 has all recovery negative for tested sigmas, and D22 Variable-T still non-positive mean recovery (-0.0001).
- Overall thesis confidence — **6/10**
  - D22 is a real upgrade from 4.5/10 because it closes finite compute window and removes over-iteration harm, but it does not resolve the recovery failure axis.

6. WHAT IS MISSING (2–3 high-value experiments)

- Recovery-first objective ablation: keep Variable-T and add explicit perturbed-state recovery term (e.g., enforce WA decrease after k extra steps) with sigma curriculum; evaluate positive recovery at multiple perturbation levels.
- Sigma-sweep denoising on separate axis: run `sigma_frac ∈ {0.01,0.03,0.05,0.1,0.2}` and failure modes by injection step; compare to D18-style E3 to calibrate whether denoising can be made useful.
- Statistical hardening: rerun Variable-T with at least 10 seeds and repeated eval draws to estimate CI for T32 and recovery; current 3-seed estimate is too thin for hard claims.

7. CROSS-DOMAIN CONNECTION

- Variable-T maps cleanly to curriculum/multi-scale time training: exposing the model to short, medium, long horizon dynamics is equivalent to learning scale-consistent transition operators.
- It aligns with numerical ODE thinking in docs (`docs/UNIFIED_ERROR_SPACE.md`, especially adaptive/noise control ideas): changing T is discrete analog of adaptive step-count or step-size policies.
- It also aligns with dynamical-systems regularization: forcing stability across rollout lengths reduces spectral sensitivity and improves extrapolation/termination behavior.

8. PARSIMONY

- It simplifies the protocol, not the model: same architecture, same optimizer, no new module—just `T` randomization in training.
- This is good for engineering confidence and does not burden the framework.
- However, a simpler explanation exists: Variable-T may be acting as a robustness regularizer on recurrent depth rather than proof of deeper UESD-specific mechanisms.
- Therefore it raises confidence in “iterative horizon-robust dynamics” more than in “new UESD recovery principle”; recovery remains the weak link.

References:
- [exp_d22_robust_dynamics.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d22_robust_dynamics.py)
- [exp_d22_robust_dynamics.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d22_robust_dynamics.json)
- [EXPERIMENTS.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/EXPERIMENTS.md)
- [UNIFIED_ERROR_SPACE.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/docs/UNIFIED_ERROR_SPACE.md)