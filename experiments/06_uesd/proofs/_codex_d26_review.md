**Verdict: partial launch ready (not production-ready as a definitive D26 claim run).**

### 1) Correctness findings

1. **`compute_spectral_penalty` is differentiable, but only partially.**
   - In [`exp_d26_criticality_recovery.py`](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d26_criticality_recovery.py), `compute_spectral_penalty` builds graph through `model.dynamics_step(...)` and model parameters.
   - But `s_clean = s.detach()` blocks gradients through the **precomputed trajectory** (`s` generation path), so the penalty cannot influence how earlier dynamics steps move the state.
   - It is therefore a valid local objective on the current Jacobian neighborhood, but **not** full end-to-end trajectory-aware regularization.

2. **Spectral “radius” term is a directional amplification estimate, not a spectral radius estimator.**
   - `||G(s+eps*v)-G(s)|| / eps` with one random `v` approximates `||Jv||/||v||` for that direction only.
   - It does **not** reliably estimate spectral radius unless `v` samples top-singular direction (it usually does not in one sample).
   - Failure modes:
     - Non-normal Jacobians (norm amplification does not equal spectral radius).
     - Highly anisotropic spectra (random direction misses critical direction).
     - Too-large `eps` (nonlinear finite-difference contamination), too-small `eps` (float under/instability).
     - Using non-converged `s` (see #4).

3. **`compute_calibration_penalty` is mathematically consistent as a weak metric, but weak/incomplete.**
   - The implementation (`softmax -> max -> mean`, compare with mean accuracy) gives a simple confidence-accuracy gap (ECE-like surrogate).
   - It is not proper calibration (no binning, no reliability regression, no marginal correction for class imbalance), and it’s token-averaged.
   - Useful as a lightweight alignment signal, not a calibration diagnosis.

4. **State used for spectral probing is not guaranteed “converged”.**
   - In training loop `s` is taken after random `T in [4,6,...,16]`, yet docs/preamble says “at converged state”.
   - So this is a **local Jacobian-at-current-time** target, not fixed-point Jacobian-targeted regularization.

5. **Loop option handling is mostly correct.**
   - `use_recovery / use_spectral / use_calibration` flags are wired correctly via `cfg`.
   - Loss mixing logic is straightforward and coherent with defaults.
   - Minor caveat: recovery-only baseline in D26 is not equal CE-only baseline, only D25-like baseline.

6. **Variant config bug / design gap.**
   - `VARIANTS` has 4 configs, but does not include a true “no recovery, no spectral” control in D26.
   - Full comparison set is effectively:
     - spectral-only
     - recovery-only
     - both
     - both+calibration  
   - If you want a clean 2×2 plus full-Nishimori additive arm, you are missing the explicit plain control.

---

### 2) Performance findings

1. **Overhead estimate is likely modest but nonzero.**
   - One spectral event adds 2 extra `dynamics_step` forwards; this runs every 5 steps.
   - With average `T` around 10, this is roughly ~40% extra cost on those steps, ~8% every-5-step burst, i.e. ~1.6–2.5% wall-time average depending on variant/recovery extra-steps.
   - Not excessive.

2. **Memory peaks increase on spectral steps.**
   - Yes: extra activations for two full state trajectories (`out_noisy`, `out_clean`) are kept in graph for backward.
   - `s.detach()` does help by cutting one long dependency chain, but does not remove this extra-step memory.

3. **Potential inefficiency / missed reuse.**
   - `compute_spectral_penalty` computes both `dynamics_step(s_clean)` and `dynamics_step(s_noisy)` separately; unavoidable for finite-difference style.
   - Could be reduced via reusing `context` and mixed-precision or checkpointing, but functional correctness is fine.

---

### 3) Experimental design findings

1. **Design is close but not fully orthogonal.**
   - Good inclusion of recovery/spectral interactions, and full_nishimori as additive check.
   - Missing strict baseline arm limits causal attribution.

2. **Hyperparameters:**
   - `lambda_s=0.1` gives small absolute gradient compared to CE unless gradients are scaled favorably.
   - `lambda_c=0.01` likely tiny (consistent as weak regularizer).
   - `SPEC_EPS=0.01` is reasonable, but only if state scale is order 1 (randomly normalized noise vectors make this partly robust, still task-dependent).

3. **20k steps sufficiency**
   - 4000 spectral updates (`/5`) with one vector each is probably too noisy for stable radius targeting.
   - Expect high variance in achieved ρ unless this is long-run averaged.

4. **Confounds**
   - Recovery and spectral both act on same states and can interact nonlinearly; observed gains may be hard to attribute without full orthogonal arms + matched controls.

---

### 4) Critical detach check (your specific question)

- `s_clean = s.detach()` is **not a hard correctness bug**; it is an intentional gradient-stop design:
  - gradient flows into model parameters in the two penalty forward passes,
  - but **does not flow through the trajectory that produced `s`**.
- If your intent is full control of basin-shaping across time-to-convergence path, this is incomplete and may blunt recovery/robustness effects.
- If your intent is cheap local regularization of current-step Jacobian behavior, it is acceptable.

---

### 5) Safe-to-launch recommendation

- **Safe to run as exploratory ablation**, but **not safe yet for final conclusions** on Proposition 21 effect size.
- I’d launch only after these minimum fixes:
  1. add explicit plain control (`recovery=False`, `spectral=False`, `calibration=False`),
  2. decide whether spectral should be full-trajectory-aware (`s` not detached) in at least one arm,
  3. harden spectral estimate (multi-direction probes or small power-iteration step) and/or move probing to near-converged states.