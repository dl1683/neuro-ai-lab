Overall: the new implementation is **mostly correct in form**, but it is **not yet a strict implementation of Proposition 21** as written (fixed-point-local claim), and a few robustness fixes are recommended.

Findings (highest risk first):

1. [**Medium**] The spectral estimate is not necessarily at `s*` (fixed point).
   - The code measures at states after 10 unrolled steps (`s` from 10-step rollout), not a converged fixed point check.
   - This is reasonable for a proxy, but it does not strictly match “`dG/ds` at fixed point” language in the proposition.  
   - Location: [`estimate_spectral_radius`](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L318)

2. [**Medium**] Sampling is deterministic and under-specified for reliability.
   - Uses `for sample_idx in range(min(n_samples, 64))` (first 64 examples only), so the 64 are fixed-prefix, not random.
   - This is cheap and usually okay, but can bias if batch ordering correlates with data patterns.  
   - Location: same function around line ~328.

3. [**Medium**] Power-iteration convergence diagnostics are absent.
   - 10 iterations may be enough in a benign spectral-gap regime, but no check for convergence/stability is done.
   - If eigengap is small, 10 iterations can under-estimate `rho`.
   - Same location.

4. [**Low**] The `lam < 1e-8` branch is logically awkward.
   - If `lam` is tiny they skip normalization and continue, which can leave `v` unchanged for subsequent steps.
   - Better behavior is to break or reinitialize `v` to avoid stalled/undefined behavior in flat regions.

5. [**Low**] Non-normal/complex-spectrum caveat is unaddressed in metric.
   - For non-normal `J`, `rho` alone can miss transient amplification; and complex-dominant pairs can make plain power iteration numerically noisy.
   - This is acknowledged in `spectral_contraction.md` and is relevant for UESD attention blocks.

What is correct:

- `dynamics_step` returns `s_new` (full update map output), not `F(s,c)`.
  - So `Jv = (G(s+eps*v)-G(s-eps*v))/(2eps)` is a finite-difference action of `dG/ds`.  
  - See [`shared/model.py` dynamics_step](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py#L41).
- Central finite differences is the right choice here (higher-order accurate than forward diff), and it is used correctly.
- `lam = ||Jv||` is correct each iteration since `v` is unit-normalized before use (effectively same as `||Jv|| / ||v||`).
- `@torch.no_grad()` and `model.eval()` are used correctly.
- Memory footprint is fine: only 64 scalar `rho` samples are stored.

Integration check:

- Called in `run_one()` after training, alongside other evals, with a clear log print and returned in results — this is correctly wired.  
  - [`run_one` call site](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d25_recovery_training.py#L416)

Recommendations to be comfortable shipping as “Proposition-21 diagnostic”:

1. Use the converged state definition from a residual threshold (or fixed-point warm-up) before measuring `dG/ds` at or near `s*`.
2. Randomly sample 64 states (or stratify) rather than first-`N` prefix.
3. Add a convergence check/early-stop for power iteration or increase iterations (e.g., 20) for safer estimates.
4. Keep a fallback for near-zero `Jv` (reseed `v` or break).
5. Consider logging both spectral radius and non-normality proxy (`sigma_max / rho`) when feasible (per existing diagnostic utilities).

Safe to ship: **conditionally yes** for exploratory experiments; **not yet** as a strict proof-linked estimate of Proposition-21 fixed-point spectral radius without the caveats above.