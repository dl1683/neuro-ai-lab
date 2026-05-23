# Codex Review: D11 Energy Landscape + D18 Error Function Comparison

## Scope and required-context note
- Followed the instructions in `experiments/06_uesd/results/_d11_d18_review_prompt.md`.
- I could not find a project-root `CLAUDE.md` under this repo (`C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab`); the closest methodology note is `README.md`, and `experiments/EXPERIMENTS.md` plus the referenced UESD artifacts were used as context.

## 1) Regime Classification (D11 + D18 synthesis)

### D11 signal
- Control (untrained) behavior is effectively random in this task slice: `seq_accuracy = 0.0`, with all 512 basins singleton-sized and high final energies (`energy_profile` rising from `18.40 -> 29.67`).
- `e5_seed42` reaches very low final-energy basins (`~0.02` mean basin energy for size-1 basins) and nearly perfect trajectory end-task accuracy (`1.0`), with near-geodesic path efficiency (`ratio ~ 1.096`).
- `dynamics_ce_seed512` also reaches perfect `seq_accuracy = 1.0`, with high path efficiency (`ratio ~ 1.075`) and a large energy drop (`63.35 -> 31.24`), plus strong basin agreement under small perturbations (`scale=1.0`: `tok_agreement=0.99`, `seq_agreement=0.96`) before degradation at larger scales.

### D18 signal
- **CE-dynamics (this is the "scattered" track in D18)**
  - `token_acc = 0.99994`, `seq_acc = 0.99976`
  - high diagnostic movement: `update_norms` start around `5.95` and decay to `4.74`
  - `lyapunov = 0.1814`, `amplification = 6.15`, `alignment = 0.6206`
  - weak denoising robustness at larger noise (`recovery_ratio` `1.68` at sigma=0.5 and `1.08` at sigma=1.0)
- **E5 (highway)**
  - `seq_acc = 1.0`, lower movement (`update_norms 3.24 -> 0.219`), lower `lyapunov = 0.0698`, low `amplification = 2.01`
  - stronger geometric alignment (`alignment = 0.7744`)
  - weaker denoising at sigma>=0.05 (`recovery_ratio < 1`), and non-zero but very low residual energy (`sc_energy = 0.00030`)
- **E3 (contractive denoiser)**
  - `seq_acc = 1.0`, highest contraction-style profile (`lyapunov = 0.0634`, `amplification = 1.89`)
  - best high-noise denoising among the three (`denoise_error = 18.68` at sigma=1.0, lower than CE/E5), with lower recovery ratios at small noise.

### Verdict on the three-regime model
The D18 regime categories are directionally supported, but D11 adds two constraints:
1) They are not separated by final-task success (all non-control tracks are near perfect on this addition slice), so regime claims are mostly about **dynamics geometry and robustness**, not raw accuracy.
2) CE and E5 are both still consistent with finite-step, non-fixed-point computation: CE has much broader transient exploration, while E5 is more strongly contractive with lower residual energy.

## 2) Interpreting CE-dynamics non-fixed-point performance

- The claim that CE-dynamics "scatters" then lands correctly is mostly consistent with D11/D18:
  - high movement magnitudes,
  - non-zero/large transient energy dynamics,
  - high final accuracy without requiring converged residuals.
- A mechanistic read: CE likely computes along a **transient attractor-hunting trajectory**, where correctness is read out from an evolving latent state path, not from a strict fixed-point of `s_{t+1}=s_t`.
- This is still a valid computation style, but it is closer to "iterative state refinement" than to classical solver-to-convergence semantics.

## 3) Ballistic Computation Hypothesis

- D11/D18 provide partial support for "ballistic-like geometry":
  - path efficiency ratios are low-contrast (`1.066` to `~1.10`), indicating near-straight motion in state space;
  - step norms for CE are large-to-steadily-lower, while E5/CE both reduce quickly.
- But this is not strong proof of a new paradigm:
  - near-geodesic movement is compatible with a recurrent transition function with strong linear component and weak curvature regularization;
  - the behavior can still be framed as standard recurrent network forward compute unrolled over depth.
- Relation to D7 parallel engine:
  - D7's carry-depth findings (near-simultaneous correction, weak chain-depth coupling) fit better with this "parallel refinement" interpretation than with literal sequential carry-wave processing.
  - In that sense, ballistic is a useful descriptive shortcut, not yet a mechanism-separated ontological claim.

Why T>1 can still help if motion is ballistic:
1. One step may move quickly toward a coarse target manifold, but multiple steps can reduce anisotropy and improve readout separation.
2. Noise handling and correction capacity improve with additional steps even if the geometric path is straight (distance is not the only objective; margin and state-to-readout alignment are).
3. Under wrong-attractor risk, later steps can still move examples into readout-correct regions even if the path geometry remains relatively direct.

## 4) D19 (step ablation) predictions

- **CE-dynamics prediction:**
  `seq_acc(T=1)` should be noticeably lower than `seq_acc(T=10)`.
  A reasonable forecast is ~mid- to high-90% at `T=1` rising to near-unity by full `T=10` (a gap large enough to reject "single-pass solves it").
- **E5 prediction:**
  smaller drop with `T`, likely still improving with 2-4 steps and saturating quickly; sequence accuracy may remain very high at `T=1` but should still improve slightly by larger `T`.
- **Most surprising possible outcome:**
  If both CE and E5 show near-equal performance at `T=1` and `T=10` (gap <=0.02), then the iterative engine claim is strongly weakened (directly touching falsification condition in the framework).
  If only one track shows this, it is still regime-specific evidence against universality of the "iterative compute" story.

## 5) Cross-Validation and falsification checks

- **Seed count is too low.** D11 uses only two trained seeds across tracks (`seed 42` and `seed 512` split across models). That is insufficient for robust variance estimates for regime-level claims.
- **Basin count at threshold 0.95 = 512 is likely a methodological artifact risk.**
  - With 512 evaluation points and strict cosine clustering, singleton basins can emerge even when basins are topologically meaningful but high-dimensionally close.
  - The threshold dependence shown (e.g., changing count by threshold in CE dynamics) suggests the metric is fragile.
- **Alternative clustering/probing needed**
  - Run distance-threshold and DBSCAN/HDBSCAN checks in PCA-reduced and raw spaces,
  - report cluster stability across seeds and bootstrap subsamples,
  - avoid single-link leader initialization bias.
- **Energy profile vs landscape slice are different measurements**
  - `energy_profile` in D11 is temporal: mean `||F_theta(s_t,c)-s_t||^2` along the actual trajectory.
  - landscape slice is conditional spatial geometry around one perturbed final state under a fixed context.
  - these are not interchangeable; trajectory energy can indicate transient solver dynamics while slice variation measures local manifold anisotropy for one neighborhood.

## 6) What to test next (maximal distinguishing power)

1. D19 with full D11-style models: `T=1,2,4,8,10,20` for each checkpoint plus corrupted/carry-hard subsets.
2. D11 recomputation with larger evaluation set and a non-cosine clustering protocol (plus cluster stability curves vs threshold and metric).
3. Paired run where path-efficiency + energy-profile and landscape-slice are measured on **the same context and with matched perturbation sets** to remove interpretation drift.
4. D7-style carry-depth and position-stratified causality probes recomputed on CE/E5/E3 checkpoints from D18 to test whether D11's geometric claims map to algorithmic carry control.
5. Causal intervention tests that directly perturb latent trajectory (and not just readout or one-step noise injection), then replay with identical readouts to separate representational from computational effects.

## 7) Integrity score versus falsification criteria
- T4 (E5 advantage in step-dependence/basins/stability): **INCONCLUSIVE** (E5 is cleaner/contractive but not uniformly superior in denoising under all corruption scales).
- T5 (parallel-not-sequential): **PARTIAL SUPPORT / INCONCLUSIVE** (current carry-chain evidence remains mostly weak-positive, but still not a clean disproof of alternatives).
- T6 (carry representations are causal): **WEAKENED** by prior causal analyses showing high carry decodability but shallow persistent effect after interventions.
