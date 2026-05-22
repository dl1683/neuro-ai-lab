# UESD Experiment C (Sort) Integrity Audit

## Review scope
- Evidence sources: `experiments/06_uesd/results/exp_c_sort.json` and `experiments/EXPERIMENTS.md` (Exp C section).
- Task: evaluate whether dynamics are necessary for sorting and whether the “encoder confound” conclusion is justified.

## Findings (ordered by severity)

1. High — Statistical conclusions are over-claimed for near-ceiling metrics.
   - Exp C is reported as a single-condition result set (no seed sweep, one eval sample of 10k sequences).
   - Reported differences are tiny:
     - `encoder_only` token accuracy = `0.999775`, `E5 lam=0.1` token accuracy = `0.999838`, `E5 lam=1.0` token accuracy = `0.999838`.
     - `encoder_only` seq accuracy = `0.99830`, `E5 lam=0.1` seq accuracy = `0.99870`, `E5 lam=1.0` seq accuracy = `0.99880`.
   - With `n=10000` sequence samples, these gaps are within sampling noise unless variance-reduced CI or seed averaging is provided.
   - Caution: `experiments/EXPERIMENTS.md` rounds/reports E1 as `1.0000/0.9999` while JSON has `0.99975/0.9981`, so the doc and artifact are inconsistent.

2. High — E1 dynamics are not functioning as a fixed-point solver despite near-ceiling accuracy.
   - `track_a_e1.wrong_attractor.converged_frac = 0.0`, residual `mean = 0.0691`.
   - This indicates success is primarily through non-iterative readout behavior, not dynamical convergence, so it cannot be interpreted as robust iterative computation.

3. Medium — “Encoder confound” label is appropriate, but the interpretation needs tightening.
   - `encoder_only` achieves `~99.93%` seq accuracy while remaining within parameter budget below UESD variants and still close to all UESD runs.
   - That means Exp C does **not** establish dynamics as necessary at `L=8, V=64`.
   - A better claim is: **sort confound remains for this task scale**, not “dynamics useless.”

4. Medium — Fairness gap in comparisons is real and should be controlled.
   - Parameter counts are asymmetric:
     - E1/E5: `694,016`
     - AR: `950,336`
     - Encoder-only: `425,344`
   - For “dynamics-necessity” claims, comparison should be either capacity-matched or compute-matched.
   - Using fewer parameters for the encoder-only control can only strengthen the confound signal (it is still too strong), but it weakens strict inferential control.

5. Medium — `max_rho > 1` in E1 is a meaningful stability warning.
   - `track_a_e1.spectral_radius.max_rho = 1.19865` (JSON) and mean_rho near `0.999`.
   - For local linearized dynamics, max spectral radius above 1 means the one-step map is expansive in at least some local directions.
   - Interpretation:
     - Not automatically catastrophic at finite `T=10` if expansion is rare/non-persistent and training couples readout to a stable manifold.
     - Theoretical tension remains: contractive assumptions used for guaranteed fixed-point/perturbation guarantees do not hold uniformly.
     - This is a strong reason to monitor non-normality, residual decay, and basin stability at larger `T`/deeper/longer settings.

## Conclusion
- Exp C supports only a weak conclusion: **at this scale, sort does not separate iterative dynamics from single-pass encoders**.
- It does not support a necessity claim for UESD dynamics, and it weakly supports the current “encoder confound” label in the intended sense.
- The more important technical signal is that E1 can be highly accurate while not convergent and locally expansive; this undermines the stability story more than the token/seq metrics alone suggest.

## Recommended minimum follow-ups
1. Report seed-averaged means and intervals (or at least 5 seeds) for seq accuracy and wrong-attractor/convergence metrics.
2. Include a parameter-matched encoder control and a compute-matched AR/encoder comparison.
3. Extend sort to larger sequence lengths / harder duplicate patterns to find a regime where single-pass attention degrades while dynamics remain correct.
4. Add explicit statistical tests (e.g., paired proportion test / bootstrap on run seeds) before declaring “competitive” differences.
