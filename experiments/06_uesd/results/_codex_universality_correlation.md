# Codex Review: VT k Universality Correlation (2026-05-25)

**Session:** 019e5f80-6843-7e13-a5c9-91da3517f28e
**Model:** (reviewer model redacted for public release)
**Scope:** r=-0.922 correlation across 25 paired VT/FT comparisons

---

**Bottom Line**

The correlation is mostly algebraic, not an independent discovery. Since `dk = VT_k - FT_k`, and measured `VT_k` has very low variance around `0.9880`, the regression `dk ≈ 0.9880 - FT_k` is nearly guaranteed. The substantive empirical finding is not `r=-0.912`; it is that `VT_k` collapses to a tight band across tasks/depths/seeds.

From the JSONs present, I recover `25` paired runs, all negative:

`mean VT_k = 0.988040`, `sd = 0.000558`
`mean FT_k = 0.990720`, `sd = 0.001360`
`r(FT_k, dk) = -0.922`, `R² = 0.851`
`VT variance / FT variance = 0.169`

So yes, the small VT variance is real in the current artifacts. But the correlation should be described as a consequence of that collapse, not as separate evidence.

**1. Genuine Or Tautological**

The correlation is tautological conditional on VT collapse.

The real claim is:

> Variable-T training maps learned models to `VT_k ≈ 0.9880` under this architecture, metric, and measurement protocol.

Then `dk` follows mechanically. If `FT_k=0.9940`, expect `dk≈-0.0060`; if `FT_k=0.9895`, expect `dk≈-0.0015`; if `FT_k=0.9880`, expect `dk≈0`.

Do not sell `r=-0.912` as evidence that "FT explains dk." That is just the identity with low VT noise. Sell the VT attractor itself.

**2. Alternative Explanations**

Estimator artifact remains the biggest unresolved threat. All values come from the same k estimator, same fixed-point procedure, same trajectory window, same batch regime, and rounded JSON summaries. A metric with a floor/ceiling or biased fixed-point search could manufacture an apparent attractor.

Architecture attractor is very plausible. The cleanest parsimonious version is: for `d_model=128`, `heads=4`, tied 2-layer dynamics, train_T=10, and this optimizer, successful VT training lands near `k≈0.988`, independent of task. That is still interesting, but it is not universal in the broad sense.

Training length artifact is weakened but not killed. D35b checkpoints show VT near `0.988` already from 5K onward while FT remains higher at matched loss. That argues against "all 20K runs converge to same k." But it does not rule out a broader optimization-time attractor specific to VT.

Publication bias is not dismissible. D35 V=64 failure is documented, which helps. But D35b is incomplete in the artifact I read: `7/16` runs, only `3` complete prefix-sum pairs. The claim should explicitly include all failed, killed, partial, NaN, and non-learning runs in a CONSORT-style accounting.

**3. Statistical Rigor**

The sign test p-value is arithmetically valid for 25 independent Bernoulli signs:

`p = 2^-25 = 2.98e-8` one-sided.

But the independence assumption is too strong. These are not 25 independent scientific replications. They share architecture, optimizer, data generator, measurement code, train_T, VT range, and some overlapping addition depths/seeds across D31/D33. Treat it as clustered evidence, not iid evidence.

The sign test is appropriate for the narrow question "is dk consistently negative?" It is actually better than a paired t-test if you distrust normality. A Wilcoxon signed-rank test would add magnitude sensitivity. A paired t-test is acceptable as secondary, but it will overstate precision if clustering is ignored.

The rigorous analysis should be hierarchical:

`dk ~ variant + task + depth + (1|seed) + (1|experiment)`

or bootstrap by experiment/depth clusters, not by individual pair.

**4. What Would Break The Claim**

Strong falsifier: Run a preregistered learned task where FT learns cleanly, VT learns cleanly, but final `VT_k` lands outside the attractor band, e.g. `VT_k >= 0.9900` or `VT_k <= 0.9860` under the same measurement protocol, with normal accuracy and no estimator failure.

Sharper falsifier: Find a learned pair with `FT_k > 0.990` and `VT_k >= FT_k`, i.e. `dk >= 0`.

Critical null test: The model predicts `dk=0` when `FT_k ≈ 0.9880`. So the critical FT baseline is `FT_k = 0.9880 ± measurement noise`. If VT still gives `0.9880`, `dk≈0`, and the sign streak should break or become statistically meaningless. If VT still produces a meaningful negative dk from an already-0.988 FT model, the attractor story is wrong or incomplete.

**5. Prediction Quality**

The 2% prediction is not impressive as a correlation prediction. It is mostly `dk = 0.9880 - FT_k`.

It is somewhat impressive only if the protocol was: (1) FT seed=256 completed first, (2) `VT_k=0.9880` was fixed from previous runs, (3) The VT seed=256 run was genuinely unseen, (4) No tuning of estimator/window/rounding occurred afterward.

The next prediction should be maximally informative by targeting a case where the model is likely to fail: Choose a condition expected to produce `FT_k ≈ 0.9880`. Preregister `predicted VT_k = 0.9880 ± 0.0008`, `predicted dk ≈ 0`, `sign may be zero/slightly positive/slightly negative`. That is a better test than another high-FT_k case where negative dk is guaranteed.

**6. Confidence Assessment**

Do not increase T5 above `8.8/10` based on the correlation. The correlation adds almost no independent evidence.

Keep T5 at `8.8/10`, or slightly phrase it as: `8.8/10 for VT k-contraction under the current architecture/metric stack; lower for broad universality.`

Evidence bar for `9/10`:
- Preregistered full D35b completion, including L=8
- Alternate k estimator/window/batch/fixed-point settings
- One third non-addition task
- Architecture sweep: at least `d_model`, heads, and layers
- Cluster-aware statistics
- All failed/partial runs accounted for

**7. Aggressive Skepticism**

The finding is too clean because the reported "law" may be the measurement apparatus plus one architecture.

The danger signs:
- `VT_k` values are rounded and very tightly clustered
- Same estimator everywhere
- Same architecture everywhere
- Same train_T and VT range everywhere
- Prefix-sum evidence is currently only 3 complete pairs in the artifact
- The headline correlation is mathematically induced
- D31/D33 addition pairs are not fully independent replications
- The attractor value might move if fixed-point tolerance, trajectory length, eval batch, or hidden-state norm changes

The claim should not be "VT k universality" yet. It should be:

> Under the current UESD architecture and k measurement stack, successful VT training appears to impose a stable contraction setpoint near `k=0.988`.

**8. Parsimonious Explanation**

Simpler explanation not requiring broad universality:

Variable-T training regularizes the finite-step solver toward a fixed contraction setpoint determined by architecture, train_T distribution, LayerNorm, optimizer, and the k estimator. FT is free to land at task-dependent contraction values. Therefore `dk` is just the gap between FT's task-specific k and the VT setpoint.

That explains all current observations, including the 2% prediction, without claiming a universal law.
