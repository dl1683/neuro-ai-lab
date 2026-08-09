Review of [bottleneck_depth_scaling.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:Part%20E%20:%20Difficulty-Dependent%20Contraction) and [recovery_impossibility.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/recovery_impossibility.md:Proposition%2020) as requested.

1) Logical soundness

- Stronger points:
  - The decomposition idea in Part E (`T_min = max(T_conv, T_depth)`) is a useful working model and is consistent with how the D23 curves are being interpreted.
  - In Prop 20, linking negative recovery to propagation of perturbed carry state through depth-dependent dependencies is plausible and mechanistically consistent with the task semantics.

- Main issues:
  - In Proposition 19, assumptions are too strong for a theorem-level statement:
    - A19.1 (gradient magnitude tracks residual task difficulty) is heuristic, not derivable; it is plausible but not proved.
    - A19.2 (monotonic response of learned `sigma_max` to gradient magnitude) is an unproven behavioral prior, especially with shared parameters and nonconvex optimization.
    - A19.3 (sufficient capacity at all D) is not justified, and conflicts with observed collapse behavior near L=24 (your own “depth crisis” narrative).
  - Theorem 11/13 logic in `recovery_impossibility.md` claims CE gradients carry no basin-shape information and thus imply non-positive recovery as a hard result. This is too absolute: shared parameters mean CE updates can still alter Jacobian structure indirectly; what is true is that CE does not *directly* evaluate perturbed states, not that it supplies zero basin signal.
  - Proposition 20 uses the phrase “JE at perturbed states is unconstrained” in A20.2 as if almost unconstrained. Better is “weakly/indirectly constrained”; otherwise the proposition overstates the negative-recovery conclusion.

- Internal consistency:
  - Prop 19 and Prop 20 are directionally compatible, but the combined claim that contraction improvement from difficulty “explains” both non-monotonic `T_min` and partially offsets deep negative recovery is sound as a hypothesis, not yet as a theorem.

2) Alternative explanations

- L=12 beating L=8 can come from factors other than purely difficulty-driven contraction:
  - Seed- or width/initialization variance at low data regimes.
  - Different local minima induced by training dynamics, especially with variable `T_min` sweeps and early stopping behavior.
  - Encoder-only anomaly could reflect curriculum artifacts (token budget, optimizer noise scale, context sparsity patterns) rather than pure task hardness.
  - Attention may enable partial direct access paths that reduce effective depth on some carries, weakening the serial-chain assumption for certain L/parity classes.
- Negative recovery scaling with D may reflect:
  - Basin threshold effects (sigma crossing from near-center to boundary regime) instead of linear-depth corruption alone.
  - Saturation effects from finite-precision numerical behavior in longer unrolled dynamics, especially above the reported narrow compute windows.
  - Regularization / norm constraints causing anisotropic expansion in some state directions and contraction in others, yielding non-monotonicity by direction rather than depth.

3) Quantitative rigor

- The current data support patterns but not the specific scaling claims yet.
  - User-provided D23 summary (`L=4: T_min 3`, `L=8: 5`, `L=12: 3`, `L=16: 5`) gives only 4 points for the depth sweep, with one oscillating anomaly point and one apparent recovery at `L=12`. That is too sparse for strong scaling inference.
  - Claiming `sigma_max ~ {0.55, 0.70, 0.45}` from per-position `T=2` accuracy is under-identified and method-sensitive; it needs a defined estimator and uncertainty bands.
  - The negative recovery table (`WA@+20` rising to ~83.7% by L=16) is compelling, but proving `O(D)` scaling requires more points and fixed-confidence intervals, not just trend.
  - Units/notation are sometimes mixed (percent vs fraction) across sections; this weakens comparability.
  - The inequality/derivation in Prop20 (`N_corrupt`, `WA` bound) is informal and should be stated as an upper/lower bound model rather than a direct law.

4) Claim calibration

- Prop 19 calibration:
  - Current status should be moderate-to-weak, not strong.  
  - The core mechanism is credible, and the L=12 anomaly aligns with it, but the proof’s monotonic assumptions and inferred-`sigma_max` mapping are heuristic.
- Prop 20 calibration:
  - Current status is moderate, maybe weak in the strict scaling sense.
  - The D23 direction is real (deeper carry seems to worsen recovery under +20 dynamics), but non-monotonicity at D=6 (better than D=4 despite deeper chain) means the “purely monotonic in D” phrasing is not yet justified.
  - The “cannot get positive recovery under CE-only” claim is directionally correct for typical scales but should be stated as “systematically limited without explicit perturbation objectives,” not absolute impossibility.

5) Single priority directive

- Run a controlled 2x2 ablation and fit an explicit causal chain: freeze encoder at each L and measure `(a) sigma_max`, `(b) C_step`, `(c) WA@+K` at fixed sigma and K across seeds, then unfreeze encoder to isolate whether the non-monotonicity is driven by dynamics contraction, encoder drift, or carry-path reuse.  
  This single experiment resolves the strongest ambiguity between Propositions 19 and 20 by testing whether depth effects persist when one branch is held constant.

