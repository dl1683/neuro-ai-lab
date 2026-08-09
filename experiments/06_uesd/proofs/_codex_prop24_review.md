Short verdict: Proposition 24 is **plausible but not yet rigorous**. The core gap is that “full self-attention gives O(1) information reachability” is being used as if it directly implies O(1) algorithmic carry-depth resolution, which does not follow without extra assumptions about representational and convergence capacity of the dynamics map.

1. MATHEMATICAL RIGOR

In [bottleneck_depth_scaling.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md) the argument jumps from (i) all-to-all mixing per step (self-attention + cross-attention) to (ii) `C_step` scaling with `D`, then to O(1) `T_depth`.  
The weak point is Theorem 16’s `(A16.2)`—`C_step` is explicitly treated as an empirical architecture-dependent constant, not derived from attention topology.  
So “full attention ⇒ O(1) computation” is not proved; it is a **capacity hypothesis**:
- access is clearly O(1) in graph distance;  
- computation in O(1) still requires `G(s,c)` to implement a `D`-independent carry solver as a contraction map on the joint state.

Also, `T_min = max(T_readout, T_depth)` is only a lower bound under loose independence assumptions in Theorem 17. The text also admits interaction term `I(V,D)`; with shared encoder/dynamics parameters and finite-step nonlinearity, those interactions can be substantial. This is especially relevant since `C_step` is inferred from the same accuracy curve used to infer `T_readout`, so decomposition is under-identified.

2. ALTERNATIVE EXPLANATIONS

- Memorization / short-horizon overfitting: not impossible, but weaker.
  - The D values here are small (max carry depth 12) and one seed only in the JSON run, so “global rule induction” is not ruled out.
- T=10 training bias: this is a real alternative.
  - `d22_best_variant = "variable_t"` in the data config, and train/eval asymmetry can favor a model tuned for the sampled horizon regime, which can make a T* around 5 look universal.
- Different algorithm entirely: also realistic.
  - The model could be using a global denoising-like map that bypasses explicit serial carry, i.e., it may solve the digit relation directly from context rather than emulate a carry automaton. This is behaviorally compatible with the observed curves and does not require serial carry-depth mechanics.

3. FALSIFICATION ANALYSIS

Your three predictions are good first controls, but they are **not sufficient** to fully disambiguate Proposition 24 from the alternatives:
- Local-attention ablation is useful and probably the strongest discriminator for the access-vs-computation claim.
- Reduced heads is useful but confounded by capacity/optimization effects.
- Variable-T invariance is weak as a falsifier because variable-T itself changes training dynamics in a way that can induce horizon invariance-like behavior even when depth scaling still exists.

More discriminative tests:
- Out-of-range carry depth test: evaluate at `D > 12` with the same model, same data distribution.
- Encoder ablations by intervention, not just baseline: randomize or replace encoder input per-position while preserving marginal token distribution.
- Frozen-encoder + controlled noise experiments to quantify whether gains persist when contextual carry hints are degraded below token-level MI.
- Structural probing: track whether internal hidden states encode carry bits across positions after each step, and whether this converges in constant rounds.
- Fixed-T training ablations: train at fixed `T=5` vs `T=10` vs fixed high T to separate optimization-horizon effects from architectural mechanics.

4. CALIBRATION

“MODERATE” is reasonable for the current write-up, but the current `T_min` definition is itself definition-dependent and inconsistently applied in the same section: the table for Proposition 24 claims `D=6` has `T_min≈3`, while `T=3` sequence accuracy in the JSON is ~96.5%, not >99% ([exp_d23...json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d23_carry_depth_scaling.json)).  
To move toward **STRONG**, you need:
- replicated seeds and CI/variance,
- explicit threshold protocol (`T_min` based on fixed criterion),
- extrapolation beyond observed D, and
- at least one causal ablation (local attention / encoder noise / fixed-T training).

5. INTERACTION WITH PROPS 22/23

The claim that Prop 24 “weakens BP” is **too strong as stated**. Standard sparse, local BP intuition and O(D)-like propagation are weakened, but dense-graph BP (or learned dense dependency maps) can still be consistent with fast convergence.  
So a safer statement is: it weakens the *strict sparse local* BP interpretation and is **consistent with dense/global message interactions plus nonlinear global solver dynamics**.  
This is important because Props 22/23 are still compatible if “decoder” is dense, iterative, and learned, not literal sparse BP on a chain factor graph.

6. THE C_step ~ D CLAIM

The six reported values are too sparse and internally noisy to justify “Θ(D)” with confidence:
- Using the `T=99%` interpretation from the raw step ablations gives `C_step = D / T_min`: `{0.67, 0.80, 1.20, 1.60, 2.00, 2.40}` for D={2,4,6,8,10,12}.
- The proposition’s table mixes inferred `T_min` values (`3` for D=2 and D=6) that do not align with the shown 99% criterion in the same run.
- High-encoder-weak regime points are suggestive of growth with `D`, but “proportional” is only partly supported and not stable across all points.

So the current evidence supports “C_step increases with effective difficulty and eventually grows with D at this narrow scale,” not a clean asymptotic `C_step ∝ D` law.

Sources used:
- [bottleneck_depth_scaling.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md)
- [theory_summary.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md)
- [exp_d23_carry_depth_scaling.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d23_carry_depth_scaling.json)