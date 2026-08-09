1. Evidence audit against JSON

- Raw config is consistent across all 4 runs: `seed=42`, `L?` represented by `seq_len` 4/8, same architecture (`d_model=128, heads=4, ff=512, enc_layers=2`), `batch_size=256`, `lr=0.0003`, `training_steps=20000`, `trajectory_steps=30`. [exp_d28 JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
- Config summary (`runs`: 4 total) is exactly the 4 requested configs: L=4/8 × fixed_t/variable_t. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
- Claim 1 (constant Δρ)
  - L=4: 1.0018 → 0.9994 (Δ=0.0024)
  - L=8: 1.0026 → 1.0001 (Δ=0.0025)
  - Support status: numerically true for these 2 points, but “constant” is a 2-point pattern, not a tested law. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
- Claim 2 (L=8 VT at criticality)
  - L=8 VT mean ρ = 1.0001, std = 0.0001, readout @ t=30 = 0.9968 (99.68%). [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
  - Support status: not “exactly” 1.0000; distinction from 1.0 is within the reported spread and no confidence interval is provided from multiple seeds. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
- Claim 3 (VT suppresses U-shape at L=8)
  - L=8 FT update means: 17.9412 → 11.5384 (min at step 19) → 12.2166 at step 30 (U-shaped).
  - L=8 VT update means: 18.3547 → 12.3177 at step 30 (monotone decrease in reported window).
  - Support status: strongly supported in L=8 by scalar update trajectories. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
- Claim 4 (readout stability VT = 99.7% vs 83.5% at t=30)
  - L=8 FT readout @ t=30 = 0.8345 (83.45%), L=8 VT readout @ t=30 = 0.9968 (99.68%).
  - Support status: directionally exact; exact percentages are 99.68% and 83.45% in this file, not 99.70% exactly. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
- Claim 5 (Prop 30 confirmed by ρ scales with D)
  - In D28 subset: D=2 (L=4) VT=0.9994 / FT=1.0018; D=4 (L=8) VT=1.0001 / FT=1.0026.
  - Support status: directional support within this 2-point D range and same-experiment pairings; not a strong “scaling law” from 2 points. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1), [EXPERIMENTS D28](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/EXPERIMENTS.md:698)
- Claim 6 (Prop 32 supported)
  - L=8 FT T99=6, VT T99=4; readout @ t=30 VT 99.68% vs FT 83.45%; these align with “readout convergence tightened under VT”.
  - Support status: directionally supported, but proof-level support is indirect unless full D29 decomposition is run. [Proof Prop 32](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2957), [Theory summary](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md:657)

2. Statistical rigor with only 2 depth values

- Two depth points (D=2 and D=4) are mathematically insufficient to infer a monotonic trend, slope, or “constant Δρ” beyond descriptive support.
- No seed replication in this run set (`seed=42` fixed, runs=4). No variance across inits or data shuffles for this conclusion.
- Distinguishability of ρ from 1.0:
  - L=8 VT: mean 1.0001, std 0.0001 (95% interval depends on estimator details, not provided).
  - With no repeated runs, you cannot claim “distinguishable at high confidence.”
  - Combined with non-convergence to a fixed point (all `fp_converged=false`), ρ is also not a clean stability statistic for this measurement regime. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)

3. Confound check (architecture, duration, batch, lr, task)

- Architecture/optimizer/training recipe confounds are mostly controlled within this run set:
  - same `d_model`, `heads`, `ff`, `batch_size`, `lr`, `training_steps`, `train_T`, task choice, and seed. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1)
- Strong remaining confounds:
  - Single seed only; no stochastic error bars.
  - Single task family/sequence family (`addition` implicit from D28 context), and only D=2,4. Results may be regime-specific.
  - Fixed-point residuals are high (~0.098–0.129), so “criticality” interpretation is partially about transient dynamics, not asymptotic linear stability.
  - Fixed vs variable variants are not only “two ρ values”; they are different training objectives over horizon, so VT effects mix objective-shape effects with dynamics shape effects.

4. Proposition consistency and tensions (25,28,30,31)

- Prop 25 (Banach k≈0.4): contradicted by this run set. k_frob is flat near 0.988–0.993, not 0.35–0.45.
  - This is consistent with D28 summary and proposition 28 narrative. [Proof](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2176), [Theory summary](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md:533)
- Prop 28 (readout manifold) is strongly consistent:
  - `fp_residual` not near zero (0.098–0.129), yet readout hits high accuracy in 2–6 steps.
  - Predicted_T99=infinite vs actual readout_T99 finite is exactly the paradox Prop 28 addresses. [JSON](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json:1), [Proof Prop 28 context](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2053)
- Prop 30 (ρ scales with depth): directionally supported in D28 subset, but with only two within-run depth points.
  - Current confidence is “trend present, low power.” [Proof line-level](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2390), [Theory summary calibration pre/post](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md:594)
- Prop 31/Corollary 31.1: broadly consistent with patterns (L=8 FT U-shape and VT monotone), but **not empirically verified in this run set** because directional decomposition is inferential.
  - D29 is needed to test λ_R/λ_⊥ separation directly. [Proof](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2509), [Theory summary](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md:604)

5. Thesis impact update

- “Thesis confidence” should move up, but not to high confidence.
- Recommended update: from 5/10 → 6/10.
- “ρ scaling confidence” should move from 3/10 → 5/10.
- This reflects strong within-run directional signal, but weak inferential power (single seed, two depths, non-converged fixed point). [Proof](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2440), [Theory summary](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md:599)

6. Missing controls that would strengthen/weaken conclusions

- Add seed replication for each configuration (minimum 5–8) and report run-level variance of ρ, k, T99, residual, and readout@t.
- Add D=12, 16, 20 (and maybe 24+) in D28 to test actual depth trend and the alleged constant Δρ.
- Add a fixed-t control with same variable-horizon sampling statistics (or vice versa) to isolate optimization-horizon effects from objective-shape effects.
- Report power-iteration estimator protocol (num starts, restarts, and CI computation); otherwise ρ uncertainty is under-specified.
- Add independent test-set readout trajectories to confirm stability, not just training/eval trajectory traces.
- Run D29 directional FTLE partitioning and report λ_R, λ_⊥ directly (and their confidence). [Proof D29 plan](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2760)

7. VT norm suppression claim (λ_perp inference) — validity assessment

- The inference “VT suppresses λ_perp from U-shaped norms” is plausible but indirect only in D28.
- What is valid: VT lowers update-norm curvature at L=8 from U-shaped to monotone in this run set.
- What is not valid yet: identifying that shape change as specific λ_perp suppression without decomposition into readout-parallel vs readout-orthogonal directions.
- Alternative explanations:
  - VT changes optimization pressure and may induce different effective operating points (`s*` norms differ: FT 1307.63 vs VT 1244.25 at L=8), not only orthogonal-expansion damping.
  - Non-convergence (`fp_residual ≈ 0.098`) means you are observing transient manifold behavior, not full fixed-point stability.
  - Scalar norms can hide cancellation across coordinates; same norm trajectory can arise from different directional dynamics.
- Therefore: the VT suppression claim is a **testable hypothesis**, not a direct measurement-based conclusion, from this JSON alone. [Corollary 31.1 context](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2834), [Theory summary calibration](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md:654)