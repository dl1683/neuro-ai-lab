1. **Complete D28 read (all 12 configs confirmed)**
   - `exp_d28_contraction_ratio.json` contains 12 rows: 6 carry depths (`2,4,6,8,10,12`) × 2 variants (`fixed_t`, `variable_t`) with no missing entries. That matches the complete D28 spec in [`EXPERIMENTS.md`](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/EXPERIMENTS.md) and the D28 section in [`bottleneck_depth_scaling.md`](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md).

2. **Three-regime interpretation: signal or over-fit?**
   - The data do show the claimed three pattern points: steady suppression at D=2,4,6,10; sign flip at D=8; zero at D=12.
   - But this is a fragile regime boundary because it is effectively inferred from a single anomalous point (D=8) plus one terminal point (D=12).  
   - A simpler explanation exists: **depth-near-boundary crossover / numerical-sampling crossover** rather than a fundamental three-phase law. If `T_99 = min(T_min, D_intrinsic)` behavior is active (as reinforced by D30), then one regime change near critical intrinsic depth can look like “anomaly” unless sampled with dense adjacent-depth points and a direct q-measure.
   - So the regime story is plausible, but currently more descriptive than explanatory. (See theory framing in [`theory_summary.md`](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/theory_summary.md), especially Prop 30-34 context.)

3. **Are rho≈1.001–1.004 significant given bars?**
   - Using provided 1σ bars as standard deviations and assuming independent FT/VT runs:
     - D=2 Δρ = -0.0024 with combined σ ≈ √(0.0005²+0.0003²)=0.000583 → |z|≈4.1
     - D=4 Δρ = -0.0025 with combined σ ≈0.000316 → |z|≈7.9
     - D=6 Δρ = -0.0028 with combined σ ≈0.000141 → |z|≈19.9
     - D=8 Δρ = +0.0014 with combined σ ≈0.000224 → z≈6.3
     - D=10 Δρ = -0.0025 with combined σ ≈0.000141 → |z|≈17.7
     - D=12 Δρ = 0.0000 with same-scale bars (not distinguishable from zero at this precision)
   - So if interpreted as stochastic error bars, the nonzero deltas are statistically nontrivial; values near 1 are significantly above/below 1 in most cells.
   - Caution: this is only as good as the estimator semantics. If those bars are from short-run/truncated CI or contain shared-noise components, effective uncertainty is underestimated.

4. **Corrected Prop 34: Δρ(D,T_min) ≈ -A·q(D,T_min)**
   - Conceptually: coherent with the bottleneck/suppression narrative and gives a clean directional test.
   - But as written in the current evidence chain, it is **not yet independently validated**:
     - q appears inferred from observed behavior, not directly measured in the same experiment.
     - A single-scale law with one slope A is underdetermined against depth-dependent mechanism shifts (e.g., D=8 anomaly).
   - This reads as a useful working hypothesis, not a confirmed structural law yet.

5. **Thesis confidence**
   - I would move from **6/10 down to ~5.5/10** right now.
   - Upward evidence: complete 12-config D28 dataset and large-magnitude deltas with small bars; D30 supports Prop 32 monotonicity.
   - Downward evidence: only one depth exhibits the sign anomaly, no adjacent-depth density around anomaly boundary in D28, and Prop 34 remains weakly identified without direct q measurements.

6. **Top 3 remaining risks not covered by planned D31/D32**
   1. `q` is not directly measured—risk of circular inference (model fit to itself).
   2. T-clipping / implementation regime transitions may drive apparent phase boundaries (especially near D=8), so “mechanism regimes” may be numerical artifacts.
   3. Limited external validity: only one setup family is represented in D28/D30; no orthogonal perturbations (architecture/input family/horizon budget/calc precision) to show robustness.

7. **One additional high-leverage experiment**
   - Do a **paired-depth crossover probe** around the anomaly:
     - Run D=7,8,9 (and maybe 10) with fixed/variable schedules,
     - Use matched seeds and paired runs,
     - Measure rho, k, and an **independent empirical q(D,T_min)** in the same pipeline,
     - Report full uncertainty as CIs from a larger seed set and preregister a test of Δρ = -A q with residual checks.
   - This single design would strongly disambiguate: true mechanistic regime change vs D=8 boundary/crossing artifact, and it tests Prop 34 in one shot.