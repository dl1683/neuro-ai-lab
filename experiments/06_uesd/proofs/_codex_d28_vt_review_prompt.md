You are in TESLA MODE — acting as the senior architectural authority reviewing an experimental result that challenges the current theoretical framework.

CONTEXT: Read the following files for full background:
- experiments/06_uesd/proofs/bottleneck_depth_scaling.md (sections: Training Horizon Strain Model, Corollary 30.1, Corollary 30.2)
- experiments/06_uesd/proofs/theory_summary.md (sections 2.24, 2.24b, 2.25)
- experiments/06_uesd/results/exp_d28_contraction_ratio.json

KEY FINDING TO REVIEW:

D28 L=20 VT (D=10, carry depth 10, variable-T training with T in [4,16]) completed with:
- rho = 1.0017 +/- 0.0001
- k = 0.9874 (lowest k in all D28 — most contractive trajectories)
- T_99 = 4
- acc = 95.70%

The Training Horizon Strain model predicted rho_VT > 1.004 (likely 1.005-1.007) based on gradient conflict from unsolvable T_min=4 batches at D=10. The prediction was FALSIFIED.

The full delta-rho pattern (VT minus FT) is now:
- D=2: -0.0024
- D=4: -0.0025
- D=6: -0.0028
- D=8: +0.0014 (ONLY outlier)
- D=10: -0.0025

Constant delta-rho approximately -0.0026 +/- 0.0002 holds at 4/5 depths. D=8 is a 20-sigma outlier.

YOUR TASK — apply ALL of the following:

1. RESULT INTERPRETATION
   - Is the constant delta-rho restoration at D=10 statistically significant given single-seed data?
   - What does the D=8 anomaly most likely represent: (a) seed-dependent fluctuation, (b) V_crit near-saturation effect at D=8 specifically, (c) genuine phase boundary, or (d) something else?
   - How should we interpret the FT rho non-monotonic pattern {1.0018, 1.0026, 1.0024, 1.0016, 1.0042}?

2. THEORY STATUS
   - The strain model was designed to explain BOTH FT non-monotonicity AND the D=8 VT reversal. With the D=10 VT result, the VT prediction is wrong. Does the FT explanation still hold?
   - Is the two-component decomposition ln(rho) = f_complex(D) + g_strain(D/T) still the right framework, or should we simplify?
   - What does the constant delta-rho mean physically? It suggests VT regularization is a simple additive effect independent of the FT dynamics structure.

3. D=8 DIAGNOSIS
   - Given that D=8 is the only outlier: what is the minimum experiment needed to determine if it replicates?
   - Calculate: at 1 seed per config, what is the probability of observing a 20-sigma outlier by chance across 5 depths? (Consider multiple testing.)

4. PREDICTIONS FOR PENDING EXPERIMENTS
   Based on the revised understanding:
   - D28 L=24 FT (D=12, eta=1.2): what should rho be?
   - D28 L=24 VT (D=12): what should delta-rho be?
   - D30 Config D (T_min=8): what should rho be?

5. THREE COMPETING FT MODELS
   With 5 FT points {1.0018, 1.0026, 1.0024, 1.0016, 1.0042} at D={2,4,6,8,10}:
   - Which model class (A: piecewise linear, B: logistic, C: peaked strain) best fits?
   - What does L=24 FT (D=12) discriminate?
   - Are there other model classes we should consider?

6. ANTI-OVERCONFIDENCE CHECK
   - We've been through: quadratic model -> falsified -> strain model -> VT prediction falsified -> constant delta-rho restored. Each revision was data-driven but the pattern of serial falsification suggests our modeling framework may be fundamentally wrong.
   - What would a critic say about this line of reasoning?
   - Is there a simpler explanation that accounts for all 10 data points without invoking strain, solvability, or gradient coherence?

7. PRIORITY DIRECTIVE
   What is the single most important thing to do next? Be specific: not "more experiments" but exactly what experiment, with what configs, testing what specific claim.
