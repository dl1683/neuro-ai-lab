# Codex Review: Training Horizon Strain Model

You are reviewing a newly proposed theoretical model for UESD (Unified Error-Space Dynamics). This is an Architecture Theorist + Correctness Engineer review.

## Context

Read these files:
- `experiments/06_uesd/proofs/bottleneck_depth_scaling.md` — the full proof document. The new "Training Horizon Strain Model" section is within Proposition 30 (search for "Training Horizon Strain Model (PROPOSED REPLACEMENT)").
- `experiments/06_uesd/proofs/theory_summary.md` — for overall theory context, especially Prop 30 status.
- `experiments/06_uesd/results/exp_d28_contraction_ratio.json` — the empirical data this model must explain.
- `CLAUDE.md` — for review methodology.

## What Happened

Proposition 30 originally predicted rho(D) follows a quadratic model peaking at D≈4 then declining. D28 L=20 (D=10) fixed-T measured rho=1.0042, the HIGHEST FT rho — completely falsifying the quadratic prediction of rho≈1.000.

The FT rho data is: {D=2: 1.0018, D=4: 1.0026, D=6: 1.0024, D=8: 1.0016, D=10: 1.0042}.

A new "Training Horizon Strain" model has been proposed as replacement. It decomposes ln(rho) into two terms:
1. f_complex(D): complexity-dependent term (peaks at D≈4, declines — the original mechanism)
2. g_strain(eta): horizon strain term where eta = D/T_train (FT) or D/T_min (VT)

The model claims g_strain peaks at eta≈1 (not monotonically diverging), because:
- eta < 1: solvable with margin → strain low
- eta ≈ 1: just saturating horizon → maximum strain
- eta > 1: unsolvable at minimum-T → noisy gradients average out → strain diminishes

## Your Task

1. **Mathematical rigor check:** Is the two-component decomposition well-defined? Is the peaked g_strain function physically motivated or ad hoc? Could the data be explained by simpler models (e.g., piecewise linear, pure phase transition)?

2. **Consistency check:** Does the model consistently explain ALL 9 D28 data points (5 FT + 4 VT)? Does it explain the D30 T_min control data (rho monotonic with T_min)?

3. **Prediction quality:** Are the 5 predictions falsifiable and discriminating? Could other models make the same predictions?

4. **Alternative models:** Propose at least one alternative model that explains the same data. Which discriminating experiment would distinguish them?

5. **The peaked strain function:** This is the most novel (and most questionable) claim. The argument that eta >> 1 produces noisy rather than coherent gradients is plausible but unproven. What would prove or disprove it?

6. **Cross-domain check:** Does the training horizon strain concept have analogues in other fields? (Optimization theory, control theory, physics)

7. **Priority directive:** What is the single most important thing to validate or revise about this model?

Apply the full anti-overconfidence protocol. The model was derived today from 9 data points. Be skeptical.
