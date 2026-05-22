# Experiments Log

Reverse chronological order. Each entry links to configs, artifacts, and key findings.

---

## 06: Unified Error-Space Dynamics (UESD)

Framework where AI generation happens in continuous embedding space via iterative dynamics, with no softmax collapse. Tests whether self-consistency energy E(s) = ||F_theta(s,c)||^2 produces correct, stable attractors.

### Exp C: Sort — Dynamics Necessity Test (RUNNING)
- **Config:** `experiments/06_uesd/exp_c_sort.py`
- **Purpose:** Test whether iterative dynamics add value on a task requiring data-dependent reordering. Sorting is not a fixed permutation like reversal — it requires computing element ranks via global comparison. Directly addresses encoder-only confound from Exp A/B.
- **Models:** E1 UESD, E5 UESD (lambda_1 in {0.1, 1.0}), AR baseline, encoder-only ablation
- **Gate:** E1 sort acc >= 80%, encoder-only acc < 80% (dynamics necessity), E5 wrong-attractor < 5%
- **Artifacts:** `experiments/06_uesd/results/exp_c_sort.json`

### Exp B: Reversal Main Test (COMPLETE - ALL GATES PASS)
- **Config:** `experiments/06_uesd/exp_b_reversal.py`
- **Purpose:** Core test — can dynamics solve non-trivial transformations? Lambda sweep for E5.
- **Models:** E1 UESD (694K), E5 UESD (694K) x4 lambdas, AR baseline (950K), encoder-only (425K)
- **Architecture:** d=128, heads=4, d_ff=512, V=64, L=8, T=10 (same as Exp A)
- **Results:**
  | Model | Token Acc | Seq Acc | Margin | Mean Rho | WA Rate |
  |-------|-----------|---------|--------|----------|---------|
  | E1 (embed reg) | 1.0000 | 1.0000 | 9.22 | 0.999 | 0.00 |
  | E5 (lam=0.0) | 1.0000 | 1.0000 | 10.11 | 0.981 | 0.00 |
  | E5 (lam=0.1) | 1.0000 | 1.0000 | 10.09 | 0.962 | 0.00 |
  | E5 (lam=1.0) | 1.0000 | 1.0000 | 9.82 | 0.975 | 0.00 |
  | E5 (lam=10.0) | 1.0000 | 1.0000 | 9.45 | 0.989 | 0.00 |
  | AR baseline | 1.0000 | 1.0000 | — | — | — |
  | Encoder-only | 1.0000 | 1.0000 | 9.97 | — | — |
- **Gates:** Track A PASS (100%), E5 VIABLE (0% wrong-attractor all lambdas), COMPETITIVE (gap=0%), encoder CONCERN (100%)
- **Lambda sweep analysis:**
  - Lambda=0.0 (pure CE): Perfect token accuracy BUT converged_frac=0.0% (residual mean=0.196). D4 wrong-attractor=0% is VACUOUS (no converged examples exist to be wrong). SC loss decreases from ~80 to ~6 without penalty, but model does NOT reach fixed points — it solves via CE alone without dynamical convergence.
  - Lambda=0.1: Best spectral radius (0.962 mean, 0.996 max). converged_frac=99.99%. The legitimate best lambda by stability metrics.
  - Lambda=1.0: Lowest residual (0.00036). Fully converged (100% converged_frac). Good overall.
  - Lambda=10.0: Worst margin (9.45) and highest rho (0.989). SC overly dominates. Training instability visible.
  - Non-monotonic rho-lambda relationship: 0.1 > 1.0 > 0.0 > 10.0 (lower rho = more contractive = better). Expected in constrained optimization.
- **Key findings:**
  1. Lambda=0.1 is the sweet spot: best contraction (rho=0.962), near-perfect convergence (99.99%), and full accuracy.
  2. Lambda=0 (pure CE) achieves correct tokens but NOT via dynamical convergence. Task is solvable by the encoder+readout pathway without fixed-point structure. The claim "CE creates implicit convergence pressure" is NOT justified by this data.
  3. High SC pressure (lambda=10) hurts: reduces decoder margin, increases rho, causes training instability.
  4. CRITICAL CONFOUND: Encoder-only ablation achieves 100% on reversal. Both copy and reversal are bijective V→V mappings solvable by attention. Dynamics not proven necessary on either task.
- **D5 basin perturbation:** INVALID for all runs (same calibration bug as Exp A, fixed in code post-run). Reported values should be disregarded.
- **Codex Evidence Gate review:** See `results/codex_exp_b_review.md`. Key points: (1) Lambda=0 WA rate is vacuous (converged_frac=0), (2) encoder-only confound is real and severe, (3) need ≥5 seeds for robustness claims, (4) gate logic should use secondary criteria (converged_frac, residual) not just accuracy for best-lambda selection.
- **Wall time:** 5287s (E1: 933s, E5x4: 3994s, AR: 240s, Enc: 119s)
- **Decision table output:** E5 viable (specifically lambda=0.1) → Proceed to harder tasks where encoder-only fails. Required: non-bijective mappings, variable-length output, longer sequences, or latent-constraint tasks.
- **Artifacts:** `experiments/06_uesd/results/exp_b_reversal.json`, `experiments/06_uesd/results/codex_exp_b_review.md`

### Exp A: Copy Smoke Test (COMPLETE - ALL GATES PASS)
- **Config:** `experiments/06_uesd/exp_a_copy.py`
- **Purpose:** Gate — do dynamics converge to correct embeddings at all?
- **Models:** E1 UESD (694K), E5 UESD (694K), AR baseline (950K), encoder-only (425K)
- **Architecture:** d=128, heads=4, d_ff=512, V=64, L=8, T=10
- **Results:**
  | Model | Token Acc | Seq Acc | Notes |
  |-------|-----------|---------|-------|
  | E1 (embed reg) | 1.0000 | 1.0000 | MSE=0.0003, CE=0.0029 |
  | E5 (SC+CE, lam=1) | 0.9999 | 0.9997 | WA=0.01%, margin=9.33, rho=0.98 |
  | AR baseline | 1.0000 | 1.0000 | Loss≈0, trivial for copy |
  | Encoder-only | 1.0000 | 1.0000 | Margin=10.15, copy trivial for encoder |
- **Gates:** Track A PASS (100%), E5 wrong-attractor VIABLE (0.01%), decoder margin PASS, spectral radius PASS
- **Key finding:** E1 originally had 0% accuracy due to readout_proj being untrained (pure MSE loss). Fixed by adding 0.1*CE auxiliary loss. Encoder-only CONCERN is expected — copy is position-wise identity, solvable without dynamics.
- **D5 bug (fixed post-run):** Basin perturbation noise was 320% of state norm, not 10%. Per-element std was `sigma_frac * ||s||_F` but should be `sigma_frac * ||s||_F / sqrt(n_elements)`. E5 basin_stability=0.0 and E1=0.203 are INVALID. Fix in diagnostics.py. D1-D4, D6 unaffected.
- **Wall time:** 2274s (E1: 937s, E5: 995s, AR: 234s, Enc: 108s)
- **Artifacts:** `experiments/06_uesd/results/exp_a_copy.json`

### Exp 0: Information Bottleneck Analysis (COMPLETE)
- **Config:** `experiments/06_uesd/exp_0_bottleneck.py`
- **Purpose:** Mathematical derivation + numerical verification of softmax MI bottleneck
- **What we learned:** Softmax limits MI to log2(V) bits per step via DPI. UESD POC has 21.3x compression ratio (95.3% deficit). Corrected overclaim (storage vs MI) per Codex R1 review.
- **Artifacts:** `experiments/06_uesd/results/exp_0_bottleneck.json`

### Mathematical Proofs
- `proofs/convergence_correctness.md` — When does r->0 imply correct readout? Answer: only with training coupling + sufficient coverage + no wrong attractors.
- `proofs/information_bottleneck.md` — Corrected DPI derivation. Advantage is process (parallel refinement, no premature commitment), not raw capacity.
- `proofs/spectral_contraction.md` — Spectral norm, contraction bounds, basin size analysis. Non-normal Jacobian caveats.
- `proofs/finite_step_convergence.md` — T=10 sufficiency depends on rho. rho<0.8 gives 89%+ reduction; rho>0.9 may need more steps.
- **Codex review:** `proofs/codex_proof_review.md` — MODERATE overall. CE-to-margin conversion, temperature->0 MI, and non-normal Jacobian issues fixed.

### Design Documents
- `design_revision_r3.md` — LOCKED build spec. 2 experiments, 2 tracks, 6 diagnostics.
- `design_revision_r2.md` — Three-track proposal (response to Codex R1).
- `EXPERIMENT_DESIGN.md` — Original Tesla mode design document.
- `codex_review_r1.md` — Core critique: E5 is stopping condition, not error function.
- `codex_review_r2.md` — Narrowing to Round 3 pilot.

---

## 05: DDM Depth Analysis
*(previous experiment series, see 05_ddm_depth/)*

## 04: Criticality Pruning
*(previous experiment series, see 04_criticality_pruning/)*

## 03: Reconsolidation
*(previous experiment series, see 03_reconsolidation/)*

## 02: Sleep Training
*(previous experiment series, see 02_sleep_training/)*

## 01: Grokking Prediction
*(previous experiment series, see 01_grokking_prediction/)*
