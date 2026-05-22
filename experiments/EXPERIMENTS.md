# Experiments Log

Reverse chronological order. Each entry links to configs, artifacts, and key findings.

---

## 06: Unified Error-Space Dynamics (UESD)

Framework where AI generation happens in continuous embedding space via iterative dynamics, with no softmax collapse. Tests whether self-consistency energy E(s) = ||F_theta(s,c)||^2 produces correct, stable attractors.

### Exp D: Compositional Tasks — Dynamics Necessity (Hard) (COMPLETE — REVISED: SEE D2)
- **Config:** `experiments/06_uesd/exp_d_compositional.py`
- **Purpose:** Test dynamics necessity on compositional tasks that should exceed single-pass encoder capacity. Addresses persistent encoder confound from Exp A/B/C.
- **Tasks:**
  1. **Addition** (base-64 multi-digit): Input interleaves digit pairs `[a0,b0,a1,b1,...]`, output is A+B mod base^half. Carry propagation goes right-to-left — requires O(L) sequential computation. At L=8, carry chains up to 4 deep should exceed 2-layer encoder depth.
  2. **Dedup** (deduplicate + sort): Input has repeated elements (values 1 to V-1), output is unique sorted values + zero padding. Non-bijective mapping requiring counting/grouping, not position-wise routing.
- **Models per task:** E1 UESD (694K), E5 UESD (694K) x2 lambdas {0.1, 1.0}, AR baseline (950K), encoder-only (425K)
- **Architecture:** d=128, heads=4, d_ff=512, V=64, L=8, T=10 (same as Exp A/B/C)
- **Gate:** UESD acc >= 80%, encoder-only acc < 80% (dynamics necessity), E5 WA < 5%
- **Note:** Addition output is half-length + zero padding → token accuracy inflated by trivial zeros. Use seq_acc as primary metric.
- **Results (Addition):**
  | Model | Token Acc | Seq Acc | Margin | Mean Rho | Max Rho | WA Rate | Conv% | Basin |
  |-------|-----------|---------|--------|----------|---------|---------|-------|-------|
  | E1 (embed reg) | 0.5077 | 0.0000 | 6.07 | 0.993 | 1.008 | 1.00 | 100% | 1.0 |
  | E5 (lam=0.1) | 1.0000 | 1.0000 | 9.37 | 0.991 | 1.003 | 0.00% | 99.98% | 1.0 |
  | E5 (lam=1.0) | 1.0000 | 1.0000 | 9.30 | 0.998 | 1.009 | 0.00% | 100% | 1.0 |
  | AR baseline | 1.0000 | 0.9998 | — | — | — | — | — | — |
  | Encoder-only | 0.7316 | 0.0011 | 4.79 | — | — | — | — | — |
- **Results (Dedup):**
  | Model | Token Acc | Seq Acc | Margin | Mean Rho | Max Rho | WA Rate | Conv% | Basin |
  |-------|-----------|---------|--------|----------|---------|---------|-------|-------|
  | E1 (embed reg) | 0.9997 | 0.9983 | 7.74 | 1.065 | 1.709 | — | 0.0% | 0.966 |
  | E5 (lam=0.1) | 0.9993 | 0.9959 | 8.19 | 0.972 | 0.996 | 0.00% | 72.2% | 0.988 |
  | E5 (lam=1.0) | 0.9997 | 0.9980 | 8.28 | 0.981 | 0.993 | 0.19% | 99.96% | 0.994 |
  | AR baseline | 0.9997 | 0.9996 | — | — | — | — | — | — |
  | Encoder-only | 0.9923 | 0.9570 | 6.23 | — | — | — | — | — |
- **Gates (Addition):** Track A INVESTIGATE (E1 fails), **E5 VIABLE (WA=0.00%)**, COMPETITIVE (gap=0.00%), **DYNAMICS NECESSITY CONFIRMED** (encoder=73.16% < 80%, E5=100%)
- **Gates (Dedup):** Track A PASS, E5 VIABLE (WA=0.19%), COMPETITIVE (gap=0.00%), encoder CONCERN (99.23% token, 95.70% seq)
- **Key findings:**
  1. **DYNAMICS NECESSITY CONFIRMED ON ADDITION.** This is the critical result of the entire UESD experiment series. Encoder-only achieves only 73.16% token accuracy (0.11% seq accuracy) on base-64 addition, while E5 achieves 100%/100%. Carry propagation requires sequential computation depth that exceeds 2-layer encoder capacity. The iterative dynamics (T=10 steps) provide the additional computation depth needed.
  2. **E1 fundamentally fails on addition.** CE stuck at exactly 2.08 = ln(64)/2 for all 20K steps despite MSE→0.0002. The 0.1*CE coefficient provides insufficient gradient to learn carry-dependent token mappings. The dynamics converge to fixed points (MSE→0) but those fixed points don't decode to correct tokens. This is a structural limitation of the E1 loss design.
  3. **Phase transition at step ~4000.** Both E5 lambdas show a dramatic CE drop from 2.08 to <0.02 around step 4000 (during lambda warmup). This is a genuine learning phase transition — the model suddenly discovers carry propagation after sufficient SC pressure builds.
  4. **E5 matches AR on addition.** E5 achieves 100% seq accuracy vs AR's 99.98% — E5 is competitive or slightly better. The parallel dynamics (all positions simultaneously) match the sequential AR approach.
  5. **Dedup: encoder confound persists.** Encoder-only achieves 99.23% token accuracy on dedup at L=8 V=64. Self-attention can implement counting/grouping for short sequences. However, encoder-only seq_acc is lower (95.70%) suggesting it struggles with some edge cases.
  6. **E1 works on dedup but not addition.** Unlike addition's carry chain, dedup doesn't require sequential depth — the MSE+0.1*CE loss provides sufficient gradient for learning. E1 dedup: 99.97% token, 99.83% seq.
  7. **E1 dedup has rho > 1 (mean=1.065, max=1.709).** Highly expansive dynamics without SC pressure. Works at this scale due to training coupling but is a theoretical concern. Contrast with E5 lam=0.1 on dedup: rho=0.972 (contractive).
  8. **E5 lam=0.1 dedup convergence concern.** Only 72.17% of examples converge (residual < threshold), but converged_correct_frac=1.0 — all converged examples are correct. SC=0.073 at step 20K is still relatively high. More training or higher lambda may help.
  9. **Lambda selection: 1.0 selected for both tasks** by gate criteria (acc, conv_frac, -rho). Lambda=1.0 achieves higher convergence fraction (100% vs 99.98% on addition, 99.96% vs 72.17% on dedup) at cost of slightly higher rho. Lambda=0.1 gives better contraction (lower rho) but lower convergence fraction, especially on dedup.
- **Encoder confound analysis (cumulative across all experiments):**
  - Copy: encoder-only 100% (trivial — position-wise identity)
  - Reversal: encoder-only 100% (bijective V→V mapping, solvable by attention)
  - Sort: encoder-only 99.99% (attention computes pairwise comparisons → element ranks)
  - **Addition: encoder-only 73.16% (FAILS — carry propagation requires O(L) sequential depth)**
  - Dedup: encoder-only 99.23% (attention can implement counting/grouping at this scale)
  - **Conclusion: Addition is the first and only task where dynamics necessity is confirmed.** The sequential nature of carry propagation (right-to-left dependency chain) creates computational depth requirements that exceed single-pass encoder capacity.
- **Wall time:** 8959s (addition: E1=1330, E5x2=2237, AR=232, Enc=122; dedup: E1=1055, E5x2=2228, AR=347, Enc=229)
- **Artifacts:** `experiments/06_uesd/results/exp_d_compositional.json`
- **Codex Evidence Gate review:** `experiments/06_uesd/results/codex_exp_d_review.md`
  - Verdict: **Not publishable as-is.** Key concerns: (1) Loss confound (E1 uses 0.1*CE, E5 uses 1.0*CE — needs CE-matched ablation), (2) single-seed validity (need 5+ seeds), (3) encoder-only only has 2 layers (need depth-matched 4L/8L control), (4) task design (need carry-chain length sweep). The encoder-only vs E5 gap (73% vs 100%) is real but the mechanism isn't cleanly isolated.
  - **Addressed in Exp D2** (follow-up controls): CE-matched dynamics ablation, depth-matched encoder-only (4L, 8L), 5-seed sweep.

### Exp D2: Additional Controls for Dynamics Necessity (COMPLETE — NUANCED FINDING)
- **Config:** `experiments/06_uesd/exp_d2_controls.py`
- **Purpose:** Address Codex Evidence Gate findings on Exp D. Three controls:
  1. **CE-matched dynamics ablation:** UESD architecture with pure CE loss (no MSE, no SC). Isolates dynamics contribution from loss design.
  2. **Depth-matched encoder-only:** 4-layer and 8-layer encoders. Tests whether depth alone (without weight-tied iteration) suffices.
  3. **Seed sweep:** 5 seeds for E5 lam=1.0 and encoder-only 2L on addition for statistical robustness.
- **Results (Control 1 — CE-dynamics):**
  | Model | Params | Token Acc | Seq Acc |
  |-------|--------|-----------|---------|
  | UESD + pure CE | 694K | 1.0000 | 1.0000 |
  - Dynamics + CE alone achieves PERFECT accuracy. SC term NOT required.
  - Phase transition at step 1-2K (much earlier than E5's step 4-5K).
- **Results (Control 2 — Depth-matched encoder):**
  | Model | Params | Token Acc | Seq Acc |
  |-------|--------|-----------|---------|
  | Encoder-4L | 822K | 0.9994 | 0.9953 |
  | Encoder-8L | 1,615K | 1.0000 | 0.9998 |
  - Both deep encoders learn addition. 8L nearly perfect but still not 100%.
  - 8L uses 2.3x the parameters of UESD for slightly worse accuracy.
- **Results (Control 3 — 5-seed sweep):**
  | Model | Seed | Token Acc | Seq Acc | Phase Transition |
  |-------|------|-----------|---------|------------------|
  | E5 lam=1.0 | 42 | 0.5081 | 0.0000 | NEVER (stuck at 2.08) |
  | E5 lam=1.0 | 137 | 0.5078 | 0.0000 | NEVER (stuck at 2.08) |
  | E5 lam=1.0 | 256 | 1.0000 | 1.0000 | Step 5-6K |
  | E5 lam=1.0 | 512 | 1.0000 | 1.0000 | Step 4-5K |
  | E5 lam=1.0 | 1024 | 1.0000 | 0.9996 | Step 4-5K |
  | **E5 mean** | — | **0.8032** | **0.5999** | **3/5 succeed (60%)** |
  | E5 std | — | 0.2695 | 0.5476 | |
  | Enc-2L | 42 | 0.9981 | 0.9851 | — |
  | Enc-2L | 137 | 0.8676 | 0.0158 | — |
  | Enc-2L | 256 | 0.9998 | 0.9981 | — |
  | Enc-2L | 512 | 0.8861 | 0.1084 | — |
  | Enc-2L | 1024 | 0.9996 | 0.9967 | — |
  | **Enc-2L mean** | — | **0.9502** | **0.6208** | **3/5 succeed (60%)** |
  | Enc-2L std | — | 0.0673 | 0.5111 | |
- **Automated verdict:** DEPTH_SUFFICIENT (both dynamics and deep encoder succeed)
- **Known issue:** Seeding bug — `set_seed()` called inside `train()` after model creation, so model initialization is not controlled by seed. Fixed in code for future runs. These sweep results reflect different random initializations, not true seed control.
- **Key findings:**
  1. **DYNAMICS NOT STRICTLY NECESSARY.** Deep encoders (4L, 8L) can learn addition, so dynamics alone aren't the deciding factor. The original Exp D claim is weakened.
  2. **DYNAMICS ARE MORE PARAMETER-EFFICIENT.** UESD (694K params) achieves 100%/100%, while 8L encoder (1.6M, 2.3x params) achieves 100%/99.98%. Weight-tied iteration provides computation depth without parameter growth.
  3. **SC TERM CAN TRAP WRONG ATTRACTORS.** E5 has 40% failure rate — SC drives convergence (SC→0) before CE can guide to correct attractors. In failed seeds, dynamics converge perfectly (residual=0) but to wrong fixed points (CE=2.08, 0% seq acc). This is Theorem 4 (wrong attractors exist) demonstrated empirically.
  4. **CE-DYNAMICS IS MORE ROBUST THAN E5.** Pure CE + dynamics (no SC) achieves perfect accuracy with no wrong-attractor failure. The SC term, while theoretically principled, creates a competing optimization objective that can lead to premature convergence.
  5. **BOTH MODELS HAVE HIGH VARIANCE.** E5 is bimodal (perfect or total failure). Encoder-only 2L ranges from 1.6% to 99.8% seq acc. Neither is reliable at 20K steps with this architecture.
  6. **PHASE TRANSITION IS INITIALIZATION-DEPENDENT.** Successful E5 runs transition at step 4-6K (after lambda warmup to 1.0). Failed runs never transition despite reaching SC=0.
  7. **REVISED CLAIM:** "Weight-tied iterative dynamics are a parameter-efficient alternative to depth stacking for sequential computation. CE-only training (without SC) is more robust than E5 (SC+CE) for carry-chain tasks."
- **Wall time:** ~12,127s total (Controls 1-3)
- **Artifacts:** `experiments/06_uesd/results/exp_d2_controls.json`

### Exp C: Sort — Dynamics Necessity Test (COMPLETE — ENCODER CONFOUND PERSISTS)
- **Config:** `experiments/06_uesd/exp_c_sort.py`
- **Purpose:** Test whether iterative dynamics add value on a task requiring data-dependent reordering. Sorting is not a fixed permutation like reversal — it requires computing element ranks via global comparison. Directly addresses encoder-only confound from Exp A/B.
- **Models:** E1 UESD (694K), E5 UESD (694K) x2 lambdas, AR baseline (950K), encoder-only (425K)
- **Architecture:** d=128, heads=4, d_ff=512, V=64, L=8, T=10 (same as Exp A/B)
- **Results:**
  | Model | Token Acc | Seq Acc | Margin | Mean Rho | Max Rho | WA Rate | Conv% | Basin |
  |-------|-----------|---------|--------|----------|---------|---------|-------|-------|
  | E1 (embed reg) | 1.0000 | 0.9999 | 8.10 | 0.998 | 1.045 | — | 0.0% | 0.9998 |
  | E5 (lam=0.1) | 0.9999 | 0.9995 | 8.35 | 0.968 | 1.001 | 0.04% | 96.7% | 0.9943 |
  | E5 (lam=1.0) | 0.9998 | 0.9986 | 8.28 | 0.973 | 0.992 | 0.14% | 100% | 0.9956 |
  | AR baseline | 0.9998 | 0.9996 | — | — | — | — | — | — |
  | Encoder-only | 0.9999 | 0.9991 | 7.73 | — | — | — | — | — |
- **Gates:** Track A PASS (100%), E5 VIABLE (WA=0.04%), COMPETITIVE (gap=0.02%), **encoder CONCERN (99.99%)**
- **Lambda sweep analysis (sort):**
  - Lambda=0.1: Best accuracy (0.9999), lowest WA (0.04%), good convergence (96.7%), best rho (0.968). Selected as best_lambda by gate criteria.
  - Lambda=1.0: Perfect convergence (100% converged_frac), all rho < 1 (max=0.992), but slightly higher WA (0.14%) and lower accuracy (0.9998). Lower self-consistency loss (SC=0.0068 vs 0.0495) but this came at cost of accuracy.
  - E1 WARNING: Max spectral radius exceeds 1 (1.045) — some Jacobians are expansive. Despite this, token accuracy is near-perfect (1.0000). This confirms that rho>1 locally does not prevent correct readout when training coupling compensates.
- **Key findings:**
  1. **DYNAMICS NECESSITY NOT CONFIRMED.** Encoder-only achieves 99.99% token accuracy on sorting at L=8, V=64. Self-attention can apparently compute element ranks and produce sorted output in a single encoder pass at this scale.
  2. Lambda=0.1 remains the best lambda across all three tasks (copy, reversal, sort). Consistent with Exp B finding.
  3. Lambda=1.0 achieves perfect convergence (100%) with all rho < 1, confirming that higher SC pressure pushes toward genuine fixed points. But it trades accuracy for stability.
  4. E1's max_rho > 1 is a new observation (1.045). E1 lacks SC pressure, so the dynamics can develop expansive regions. Works at this scale due to training coupling, but is a theoretical concern at larger T or scale.
  5. All UESD variants and AR baseline solve sort comparably — no model has a clear advantage. The task is too easy at this scale.
- **Encoder confound analysis (cumulative):**
  - Copy: encoder-only 100% (expected — position-wise identity)
  - Reversal: encoder-only 100% (bijective V→V mapping, solvable by attention)
  - **Sort: encoder-only 99.99%** (attention computes pairwise comparisons → element ranks → sorted output)
  - Conclusion: L=8, V=64 is small enough that self-attention can implement sorting networks in a single pass. Need tasks with **compositional structure** (carry propagation, deduplication) or **much longer sequences** where single-pass comparison is insufficient.
- **Next steps:** Proceed to Exp D with harder tasks. Two generators already prepared in `shared/data.py`:
  - `addition`: Multi-digit base-V addition with carry propagation (right-to-left dependency)
  - `dedup`: Deduplicate + sort — non-bijective mapping requiring counting/grouping
- **Wall time:** 3943s (E1: 934s, E5x2: 2541s, AR: 302s, Enc: 165s)
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
- `proofs/fixed_point_existence.md` — Fixed-point existence via IFT (rho<1 bootstraps local existence/uniqueness), training-guaranteed approximate fixed points, context-dependence and bifurcation risks.
- `proofs/nonnormal_stability.md` — Non-normal Jacobian analysis: pseudospectrum, Kreiss constant, singular-value vs spectral-radius bounds for finite-T dynamics. Links UESD to forward Euler stability theory. Explains why D5 (not D6) is the right empirical test for finite-step stability.
- **Codex review R1:** `proofs/codex_proof_review.md` — MODERATE overall. CE-to-margin conversion, temperature->0 MI, and non-normal Jacobian issues fixed.
- **Codex review R2:** `proofs/codex_proof_review_r2.md` — Pending (reviews all 6 proofs including new additions).

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
