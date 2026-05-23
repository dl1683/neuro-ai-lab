# Codex Review: D5 Multi-Seed Stability + D6 Random-Matrix Null Model

You are reviewing two completed experiments from the UESD (Unified Error-Space Dynamics) project. Read the project context and experiment files, then provide a rigorous architectural review.

## Context Files to Read
- `CLAUDE.md` (project rules)
- `docs/UNIFIED_ERROR_SPACE.md` (theoretical framework)
- `experiments/EXPERIMENTS.md` (prior experiment history, especially D3-D4)

## Experiment Files to Read
- `experiments/06_uesd/exp_d5_failure_stability.py` (D5 code)
- `experiments/06_uesd/results/exp_d5_failure_stability.json` (D5 results)
- `experiments/06_uesd/exp_d6_random_matrix_null.py` (D6 code)
- `experiments/06_uesd/results/exp_d6_random_matrix_null.json` (D6 results)
- `experiments/06_uesd/exp_d17_reconsideration.py` (D17 NEW - design review only, not yet run)
- `experiments/06_uesd/exp_d18_error_function_comparison.py` (D18 NEW - design review only, not yet run)

## What D5 Tests
Multi-seed validation of D4's two stability mechanisms. 5 seeds x 2 tracks (E5 + CE-dynamics) = 10 runs.

D5 KEY RESULTS:
- 10/10 runs SUCCESS
- E5 "highway": lyap=0.059+/-0.003, align=0.808+/-0.042, amp=1.81x, o/s=1.067 (always >1.0)
- CE-dyn "scattered": lyap=0.188+/-0.022, align=0.595+/-0.069, amp=6.78x, o/s=0.923 (always <1.0)
- Ranges DO NOT overlap on any metric
- E5 CV ~5% vs CE-dyn CV ~12% (E5 tighter clustering)
- CE-dyn seed 1024 outlier: lyap=0.228, amp=9.88x

## What D6 Tests
Whether the observed Jacobian cancellation (718x conservatism from D3b) is LEARNED or merely a statistical property of multiplying random matrices with similar spectra. Three null models:
- A: Isotropic (all SVs = sigma_max, random rotation)
- B: Matched spectrum (actual SVs, random rotation)
- Comparison: actual product sigma vs null models

## Review Checklist

### 1. Correctness Engineer
- Are the D5 cross-seed statistics computed correctly?
- Is D6's null model methodology sound? Are the random orthogonal matrices generated correctly (QR decomposition with sign correction)?
- Does D6's conservatism ratio use the right denominator (mean actual vs mean null)?
- Is the "fraction below actual" statistic meaningful?

### 2. Research Integrity
- Do D5's claims of "non-overlapping regimes" hold up? Are the ranges computed correctly (min/max across seeds, not mean+/-std)?
- Can D5's CE-dyn seed 1024 outlier be explained or does it threaten the regime separation claim?
- Does D6's comparison actually test what it claims? Could there be a confounder?
- Is 200 null trials sufficient for the statistical claims?
- Are 8 Jacobian samples per model sufficient?

### 3. Architecture Theorist
- What do D5's tight E5 clusters vs loose CE-dyn clusters tell us about the loss landscape geometry?
- If D6 shows cancellation is statistical (not learned), does this undermine the "two mechanisms" narrative?
- If D6 shows cancellation IS learned, what does this imply for the dynamics?
- How should D5+D6 findings inform the remaining experiments (D7-D16)?

### 4. Missing Angles
- What controls are missing from D5? (e.g., untied weights, different architectures)
- What additional null models should D6 include?
- Are there cross-experiment connections between D5+D6 and D7-D16 that should be exploited?

### 5. D17 Design Review (New Experiment)
D17 "Reconsideration Capacity" is a NEW experiment testing whether UESD dynamics can self-correct from wrong intermediate states. Three phases:
1. Answer Injection: corrupt converged state with wrong token embeddings, run extra dynamics steps, measure recovery
2. Cross-Example Transplant: swap position state between examples with different answers, test if dynamics follow context or transplanted state
3. Error-Correcting Capacity: simultaneously corrupt 1-4 positions, measure recovery as function of corruption count

Review D17's code for:
- Correctness: is the corruption methodology sound? Is corrupt_state_at_position using the right embedding injection?
- Methodology: is the cross-example transplant a fair test? Should the extra K=20 steps use additional context or just existing context?
- Controls: what controls are missing? (shuffled context? random state injection?)
- Predictions: are the stated predictions reasonable given D5 findings?
- Distinctness from D8 (carry surgery) and D11 (basin mapping)

### 6. D18 Design Review (New Experiment)
D18 "Error Function Comparison" compares E3 denoising and E5 self-consistency error functions on the same dynamics architecture. Three tracks:
1. CE-dynamics (baseline)
2. E5: CE + lambda * ||F(s_T,c)||^2
3. E3: CE + lambda * denoising_loss (corrupt s_T, use dynamics to denoise)

E3 uses a SigmaConditioner module to condition the dynamics on noise level.

Review for:
- Is the denoising training loop correct? Does detaching s_T before corruption prevent gradient leakage?
- Is one-step denoising via model.dynamics() sufficient, or should it use multiple steps?
- Does the sigma conditioning (additive embedding) interact correctly with the dynamics layer?
- Is the log-uniform sigma sampling appropriate?
- Should E3's denoising target be the pre-detach s_T or something else?

### 7. Untied Weights Control
UntiedUESDModel in shared/model.py lacks dynamics_step() API. Should we:
a) Add a step-index-aware dynamics_step() to UntiedUESDModel?
b) Create a wrapper that tracks step count?
c) Write a separate experiment script that handles the different API?
Which D-series experiments MOST need the untied-weights control to strengthen their claims?

### 8. SC Loss Consistency Bug (FIXED)
We discovered that D7/D15/D16/D17/D18 and shared/training.py used `.pow(2).sum(dim=-1).mean()` for SC loss (sum over d_model, mean over B,L), while D4/D5/D6/D8/D9 used `.pow(2).mean()` (mean over all dims). The sum version is d_model=128x larger, giving 128x stronger SC regularization. We fixed D15-D18 and shared/training.py to use `.mean()` (matching D4/D5 baseline). D7 was already running and could not be fixed mid-flight — its E5 results will have 128x stronger SC. CE-dynamics results are unaffected.

Question for Codex: Does this 128x SC strength difference fundamentally change D7's E5 model behavior, or is the qualitative result (thinking emergence) robust to loss scale?

### 9. Priority Directive
What is the single most important experiment or analysis to run next, given D5+D6 results?

Output your review as structured markdown with clear verdict per section: PASS, NEEDS FIXES, or CONCERNS.
