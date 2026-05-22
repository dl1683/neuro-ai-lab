# Codex Evidence Gate: D2-Series Combined Review

You are the Research Integrity Auditor (Persona 4) and Novelty Challenger (Persona 5) reviewing the combined findings from UESD Experiments D, D2, D2b, D2c, and D2d.

## Context

Read these files for full context:
- `experiments/06_uesd/proofs/theory_summary.md` (theoretical framework, Theorems 1-8)
- `experiments/06_uesd/results/exp_d2b_ce_dynamics_sweep.json` (D2b: properly-seeded 5-seed sweep)
- `experiments/06_uesd/results/exp_d2c_stability_analysis.json` (D2c: D7 non-normality analysis)
- `experiments/06_uesd/results/exp_d2d_depth_sweep.json` (D2d: depth-matched encoder multi-seed)
- `experiments/06_uesd/results/exp_d2_controls.json` (D2: original controls)
- `experiments/EXPERIMENTS.md` (all experiment results and findings)
- `experiments/06_uesd/results/codex_d2_review.md` (prior Evidence Gate review — check if findings have been addressed)

## Your Task

Perform a comprehensive evidence review addressing these questions:

### 1) Have the prior review's findings been addressed?
The prior review (codex_d2_review.md) identified:
- Seeding bug → Was it fixed? Did D2b properly control seeds?
- N=1 CE-dynamics → Now N=5, is the success rate robust?
- Single-run depth-matched encoders → Now multi-seed, are they stable?
- Missing D7 diagnostic → Was σ_max/ρ measured? What does it show?
- Binomial uncertainty → Are Wilson CIs reported for success rates?

### 2) Revised claims assessment
Based on ALL results (D through D2d), assess each claim:
a) "UESD dynamics are parameter-efficient for carry-chain computation" — is this now defensible?
b) "CE-dynamics (pure CE, no SC) is more robust than E5 (SC+CE)" — is this confirmed with multi-seed?
c) "SC term traps wrong attractors" — is the mechanism clear?
d) "Non-normality is mild (κ < 1.5)" — what does D7 actually show?

### 3) Statistical rigor
- Are sample sizes sufficient for the claims being made?
- Are confidence intervals appropriate?
- Is the comparison between CE-dynamics, E5, and encoder baselines fair (same seeds, same compute)?

### 4) Theory-experiment connection
- Does the D7 measurement validate Theorem 4 (finite-T stability bound)?
- Does the wrong-attractor rate align with Theorem 4 (convergence ≠ correctness)?
- Is the SC-CE competition mechanism explained by Theorem 8?

### 5) Publication readiness
What is the honest verdict? Categorize each claim as:
- PUBLICATION-READY (rigorous, well-supported)
- NEAR-READY (needs minor additions)
- NOT YET (needs significant additional work)
- OVERCLAIM (not supported by evidence)

### 6) Remaining gaps
What specific experiments or analyses are still missing for a defensible publication?
Be concrete: don't say "more experiments" — say exactly what and why.

## Output Format
Write your review to a file at `experiments/06_uesd/results/codex_d2_combined_review.md`.
Structure it with the 6 sections above. Be honest and rigorous — this is an adversarial review, not a pep talk.
