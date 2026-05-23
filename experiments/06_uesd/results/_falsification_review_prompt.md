# Codex Review: UESD Falsification Test Suite (D19 + D20 + D21)

## Context
You are reviewing the three falsification experiments designed to stress-test the UESD "iterative dynamics as computation" thesis. These are the highest-stakes experiments in the D-series — they were specifically designed to BREAK the thesis if it is wrong.

Read the project context first:
- Read `CLAUDE.md` at project root for overall methodology
- Read `experiments/EXPERIMENTS.md` for all experiment results (especially D19, D20, D21 entries)
- Read `experiments/06_uesd/results/exp_d19_step_ablation.json` for D19 raw data
- Read `experiments/06_uesd/results/exp_d20_bottleneck_sweep.json` for D20 raw data
- Read `experiments/06_uesd/results/exp_d21_wrong_attractor.json` for D21 raw data
- Read `experiments/06_uesd/results/_falsification_criteria.md` for the falsification framework
- Read `experiments/06_uesd/results/codex_meta_analysis_part2.md` for prior Codex meta-analysis
- Read `experiments/06_uesd/results/codex_d11_d18_review.md` for D11+D18 review (which made predictions about D19)

## Your Task: Research Integrity Auditor + Architecture Theorist

### 1. Falsification Scorecard
For each thesis component from the falsification framework, score the evidence:

**T1 (Dynamics compute something):** D19 directly tests this. seq_acc(T=1) vs seq_acc(T=10).
- CE-dynamics ratio: 0.0146. E5 ratio: 0.0000.
- Is this conclusive? What are the caveats?

**T4 (E5 advantage):** D20 tests bottleneck dependence, D21 tests basin stability.
- Does E5 show advantages in basin stability, step robustness, or bottleneck scaling?

**T5 (Parallel not sequential):** D19 per-position carry accuracy data.
- All carry positions converge at similar rates for both regimes. Is this sufficient evidence for parallel computation?

**T6 (Causal structure):** D21 recovery data.
- Neither regime can recover from perturbation. What does this mean for the "causal" claims?

### 2. Cross-Experiment Consistency
The D19 and D21 results paint a very specific picture:
- CE-dynamics: fast-converging, finite-horizon, wide basins, divergent past training horizon
- E5: slow-converging, stable, narrow basins, temporally robust

Check this picture against ALL prior experiments:
- D7 (thinking emergence): does T=3 saturation in D19 match D7's "first stable step" findings?
- D8 (causal carry): does parallel convergence in D19 match D8's parallel resolution findings?
- D10 (adaptive halting): does the ~6-step minimum in D10 match D19's saturation curves?
- D11 (energy landscape): does CE high-T degradation match D11's non-fixed-point finding?
- D17 (reconsideration): does negative recovery in D21 match D17's weak self-correction?
- D18 (error functions): do the regime dynamics match D18's Lyapunov/amplification profiles?

Flag any CONTRADICTIONS — places where a prior experiment's finding is inconsistent with D19/D20/D21.

### 3. The "Finite-Horizon Computation" Discovery
D19 revealed something unexpected: CE-dynamics has a COMPUTE WINDOW.
- T=1: 1.5% (catastrophically bad)
- T=3: 92% (rapid convergence)
- T=5-15: ~100% (sweet spot)
- T=20: 99.8% (onset of degradation)
- T=32: 78% (significant degradation)

Meanwhile E5:
- T=1: 0% (even worse)
- T=5: 99.9%
- T=6-20: 100% (stable plateau)
- T=32: 95.8% (mild degradation)

Questions:
- Is "finite-horizon computation" a genuinely new phenomenon, or is it just what happens when you evaluate a trained RNN at non-training-length sequences?
- Why does CE degrade more than E5 at high T? Propose a mechanistic explanation.
- What would a theory of optimal compute windows look like? Can we predict the window width from training dynamics?
- Does this have practical implications for inference-time compute scaling?

### 4. The Recovery Failure
Both D17 and D21 show the same thing: UESD dynamics cannot recover from perturbation.
- D17: weak self-correction (max 17% at 1/4 corrupted)
- D21: all recovery values negative (more steps make things worse)

This is a potential WEAKNESS of the framework. Questions:
- Is this a fundamental limitation of weight-tied dynamics, or a training limitation?
- Could Langevin noise (D12, currently running) address this?
- What modifications to the training objective might produce recoverable dynamics?
- Is recovery actually needed for the framework to be useful?

### 5. Prediction Assessment
The D11+D18 Codex review predicted CE-dynamics seq_acc(T=1) at "mid-to-high 90%". The actual was 1.5%.
- Why was this prediction so wrong?
- What does this tell us about the difference between "ballistic paths" and "single-step sufficiency"?
- Were any other Codex predictions from prior reviews validated or falsified by D19-D21?

### 6. Updated Thesis Statement
Based on ALL evidence (D5-D21), write a revised thesis statement for the UESD framework that:
- Accurately reflects what has been PROVEN vs what remains HYPOTHESIZED
- Acknowledges the specific failure modes discovered
- Identifies the strongest remaining falsification targets
- Rates overall thesis confidence (0-10 scale with specific justification)

### 7. What to Test Next
Given D19-D21 results, what experiments would be MOST informative for the next phase?
Consider:
- Scaling to harder tasks (longer sequences, more carry depth)
- Training with noise injection (Langevin training, not just inference)
- Multi-step training (train at variable T, not just T=10)
- Testing on fundamentally different tasks (not just addition)
- Larger models — do the regime differences persist at scale?

Rank by expected information gain per compute hour.

Write your analysis to `experiments/06_uesd/results/codex_falsification_review.md`.
