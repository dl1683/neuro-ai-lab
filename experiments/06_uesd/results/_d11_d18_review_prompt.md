# Codex Review: D11 Energy Landscape + D18 Error Function Comparison

## Context
You are reviewing two completed UESD experiments that together reveal the dynamical regime structure of the framework. Read the project context first:

- Read `CLAUDE.md` at project root for overall methodology
- Read `experiments/EXPERIMENTS.md` for all experiment results (especially D11 and D18 entries)
- Read `experiments/06_uesd/results/exp_d11_energy_landscape.json` for D11 raw data
- Read `experiments/06_uesd/results/exp_d18_error_function_comparison.json` for D18 raw data
- Read `experiments/06_uesd/results/_falsification_criteria.md` for the falsification framework
- Read `experiments/06_uesd/results/codex_meta_analysis_part2.md` for prior Codex meta-analysis

## Your Task: Architecture Theorist + Research Integrity Auditor

### 1. Regime Classification
D18 identified three dynamical regimes (CE-dynamics "scattered", E5 "highway", E3 "contractive denoiser"). D11 now adds energy landscape data. Synthesize:
- Does the D11 energy landscape data validate or contradict the three-regime classification?
- CE-dynamics plateaus at high energy (19-31) with nearly constant step sizes. E5 converges to near-zero energy. What does this mean theoretically for each regime?
- CE-dynamics is NOT a fixed-point system but still solves the task perfectly. What mechanism is it using instead?

### 2. The Ballistic Computation Hypothesis
D11 shows CE-dynamics has:
- Nearly constant step sizes (6->4 over 10 steps)
- Most direct paths (ratio 1.08, nearly geodesic)
- Flat energy landscape (2% variation across PCA grid)
- Wider basins than E5

This looks like "ballistic computation" — the model doesn't converge to attractors but instead TRAVERSES state space in a nearly straight line. Questions:
- Is this genuinely a new computation paradigm or just a different way to describe standard neural network forward passes?
- How does this relate to the "parallel computation engine" thesis from D7?
- If computation is ballistic, why would additional steps (T>1) help at all? (This is what D19 will test)

### 3. Implications for D19 Predictions
D19 (step ablation) is currently running. Based on D11+D18 data:
- What do you predict for seq_acc(T=1)/seq_acc(T=10) for CE-dynamics?
- What do you predict for E5?
- What finding would be most surprising and why?

### 4. Cross-Validation Concerns
- D11 uses 2 seeds per track. Is this enough to trust the regime characterization?
- The basin analysis uses cosine similarity clustering. All conditions show 512 basins at threshold 0.95 — every example unique. Is this an artifact of the threshold or real? What would happen with different clustering methods?
- Energy "profile" is measured along the dynamics trajectory. But energy landscape slice uses PCA of perturbed final states from ONE context. Are these measuring the same thing?

### 5. What Questions Should We Ask Next?
Beyond D19/D20/D21 already planned: what experiments would most sharply distinguish the ballistic computation hypothesis from alternatives?

Write your analysis to `experiments/06_uesd/results/codex_d11_d18_review.md`.
