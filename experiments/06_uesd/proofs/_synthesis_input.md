# Big-Picture Synthesis Input: All Sources

## SOURCE 1: Web Research (9 papers, 2025-2026)

### Most directly relevant:
1. **"Thinking Deeper, Not Longer" (arXiv:2603.21676, Mar 2026)** — Depth-recurrent transformer nearly identical to UESD setup. Three stabilization mechanisms: silent thinking objective, LayerScale, identity-biased recurrence. OOD generalization on compositional tasks. KEY DIFFERENCE: fixed T at training, extrapolate at inference. We use stochastic T during training.

2. **"Loop, Think, & Generalize" (arXiv:2604.07822, Apr 2026)** — Three-stage grokking: memorization → in-distribution → systematic generalization. "Overthinking" failure mode where excess recurrence degrades performance. Our FTLE decomposition could diagnose exactly when overthinking begins (lambda_null directions invade readout space).

3. **Consistency DEQ (arXiv:2602.03024, Feb 2026)** — Applies consistency distillation to DEQs. 2-20x accuracy gains at same function evaluation. REFRAMES: C-DEQ may be implicitly learning to project onto our readout-stable manifold, not reaching the fixed point faster.

4. **Divisive Normalization stabilizes rho > 1 (bioRxiv 2025)** — RNNs remain stable at rho > 1 via divisive normalization. LayerNorm in our transformer serves this function. TESTABLE: ablate LayerNorm, predict rho must drop below 1.

5. **Conservation Law Breaking at Edge of Stability (arXiv:2604.07405, Apr 2026)** — L-layer ReLU nets preserve L-1 conservation laws under gradient flow, broken by SGD. Two regimes: perturbative (spectral crossover) and non-perturbative (mode coupling). Could explain constant-delta_rho mechanistically.

6. **Latent Computing by Bio NNs (arXiv:2502.14337, Feb 2025)** — STRONG MATCH. Five attributes: (1) low-dim computations generate high-dim dynamics, (2) trajectory manifolds have coding redundancy, (3) linear readouts suffice, (4) robust to representational drift, (5) far more neurons needed to predict trajectories than decode readouts. This IS our readout-stable manifold finding from neuroscience.

7. **FTLE for Neural ODE Robustness (arXiv:2602.09613, Feb 2026)** — Direct connection between FTLEs and adversarial vulnerability. Early-stage-only FTLE regularization improves robustness. Suggests our expansive null-space directions are adversarial attack surfaces.

### Supporting:
8. Stochastic depth for adaptive inference (arXiv:2505.17626) — confirms VT robustness benefit from inference angle
9. FTLEs of Deep NNs (PRL 2024) — coherent structures analogy, FTLE ridges divide input space

## SOURCE 2: Research Repo Survey (12 findings)

### Highest priority new directions:
1. **UESD as belief propagation** — Map F_theta to factor graph, derive T_99 from coding theory's decoding threshold. Formula for minimum depth from first principles. Connects to Props 22-23 (channel-coding dynamics).

2. **Nishimori temperature prediction** — Measure rho vs 1/T_min, test rho = tanh(1/(2*tau_eff)). If confirmed, UESD joins 7-substrate universality class. Connects Nishimori framework directly to VT constant-delta_rho.

3. **Grokking-in-depth** — Monitor effective dimensionality during VT training. Phase transition where generalization "turns on." VT may induce transition from memorization basin to generalization basin.

4. **IB trajectory** — Plot I(state_t; input) vs I(state_t; target) across iterations. T_99 may correspond to IB phase transition.

5. **Iteration dropout** — Randomly skip individual iterations (not just truncate). Forces dynamics robust to arbitrary iteration removal. Novel training technique.

### Key cross-domain connections:
- VT = stress-induced mutagenesis (biology)
- k ~ 0.99 = edge-of-stability reservoir computing
- Readout-stable manifold = Waddington canalization (developmental biology)
- Predictive coding: each F_theta step = prediction + correction
- Diffusion model duality: UESD is self-organizing diffusion where noise schedule emerges from dynamics
- Fisher Information Geometry explains constant VT effect

## SOURCE 3: Our Current Empirical State

### Confirmed (high confidence):
- Readout-stable manifold, NOT fixed point (Prop 28)
- FTLE decomposition: lambda_R < 0, lambda_null > 0 (Prop 31)
- T_99 = max(T_min, D_intrinsic) (Prop 32)
- Three-way metric decoupling: k, rho, T_99 are independent (Cor 31.2)
- rho monotonically increases with T_min (D30, 4 configs)
- Constant delta_rho ~ -0.0025 at fixed T distribution (D28, 4/5 depths)
- delta_rho scales with T distribution breadth (D30)

### Weak/stalled:
- Prop 30 (strain model) at 3/10 — THREE times falsified and revised
- Recovery at 2/10 — completely stalled since D25
- Single task (addition only), single architecture, single scale
- Prop 34 (gradient coherence) at 2/10 — no data

### Blind spots identified:
1. No multi-task validation
2. No scaling experiments (d=256, V=256, etc.)
3. Recovery has been abandoned in favor of spectral analysis
4. rho obsession — 3 falsification cycles chasing rho(D) patterns
5. Channel-coding interpretation (Props 22-23) has ZERO dedicated experiments
6. No information-theoretic measurements (MI, Fisher, IB curves)

## WHAT WE NEED FROM THE SYNTHESIS:

1. What is the SINGLE most important experiment we should run next?
   (Not "more rho measurements" — something that changes the paradigm)
2. Which of the 5 priority directions from the repo survey is most falsifiable with our current setup?
3. Is there a unifying theoretical framework that explains ALL our findings with fewer components than our current 19-theorem edifice?
4. What would a paper about this work look like? What's the strongest claim we can make?
5. What's the biggest risk that our results are artifacts of the specific task/architecture/scale?
