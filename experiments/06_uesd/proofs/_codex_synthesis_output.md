# UESD Strategic Synthesis (Architecture Theorist + Integrity Auditor)

Date: 2026-05-24  
Primary scope: `experiments/06_uesd/proofs`

This document satisfies the six requested sections from `_codex_synthesis_prompt.md` and integrates:

- Web/source findings from `_synthesis_input.md` (9 papers + 12 repo-survey findings)
- Program-internal state in `theory_summary.md`, `bottleneck_depth_scaling.md` (lines 2650-2810), and `_codex_synthesis_prompt.md`
- Latest experimental context provided in user message (D28/D30 updates, in particular D28 L=24 FT and partial/complete D30 runs)

All rho values, proposition numbers, and experiment IDs below are called out explicitly for decision use.

## 1) Signal Extraction (Top 5 signals)

### Signal 1 — Non-monotonic FT rho(D) with a sharp, localizable anomaly is now the highest-resolution state signature
Across FT and VT contrasts, the core dynamical signal is the depth profile:

- FT, D28:  
  `D=2:1.0018`, `D=4:1.0026`, `D=6:1.0024`, `D=8:1.0016`, `D=10:1.0042`, `D=12:1.0039`  
  from `_synthesis_prompt.md` and `bottleneck_depth_scaling.md`.
- Key consequence: D=12 is lower than the D=10 peak; FT rho is non-monotonic and now has direct evidence of a directional peak at D=10 and decline at D=12 (D28 L=24 FT complete, `eta=1.2`, `_codex_synthesis_prompt.md`).

This is materially different from the original monotonic horizon hypothesis and from early quadratic-fits that assumed smoothness. It supports the revised model space in `bottleneck_depth_scaling.md`: Model B (plateau) and Model C (peaked strain) survive; Model A (continued increase) is weakly falsified by the observed D=12 decline.

Why this matters:
- It is not merely a small variance blip (`std`s are explicitly small, ~`0.0001`-`0.0005` in the latest rows).
- It creates a falsifiable discriminant for future experiments (D=14/16) instead of "collect more of the same" rho points.
- It is the key lever against overfitting a single interpretation of training-horizon effects.

### Signal 2 — VT constant-delta hypothesis now has both stability and anomaly in same dataset
From user-updated context and files:
- D28 VT delta-rho values across 5 depths: `[-0.0024, -0.0025, -0.0028, +0.0014, -0.0025]` for `D=2,4,6,8,10`.
- Excluding D=8, mean `-0.0026 +/- 0.0002`; D=8 is a ~20-sigma positive outlier (`+0.0014` above local expectation).
- D30 monotonic trend confirms VT regularization as a smooth function of `T_min`:  
  `T_min=2:0.9992`, `T_min=4:1.0001`, `T_min=6:1.0006`, `T_min=8:1.0017`.

Why this matters:
- Signals coexistence: **stability in four of five depths + a reproducibility bomb at one depth**.
- This mirrors the D28 FT anomaly pattern but with opposite causal direction (negative shift mostly stable), supporting hypothesis that there are two coupled mechanisms: horizon strain (FT) and a depth-agnostic regularization offset (VT), as encoded in the decomposition framing in `_synthesis_prompt.md`.

### Signal 3 — Fixed-point language is no longer the empirical core; readout-stable manifold dynamics are
`theory_summary.md` and `bottleneck_depth_scaling.md` converge on:
- **Prop 28 (readout-stable manifold, not fixed point)**: `rho` can remain near/above 1 while readout still converges fast.
- **D29b / Prop 31**: FTLE decomposition with `lambda_R < 0` and `lambda_null > 0` is now confirmed.
- D28 L=24 FT residual is non-converged (`FP residual 0.109`), while `T_99=5` and high accuracy persist (`99.22%`), again validating "projection/selection for readout" rather than classical global contraction.

Why this matters:
- It directly resolves the paradox that `rho(J)>1` would imply instability under classical Banach criteria.
- It reframes architecture design and diagnostics from "make rho < 1" to "shape orientation and subspace structure", aligning with non-normal stability and FTLE analyses.

### Signal 4 — The most explanatory cross-field correspondence is not a loose analogy, it is a three-way isomorphism
From web papers and repo survey highlights:
- "Recurrent Depth" (arXiv:2502.05171) predicted two-phase signatures (exploration/solution transition, step ~5 in D6), and UESD exhibits step-structured SV/FTLE behavior consistent with that.
- "Belief-propagation" framing (repo survey, UESD as factor-graph BP-like decoding) matches D6 bimodal alignment and the D28/D30 scaling behavior.
- Nishimori/criticality framing predicts readout-relevant critical dynamics while orthogonal directions remain near-neutral/expansive (via temperature-like fixed-point behavior).

Why this matters:
- Three independent literatures (optimization dynamics, probabilistic decoding, and stability theory) converge on the same mechanism: **phase-separated dynamics + late-stage readout consolidation**.
- This gives a publishable conceptual core that is broader than "another recurrent transformer ablation study."

### Signal 5 — Theoretical model is now at the stage where under-identification is the bottleneck, not data sparsity alone
`bottleneck_depth_scaling.md` explicitly reports:
- six FT points + five VT points fit multiple model classes (A/B/C) with similar empirical loss, especially if D=8 remains anomalous.
- Model C (peaked strain) currently has `sigma=0.034` and is effectively near-delta at `eta=1`, which is under-constrained with one point in the peak.
- Constant-delta_rho and D=8 outlier behavior survive, so the decomposition idea still has life, but its parameters are not identified.

Why this matters:
- The next experimental move should not just "add one more rho"; it should produce structural identifiability constraints (orientation/MI/ablation signatures), not just another scalar.
- This aligns exactly with the "Architecture Theorist + Integrity" objective: move from curve-fitting to falsifiable mechanism.

## 2) Pattern Recognition (Cross-source synthesis beyond any single source)

### Cross-domain convergence pattern 1: same mechanism, three equivalent descriptions
Across `_synthesis_input.md`, `_codex_synthesis_prompt.md`, and `theory_summary.md`:
- **Dynamical systems description**: non-normal Jacobian + readout-projected contraction (`kappa = sigma_max / rho > 1` possible, stable readout axes).
- **Learning-theoretic description**: variable-depth training injects a spectrum regularization effect (`Prop 9, variable_t_spectral_stability`), making short-step corrections and long-horizon extrapolation compatible.
- **Coding/inference description**: BP-like iterative decoding on a factorized task code graph (`Props 22-23`, repo survey channel-coding view).

This is stronger than a loose metaphor: the same equation (negative readout-subspace exponents + positive null exponents) is read as non-normal transient growth, decoding-phase transition, and error-space manifold routing, depending on source.

### Cross-domain convergence pattern 2: "edge of stability" + "readout relevance" resolves historical contradiction
`theory_summary.md` notes:
- Banach-style full-state contraction does not hold (`k` near 0.99, high near-criticality),
- yet practical readout convergence is often rapid (`T_99=2-6` across configs, and `D30` monotonic low-lag behavior).

This mirrors `Divisive Normalization stabilizes rho > 1` (paper 4 in web list): normalization can stabilize directional computation even when full linearized maps look risky.  
So the contradiction with high `rho` is not a failure case; it is the target case of directional stability control.

### Contradiction pattern: where strong narratives disagree and force better experiments
- `_synthesis_input.md` includes the **Strain-at-horizon** framing: anomalies near `eta~1`.
- `bottleneck_depth_scaling.md` records VT prediction falsification at D28 L=20 (`rho_VT` predicted >1.004, observed `1.0017`).
- However, FT strain explanation still explains D=10 peak and the D=12 decline better than monotonic alternatives.

Interpretation:
- Single scalar mechanisms are insufficient.
- The same "training-horizon mechanism" likely splits into at least two channels:  
  1) baseline depth scaling, and 2) task-conditioned adaptation of spectral orientation.

This contradiction itself is useful: it pushes us toward interventions on mechanism (e.g., orientation control) rather than only re-measuring `rho(D)`.

### Gap pattern: blind spots are now convergent and high-priority
`_synthesis_input.md` and reviews converge on:
- no multi-task validation,
- no scaling run at bigger `d, V`,
- no information-theoretic observables (MI/Fisher/IB),
- stalled recovery (`Prop 30` strain revisions, `Prop 34` gradient coherence unresolved),
- architecture/task specificity risk.

None of these are technical nuisances; they are all exactly the channels that can test whether this is a genuine computational principle or an arithmetic benchmark artifact.

### Pattern-of-failure pattern: where over-interpretation risk is highest
The most brittle point in current claims is the **D=8 VT positive anomaly** (`+0.0014`) and the historically non-monotonic D=10 peak pattern. Because this point is isolated and the model family treats it as critical either way (artifact vs true kink), any publication claim must not hinge on it as the core theorem.  
It should be treated as a hypothesis probe, not a conclusion.

## 3) The One Experiment (paradigm-shifting, RTX 5090 feasible)

### Proposed experiment ID: **D31-M2T** (the one experiment to run next)

**Goal:** Test the central mechanism directly: "readout-stable manifold + non-normal orientation is causal, not an artifact of addition-only single-task behavior."

**Why this is paradigm-shifting:**  
It simultaneously moves from scalar stability metrics (`rho`) to **causal mechanism inference** by combining:
1) multi-task generalization stress,  
2) architectural causality (normalization ablation), and  
3) mechanistic diagnostics (FTLE/MI/entropy curves + Jacobian geometry).

### Experimental setup
- **Model:** existing UESD architecture (`d=128, V=64`, L=20 baseline), reused code path from D28/D30.
- **Tasks:** train on **two symbolic tasks** jointly: addition and subtraction (or addition + parity-check) with shared encoder/decoder.  
  This targets the blind spot "single-task only."
- **Training schedule:** replicate the D30-style VT regime with **one seed set per condition** (at least one full repeat for this one experiment; use D31 seeds if available as post-check).
- **Conditions:** 3 arms in one run file:
  - **Arm A (baseline):** full model, VT `T_min` in `{4,6,8}` curriculum same as D30.
  - **Arm B (normalization stress):** remove/disable LayerNorm in F_theta (connects directly to paper "Divisive Normalization stabilizes rho > 1" and source signal).
  - **Arm C (sampling robustness):** apply "iteration dropout" (repo-survey priority direction): randomly drop each iteration during training with `p=0.2`, not only truncating to shorter `T`.
- **Evaluation points:** inference at `D in {6,8,10,12}`, depth schedule from D28; fixed horizon checks at 20k steps + checkpoint every 2k.
- **Measurements:** for every arm and depth:
  - `rho`, `k_frob`/`k_mean`, `T_99`, accuracy/WA.
  - `sigma_max` trajectory and `rho` trajectory.
  - per-step FTLE spectrum; top positive/negative mode separation (readout-relevant vs null-space proxy).
  - Info-bottleneck proxy: `I(h_t; input)` and `I(h_t; target)` time curves.
  - Sensitivity to task transfer: accuracy on held-out task with no extra finetune.

### Predictions (from current theory stack)
- **If unified UESD mechanism is real and general:**
  1. Baseline preserves the D28/D30 signature across **both tasks**:
     - FT peaks near D=10 and mild decline at D=12 (`~1.0042 -> ~1.0039` expected order),
     - VT shift remains close to constant (`delta_rho ~ -0.0025` outside any task-specific outlier),
     - `T_99 ~ max(T_min, D_intrinsic)` with `T_99` decoupled from global contraction.
  2. Baseline still shows negative readout FTLE modes and positive null modes (the FTLE decomposition of Prop 31), even when one task transfers poorly in raw token accuracy.
  3. `MI` curves show early exploration phase (high entropy / broad MI) followed by readout-collapse phase, matching "Loop, Think, & Generalize" overthinking threshold behavior and "Thinking Deeper, Not Longer" iterative generalization story.

- **If normalization is causal stabilizer (paper 4 mechanism):**
  - Arm B should lower rho below 1 or raise instability in readout contraction (shorter/less reliable `T_99`, higher WA errors), while still allowing some task memory.
  - This would support a **LayerNorm as required divisive-stabilization mechanism** for UESD.

- **If iteration dropout does not change mechanism positively:**  
  - Arm C with random dropping should fail gracefully (smaller degradation in accuracy than expected from depth truncation alone), supporting robustness interpretation of "iteration as redundant error-correction substrate."
  - If Arm C destroys performance sharply, "robust to arbitrary iteration skipping" may be false, indicating over-reliance on precise step timing.

### Interpretation and decision criteria
- **Prediction pass (strong):** baseline behavior is preserved across tasks and under mechanistic metrics; arm B fails as predicted and arm C either passes or is predictable by an explicit phase-shifted model.  
  Interpretation: the readout-stable manifold + non-normal orientation claim is causal and architecture-relevant.
- **Prediction fail (program-changing):**
  - no transfer between tasks,
  - no directional FTLE split in multi-task setting,
  - all arms collapse similarly once normalization is touched or dropout is introduced.
  Interpretation: current theory is likely a task/scale artifact; shift to explicit architecture-generalization program.

### Why this is feasible on one RTX 5090
- It is one training campaign plus finite-depth evaluations (`D 6-12`, 20k steps style checkpoints) in a regime already used in D28/D30.
- The added cost is in diagnostics (FTLE/MI), not in exploding hyperparameter lattice.
- It replaces another large lattice sweep with one high-information, causally decisive campaign.

## 4) Theory Consolidation (34 propositions -> 4 core principles)

### Core Principle P1 (collapsed from propositions on fixed-point existence, convergence, and coupling):
**Readout-correct trajectories are governed by readout-relevant solvability, not global fixed-point convergence.**  
Relevant propositions: 1-6 (convergence/coupling chain), 28, 31, 32, 6.11.

- Instead of "`rho < 1` => success", use:  
  - dynamics must construct a trajectory that enters the readout-stable manifold,
  - and readout-specific directions must contract fast enough for early reliable decoding.
- This principle subsumes:
  - Theorems 1-3 of convergence_correctness,
  - The fixed-point existence/IFT statements in fixed_point_existence,
  - readout contraction observations in finite_step_convergence + Prop 28.
- It also explains why `T_99` can remain small while `rho` stays near or above 1.

### Core Principle P2 (collapsed from non-normal, spectral, and FTLE components):
**UESD gains computation from structured non-normality: transient expansion first, contraction later.**  
Relevant propositions: 6, 7, 27, 31, 6.7, 6.8, 6.9.

- D6-style bimodal SV/FTLE alignment and `lambda_R < 0, lambda_null > 0` are a single mechanism, not two disconnected observations.
- This principle replaces a strict "small spectral radius" engineering goal with "orientation engineering" in Jacobian products.
- It directly explains:
  - high `kappa` tolerance,
  - strong readout convergence despite near-neutral/expansive full dynamics,
  - and the apparent anomaly handling at depths near horizon saturation.

### Core Principle P3 (collapsed from variable-T theory and VT/FT contrast):
**Depth is a controlled inference-budget controller; variable-T training regularizes the finite-time solver geometry.**  
Relevant propositions: 9, 22.1 (reinterpreted), 31.2 (three-way decoupling), Corollaries in variable_t_spectral_stability.

- `T_min` and effective intrinsic depth are both first-class variables, with `T_99 = max(T_min, D_intrinsic)` (`Prop 32` explicitly).
- This principle absorbs "D30 monotonic with `T_min`" and the D28 FT/VT comparative structure.
- It reframes `rho` as one observable derived from deeper geometry, not the optimization objective itself.

### Core Principle P4 (collapsed from channel-coding, edge-of-chaos, and information-flow framing):
**UESD is iterative self-correction under a channel-like constraint, with near-critical readout computation.**  
Relevant propositions: 22, 23, 6.9, 2.10, plus reservoir- and Nishimori-style references.

- The "factor-graph / BP" analogy is no longer metaphorical once the two-phase FTLE profile is measured.
- Nishimori-like criticality and edge-of-chaos framing gives a natural reason why effective dynamics remain exploratory yet controllable around `rho~1`.
- This principle explains the high readout efficiency and why wrong-attractor risk rises with `||shift||` across distribution shift.

### Core Principle P5 (safety principle, downgraded status):
**Current open claims need explicit separability of universal mechanism vs benchmark artifact.**  
Relevant propositions under repair/low-confidence: 30, 34, and partially 25.

- Keep Prop 25 (pure Banach-style full-state contraction) as a **warning theorem** rather than a claim.
- Keep Strain decomposition as hypothesis-level, not settled fact, until D31-M2T-type evidence resolves model identifiability.
- This principle is not weak theory; it protects integrity by preventing overstatement.

### Suggested proposition folding map (for manuscript preparation)
- Merge propositions on fixed points, correct readout coupling, bounded T, and readout robustness into one chapter:
  - [Convergence stack] = one "readout-first solvability theorem" package.
- Merge full spectral-stability propositions into one "non-normal dynamics theorem" chapter.
- Merge D/T-horizon and VT/FT interactions into one "regularized inference horizon" chapter.
- Park unresolved strain-specific claims in an appendix as hypotheses with explicit discriminative tests.

This reduces a 34-item scaffold to an explainable 4+ hypothesis package while preserving testability and traceability.

## 5) Risk Assessment (single biggest artifact risk and mitigation)

### Biggest risk
**The program may be artifactually measuring a depth-tuned, addition-specific, small-scale decoder behavior, not a task-general computational principle.**

Why this is the largest risk:
- Every core rho and `T_99` signal currently comes from one architecture family and effectively one symbolic family.
- The strongest "novelty" claims (FTLE orientation, non-normality as computation, BP-like behavior) could emerge from dataset geometry alone unless proven under changed tasks and controlled architectural ablations.
- The remaining anomalies (especially D=8 spike patterns) could be seed/task/horizon couplings.

### How to test this risk (operational protocol)
1. **Task axis stress**: in D31-M2T (or immediate variant), add a second task family and check transfer without retraining.
2. **Architectural axis stress**: remove or perturb normalization and compare directional FTLE collapse.
3. **Scale axis stress**: one or two larger `d, V` probes (`d=256, V=128`) at least on a single seed.
4. **Mechanism axis stress**: if only scalar `rho` changes and FTLE/MI patterns collapse, reject "mechanism-level" claims.
5. **Replicability axis**: require D31-M2T consistency on all three arms.

### Interpreting outcomes
- If only task-2 collapses but task-1 holds, the current UESD model is at least task-contingent and needs a bounded-domain claim.
- If LayerNorm ablation removes the mechanism entirely, then mechanism is present but architecture-dependent; publication should be framed as "LN-enabled non-normal readout dynamics."
- If both pass, risk drops sharply and we have a real cross-domain principle.

## 6) Publication Strategy (ambitious, 2-month horizon)

## Thesis for strongest 2-month paper
**Claim proposal:**  
"UESD does not rely on global contraction (`rho<1`) for robustness; instead it exploits **readout-directed non-normal dynamics** plus **variable-depth sampling** to perform iterative decoding with near-critical flexibility in hidden state and stable readout stabilization."

### Paper scaffold (high-impact, 2+ month version)

**Target contribution:**  
- A mechanistic theory of why iterative transformers can be simultaneously expressive and stable by separating readout-relevant and null dynamics (no fixed-point fetish).

**Section plan:**
1. **Problem framing**
   - Why standard spectral-radius narratives under-specify recurrent transients.
2. **Theory package**
   - P1-P4 above as formalized propositions/theorems.
   - Explain the `T_99 = max(T_min, D_intrinsic)` decoupling and its consequences.
3. **Mechanistic diagnostics**
   - Publish per-step FTLE spectra, null/reading direction decomposition, and MI curves.
4. **Empirical core**
   - D28/D29/D30 baseline table + D31-M2T.
   - Emphasize D28 FT peak+decline and D30 VT monotonicity as a hard test.
5. **Cross-field linkage**
   - Explicitly map to DEQ consistency, recurrent depth, BP-style decoding, and stability papers.
6. **Ablation/causality section**
   - LayerNorm removal and iteration dropout as mechanistic falsifiers.
7. **Scope and limits**
   - Explicitly separate proven claims from pending hypotheses.

### Concrete 8-week execution plan
- **Weeks 1-2:** run D31-M2T, collect checkpoints, FTLE+MI+WA logs.
- **Weeks 3-4:** run interpretation + outlier audit (`D=8`, D=10/D=12 replication checks), finalize discriminative claims (model B vs C vs artifact).
- **Weeks 5-6:** draft manuscript theory + methods, include preregistered hypothesis table and confidence statements.
- **Weeks 7-8:** full ablation pass (at least one architecture perturbation + one task-transfer sanity check), revise claims, prep supplemental artifacts.

### Target venue + positioning
- Strong fit to machine learning systems/theory venues that value recurrent dynamics + theory + mechanism (e.g., NeurIPS/ICLR tracks on deep learning theory and dynamics, depending on results depth).
- Position as: "readout-projected dynamical computing in iterative transformers," not "new benchmark on arithmetic."

### Explicit anti-overstatement rule
- Do **not** claim universality until D31-M2T and at least one larger-scale run confirm cross-task behavior.
- Publish first as a **mechanism+diagnostics paper** with preregistered falsifiers:
  - D=8 replication,
  - `T_min` monotonicity under fixed D with MI/FTLE support,
  - causality of normalization and step dropout perturbations.

---

## Decision Log for next research step

1. **Run one experiment now:** D31-M2T (multi-task + normalization stress + iteration dropout + mechanistic diagnostics).
2. **Primary adjudication target:** whether readout-subspace contraction and FTLE directionality are shared across tasks and robust to depth perturbation.
3. **If pass:** proceed with two-pronged publication track (mechanism + scaling/validation note).
4. **If fail:** demote D28/D30 curvature story to architectural artifact report and pivot to architecture- and task-dependent dynamics.
