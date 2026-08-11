# Unified Error-Space Dynamics (UESD): Eliminating the Softmax Collapse

> **CURRENT VERDICT — D40 NEGATIVE:** The tested self-consistency energy `E(s)=||F_theta(s,c)||^2` does not produce correct decoded fixed points. The trained systems behave as finite-time transient solvers: cross-entropy produces correct states inside the trained compute window, while stronger self-consistency lowers residual and drives convergence toward wrong decoded attractors. The defensible surviving result is D22-style variable-T k-suppression, not correct fixed-point convergence. See `STATUS.md` and `experiments/EXPERIMENTS.md`.

**Status:** Canonical UESD synthesis. Sections 1-8 preserve the original proposal (2026-05-22) as the historical statement of the thesis; sections 9-10 record the empirical verdict that closed it.
**Author:** Devansh (devansh@svam.com)
**Original date:** 2026-05-22 - **Verdict recorded:** 2026-08-09

---

## 1. The Problem: Softmax as Geometric Collapse

In standard autoregressive generation, the mapping from internal state to output
constitutes a three-stage geometric collapse:

```
h_t ∈ R^d  →  softmax(W·h_t) ∈ Δ^{V-1}  →  sample x_t ∈ {1,...,V}  →  Embed(x_t) ∈ R^d
```

Each stage destroys representational capacity:

| Stage | From | To | What's Lost |
|-------|------|----|-------------|
| Linear projection | R^d (unconstrained) | R^V (logits) | Nothing yet — linear map |
| Softmax | R^V | Δ^{V-1} (simplex) | Sign, scale, rotational DOF. All structure reduced to relative ranking. |
| Sampling | Δ^{V-1} | {e_i} (one-hot) | Entire distribution. Uncertainty = 0. |
| Re-embedding | {e_i} | R^d | Context-specific meaning of h_t replaced by generic embedding of token x_t. |

**The critical split:** The model *thinks* in R^d but is *evaluated* on Δ^{V-1}.
The cross-entropy loss

```
L = -log softmax(W·h_t)[x*_t]
```

shapes the loss landscape in simplex geometry, not representation geometry.
The gradient must flow backward through softmax and W, creating an *indirect*
coupling between what the model learns to represent and what it's rewarded for.

**The bottleneck as circuit failure:** Every autoregressive step forces the full
state h_t through a single discrete token. This is the information-theoretic
analogue of the bridge collapse studied in circuit viability: the pathway
exists, but its bandwidth is artificially crushed to log2(V) bits per step.

---

## 2. Core Proposal: One Space, One Error, No Collapse

### 2.1 Axioms

**A1 (Unified Space).** There exists a single continuous space S = R^d in which
internal representations, output representations, and the error function all live.

**A2 (No Discrete Bottleneck).** At no point during the generative process does
the state pass through a discrete space. Token readout is post-hoc projection,
not part of the dynamics.

**A3 (Single Error Principle).** The function that measures "how good is this
state" during inference is the *same* function (or from the same family) as the
training loss. Thinking well and generating well are the same optimization.

**A4 (Continuous Thinking–Generating Continuum).** "Thinking" and "generating"
are not distinct phases with different mechanisms. They are regions of the same
trajectory through S, distinguished only by the local geometry of the error landscape.

### 2.2 System Definition

A **Unified Error-Space System** is a tuple (S, F_θ, E_φ, R) where:

- **S = R^d** — the state space
- **F_θ: S × C → S** — a learned dynamics operator (C = context space)
- **E_φ: S × C → R** — an energy function
- **R: S → V*** — a readout function mapping states to token sequences (used only for human-facing output)

**Generation** is a trajectory s_0, s_1, ..., s_T in S:

```
s_0 = Encode(prompt)
s_{t+1} = s_t + F_θ(s_t, c)          [learned dynamics]
       OR
s_{t+1} = s_t - η∇_s E_φ(s_t, c)    [energy-based dynamics]
       OR
ds/dt = F_θ(s(t), c)                  [continuous-time / Neural ODE]
```

**Termination** when the system converges:
```
||F_θ(s_t, c)|| < ε    (dynamics stabilize)
   OR
|E_φ(s_t, c) - E_φ(s_{t-1}, c)| < δ    (energy plateaus)
```

---

## 3. The Error Function

This is the load-bearing element of the proposal. The error must:
1. Live in S (not in Δ^{V-1})
2. Be meaningful for both intermediate states (thinking) and final states (output)
3. Not require discrete targets during training

### 3.1 Candidate Error Functions

**E1: Embedding-Space Prediction (Baseline)**
```
E(s, c) = ||s - s*||²
```
where s* = Encode(target). Simple, but collapses the target to a single point.

**E2: Contrastive Energy**
```
E(s, c) = -sim(s, s⁺) + log Σ_j exp(sim(s, s⁻_j))
```
InfoNCE in S. Pushes the state toward good continuations and away from bad ones.
No single "correct" target — the space of good outputs has volume.

**E3: Denoising Score in S**
```
E(s, c) = ||s - D_θ(s + σε, c, σ)||²
```
Learn to denoise in representation space. Connected to diffusion, but the forward
process (noise) and reverse process (denoising) both live in S.

**E4: Predictive Coding Energy**
```
E(s, c) = Σ_l ||s_l - g_l(s_{l+1})||²
```
Each hierarchical level predicts the level below. Error signals propagate in the
same space as representations. Most biologically grounded. Connects to free energy
principle: perception and generation both minimize prediction error in S.

**E5: Self-Consistency Energy (most novel)**
```
E(s, c) = ||F_θ(s, c)||²
```
The energy IS the magnitude of the dynamics. A state is "good" when the system
has nothing left to change. Training objective: make the dynamics converge to
states that, when read out, produce correct outputs.

Full training loss:
```
L = λ_1 · ||F_θ(s_T, c)||²  +  λ_2 · L_readout(R(s_T), y*)
```
where L_readout is a weak token-level loss applied only at the final state,
ensuring the dynamics converge to something useful.

### 3.2 Why E5 Was Judged Most Aligned With The Core Insight (proposal-era judgment; falsified by D40 — see Section 9)

In E5, "thinking well" literally means "the dynamics are still active" and
"generating well" means "the dynamics have converged." There is no separate
error for thinking vs. generating — the magnitude of the update IS the error.

The training signal says: "learn dynamics that, when they converge, produce
states whose readout matches the target." The model is free to use whatever
intermediate trajectory it wants — the only constraint is the endpoint.

---

## 4. The Thinking–Generating Continuum

### 4.1 Phase Characterization

Along the trajectory s_0 → s_T, the system naturally exhibits phases:

| Phase | Signature | Interpretation |
|-------|-----------|----------------|
| **Exploration** | ||F_θ(s_t)|| large, direction varies | Broad search over possible continuations |
| **Convergence** | ||F_θ(s_t)|| decreasing, direction stabilizes | Committing to an output region |
| **Fixation** | ||F_θ(s_t)|| ≈ 0 | Output determined, ready for readout |

**No explicit phase boundaries.** The system doesn't switch modes. The energy
landscape shapes the trajectory, and regions of S naturally correspond to
different phases. This is analogous to phase transitions in physical systems.

### 4.2 Temperature as Exploration Control

Optional: inject noise to control exploration depth:
```
s_{t+1} = s_t + F_θ(s_t, c) + √(2τ(t)) · ε_t,  ε_t ~ N(0, I)
```
- High τ(t): Langevin dynamics, broad exploration (deep thinking)
- Low τ(t): near-deterministic, fast convergence (direct generation)
- τ(t) can be learned, annealed, or adapted per-query

This provides a principled mechanism for adaptive compute: hard problems get
more exploration steps, easy problems converge quickly.

### 4.3 "Going Between States"

The user's original intuition. Formally: the trajectory can revisit regions
of S, moving between partially-converged states. This happens naturally when:
- The energy landscape is multi-modal (multiple valid outputs)
- The dynamics include momentum or noise
- The system needs to "reconsider" (leave a local minimum for a better one)

This is structurally different from autoregressive backtracking (which requires
generating special tokens). Here, reconsidering is just the natural dynamics
of a continuous system navigating a complex energy landscape.

---

## 5. Token Readout (Post-Hoc Projection)

When human-readable output is needed:

**Nearest-neighbor readout:**
```
x = argmin_v ||s_T - e_v||²
```

**Soft readout (differentiable, for training):**
```
p(x = v | s_T) = softmax(-||s_T - e_v||² / τ)
```

**Sequence readout:** For multi-token outputs, options include:
- Multi-head readout: different subspaces of s_T encode different positions
- Iterative readout: unroll s_T into a sequence via a lightweight decoder
- Chunked readout: s_T encodes a fixed-length chunk, chain multiple trajectories

**Key principle:** Readout is NOT part of the generative dynamics. The model
never "sees" its own tokens during generation. It operates entirely in S.

---

## 6. Training Strategies

### 6.1 Target-State Regression
```
s* = Encode(y*),  L = ||s_T - s*||² + λΣ_t ||s_t - s*||²
```
Direct but assumes a single correct embedding per target. Intermediate
supervision (the sum term) prevents trajectory divergence.

### 6.2 Denoising in S
```
s* = Encode(y*),  s_noisy = s* + σε,  L = ||F_θ(s_noisy, c) - s*||²
```
Single-step training even though inference is multi-step. Proven scalable
via diffusion model literature.

### 6.3 Contrastive Learning
```
L = -log [exp(sim(s_T, s⁺)) / Σ_j exp(sim(s_T, s⁻_j))]
```
No explicit target state. The model learns to reach states that are similar
to good continuations and dissimilar to bad ones.

### 6.4 Self-Consistency + Weak Readout (recommended at proposal time; this is the strategy D38-D40 tested and closed negative)
```
L = λ₁||F_θ(s_T, c)||² + λ₂ L_CE(R(s_T), y*) + λ₃ Σ_t ||s_{t+1} - s_t||²
```
- Term 1: convergence loss (dynamics should stabilize)
- Term 2: readout loss (converged state should decode to correct tokens)
- Term 3: path smoothness (trajectories shouldn't oscillate)

This is the most aligned with the core philosophy: the model is trained to
produce smooth trajectories that converge to useful states, with token-level
supervision applied only at the endpoint through a lightweight readout.

---

## 7. Architectural Sketch

```
┌────────────────────────────────────────────────────┐
│                   STATE SPACE S = R^d               │
│                                                     │
│   s_0 ──F_θ──→ s_1 ──F_θ──→ ... ──F_θ──→ s_T     │
│    ↑                                       │        │
│  Encode(prompt)                        Readout(s_T) │
│                                            │        │
│                                         tokens      │
│                                                     │
│  F_θ = TransformerBlock(s, context)                 │
│      - Self-attention over state dimensions         │
│      - Cross-attention to encoded context           │
│      - Residual: s_{t+1} = s_t + F_θ(s_t, c)      │
│                                                     │
│  E_φ(s) = ||F_θ(s, c)||²  (implicit energy)        │
│                                                     │
│  No softmax in the loop. No token sampling.         │
│  No discrete bottleneck. One space. One error.      │
└────────────────────────────────────────────────────┘
```

F_θ can be implemented as:
- A weight-tied transformer block (same parameters at every step, like DEQ)
- A set of blocks with shared + step-specific parameters
- A continuous-depth network (Neural ODE with adaptive stepping)

The context encoder can be a standard transformer — the innovation is in the
generative dynamics, not the context processing.

---

## 8. Connections to Existing Work

### 8.1 Closest Existing Systems

| System | Year | What It Does | What UESD Adds |
|--------|------|-------------|----------------|
| **COCONUT** (Meta FAIR) | 2024 | Feeds hidden states back as continuous "thoughts," bypassing tokens during reasoning | Still uses softmax for final output. Explicit phase boundary between thinking and generating. UESD eliminates both. |
| **MARCOS** | 2025 | Markov chain of continuous thoughts with decoupled thinking-speaking stages. +8.66% over continuous baselines, 15.7× speedup. | Explicitly DECOUPLES thinking from speaking — the opposite of UESD's unification. |
| **Soft Thinking** (NeurIPS 2025) | 2025 | Generates "concept tokens" as probability-weighted mixtures instead of hard argmax during reasoning. Training-free. | Bolt-on decoding strategy, not architectural. UESD builds the unification into the architecture itself. |
| **STAR-LDM** | 2026 | Pauses AR generation, runs diffusion planning in continuous sentence-embedding space, resumes AR. 70%+ coherence win rates. | Maintains hard stop-think-generate boundary. UESD has no boundary. |
| **Meta LCM** | 2024 | Eliminates tokens entirely. Generates sentence-level "concepts" via diffusion in SONAR embedding space. Zero-shot cross-lingual. | Sentence-level, not token-level. No error-space framing — generation is diffusion, not self-consistent dynamics. |
| **LangFlow** | 2026 | First continuous diffusion LM to rival discrete. Embedding-space flow matching. PPL 30.0 on LM1B. | Pure generation mechanism — no thinking component. Error is denoising, not self-consistency. |
| **CLLMs** (ICML 2024) | 2024 | Maps random n-token initialization to same fixed-point result as AR decoding via Jacobi iteration. 2.4–3.4× speedup. | Fixed-point convergence in DISCRETE token space. UESD does fixed-point in continuous embedding space. |
| **DEQ** (Bai et al.) | 2019 | Finds fixed points of implicit layers. 88% memory reduction, competitive on WikiText-103. | Never combined with continuous text generation. Fixed-point finding is internal only; output still uses softmax. |
| **EDLM** (ICLR 2025) | 2024 | Energy-based diffusion for text. Full-sequence energy function using pretrained AR models. Captures inter-token correlations. | Operates in DISCRETE token space. UESD operates in continuous embedding space. |
| **CoDAR** | 2026 | Identifies token "rounding" (continuous → discrete projection) as primary bottleneck in continuous diffusion LMs. | Validates UESD's core diagnosis: the discrete projection IS the problem. But CoDAR patches rounding; UESD eliminates it. |
| **vMF Loss** (Kumar et al.) | 2018 | Replaces softmax with von Mises–Fisher loss in continuous embedding space. 2.5× training speedup. | Applied only to seq2seq translation, not to unified thinking-generating architectures. |

### 8.2 Theoretical Frameworks

| Framework | Relationship | Key Difference |
|-----------|-------------|----------------|
| **Predictive Coding** (Rao & Ballard, 1999; Millidge et al., 2022) | Same-space prediction errors at each hierarchical level. Closest theoretical ancestor. | Never applied to language generation at scale. UESD is a constructive realization for generation. |
| **Free Energy Principle** (Friston, 2010) | Unifies perception and action under single variational free energy objective. | Most general theoretical framework. UESD = specific constructive architecture for language. |
| **Nishimori Thesis** (from this lab's research) | Systems at criticality achieve optimal inference via ρ = tanh(1/2) in continuous error space. Nishimori identity: avg confidence = avg accuracy. | Provides the PHYSICS grounding: softmax collapses a system away from criticality; the proposal argued UESD would keep it there (not tested). See §8.3. |
| **CTI Law** (from this lab's research) | Networks at criticality behave as if performing Gaussian discrimination in d_eff ≈ 1–2 dimensions. SOC → critical point → Gaussian fluctuations → LDA is optimal. | Implies the network is ALREADY doing continuous discrimination internally. The proposal characterized softmax as an artificial post-hoc collapse (untested). |

### 8.3 Physics Grounding: The Nishimori Connection

From this lab's existing research (CTI universal-law thesis and Nishimori cross-domain thesis — external sibling-repository provenance; historical motivation, not tested here and not UESD evidence artifacts):

The Nishimori identity establishes that at the critical point of inference:
- Average confidence = average accuracy (perfect calibration)
- This holds in CONTINUOUS error space at βJ = 1/2
- Seven independent substrates confirm ρ = tanh(1/2) ≈ 0.462 with CV = 1%

**Connection to UESD:** Softmax forces the system OFF the Nishimori manifold.
The simplex Δ^{V-1} is a different geometry from the Fisher-Rao manifold where
optimal inference lives. By keeping generation in the same continuous space as
thinking (R^d with its natural Riemannian metric), the proposal hypothesized that UESD would allow the system to
remain at criticality — where, per the untested proposal-era argument, the Nishimori
identity would guarantee optimal calibration and the CTI law efficient discrimination.

The basin dynamics literature (grokking as phase transition, mode connectivity
via low-loss paths) describes exactly what UESD trajectories would look like:
continuous navigation through energy basins, with thinking = trajectory through
high-energy regions and generation = convergence into basins.

### 8.4 Cross-Domain Evidence (from Open Exploration archive)

| Domain | Mechanism | UESD Connection |
|--------|-----------|-----------------|
| **DNA error correction** | Three-layer correction in same biochemical channel. 10^-10 error rate without discrete bottleneck. | Error correction and signal processing share the same continuous space — same principle as UESD. |
| **Morphogenesis** (Levin) | Bioelectric patterns specify target anatomy. Cells minimize divergence from target in continuous voltage space. No discrete "organ selection." | Generation without discrete collapse. Closest biological precedent for UESD. |
| **Thermodynamics** | Landauer's principle: kT·ln(2) cost per bit erased. Discrete collapse = information erasure = thermodynamic cost. | No collapse → no erasure → thermodynamically cheaper generation. Brain's 20W efficiency: no discrete token bottleneck. |
| **Information geometry** | Fisher-Rao metric defines natural geometry of probability space. Softmax forces onto simplex = distorts this geometry. Flat minima (good generalization) = low curvature. | Softmax introduces high curvature at token boundaries → sharp minima → poor generalization. Continuous error space preserves natural metric. |
| **Stochastic resonance** | Optimal noise level maximizes information transfer in nonlinear systems with thresholds. | UESD's Langevin dynamics (§4.2) exploit noise for exploration. Discrete token space can't leverage noise the same way. |
| **Edge of chaos** | Criticality maximizes sensitivity, dynamic range, information storage simultaneously. | Softmax forces away from criticality (toward order at low T, chaos at high T). The proposal hypothesized continuous dynamics could maintain criticality (not tested here). |

### 8.5 The Novelty Gap

The literature has every PIECE of UESD. Nobody has the SYNTHESIS:

1. **No phase boundary between thinking and generating.** COCONUT, MARCOS,
   STAR-LDM all maintain explicit phase switches. UESD eliminates the boundary.

2. **Error-space framing for generation.** EDLM and CLLMs don't frame generation
   as error minimization in a continuous space shared with reasoning.

3. **No softmax at ANY point** (a design property of the proposal, not a demonstrated advantage). Every existing system uses softmax somewhere.
   vMF Loss (2018) eliminated it for output; Soft Thinking avoids hard argmax
   during reasoning; no system has eliminated softmax throughout.

4. **DEQ fixed-point + continuous embedding space for language generation.**
   DEQs exist. Continuous text generation exists. The combination is unexplored.

5. **Score/energy landscape as shared substrate for reasoning AND generation.**
   Score-based models work in latent spaces for images (LSGM). Energy functions
   work for discrete text (EDLM). Score matching in continuous embedding space
   where the gradient simultaneously represents reasoning direction and
   generation direction has not been proposed.

6. **Physics grounding via Nishimori criticality.** No existing continuous
   generation system has a statistical mechanics argument for WHY continuous
   error space should be preferred. The Nishimori connection provides this.

---

## 9. Empirical Verdict: The Evidence Arc (Exp 0 - D40)

The proposal above was tested across 40+ experiments in 2026-05 (full chronology: `experiments/EXPERIMENTS.md`; run ledger: `experiments/ledger.jsonl`; theory catalog: `experiments/06_uesd/proofs/theory_summary.md`).

1. **Kill chain (Exp 0/A/B/C):** copy, reversal, sort all succeed — but so does an encoder-only ablation. The early task suite could not discriminate the thesis.
2. **Addition (Exp D, D2 series):** base-64 carry-chain addition is the first task where dynamics beat a same-size encoder (100% vs 73% token). Codex-mandated controls weakened the claim: larger encoders learn it, so dynamics are **parameter-efficient, not necessary**. CE-only dynamics proved more reliable (5/5 seeds) than the E5 energy (4/5; seed 512 converged to a wrong fixed point — Theorem 4 realized empirically).
3. **Dynamical characterization (D3-D18):** universally supercritical (rho > 1) yet stable via structured non-normality; two distinct stability regimes (CE "scattered" vs E5 "highway"); carry information linearly decodable but not causally used; computation is parallel, not sequential "thinking".
4. **Falsification round (D19-D21):** step-dependence is real (T=1 collapses to 1.5%); perturbation recovery fails everywhere. Adversarial Codex score: 4.5/10.
5. **Variable-T (D22-D27):** the one clean positive intervention. Sampling T in [4,16] per batch creates an anytime solver (T=32: 88.5% -> 99.9%), yields the first significant recovery result (+26.5%, d=11.3), and scales to L=24 carry depth where encoders learn nothing.
6. **Mechanism (D28-D37):** variable-T works by **contraction-rate suppression** (Delta-k p=1.7e-5, 8/8 seeds at D=8), replicated across depths D=6-10, a non-arithmetic task, and 6/6 d=128 pairs across three tested architectures (48/51 pairs in the predicted direction across the full sweep); confidence 9/10. Spectral radius is a ceiling witness, not the mechanism.
7. **Convergence blueprint (D38-D40):** CE warm-start -> flow correction -> margin-gated SC -> recovery. D38/D39: 100% in-window accuracy, but 0% converged (all wrong-attractor claims vacuous) and the flow head broken by a train/inference distribution mismatch (16/16 runs). **D40 (15/16 runs) closed the arc negatively:**
   - Residual plateaus at a lambda-dependent floor by T~50; the k-based convergence extrapolation is falsified.
   - The only substantially converged case (lambda_sc=3.0, seed 42: 91% converged at T=200) decodes to **~100% wrong attractors** with 0% sequence accuracy.
   - Stronger SC lowers residual (group means) and accelerates long-horizon accuracy collapse; CE-only (lambda=0) is the best long-horizon configuration.

## 10. Surviving Core, Rejected Claims, and Possible Reframings

### Surviving core (defensible)

- **D22 variable-T k-suppression / compute-window robustness.** An anytime-solver mechanism, replicated across depth, task, and architecture. It is **not** evidence for correct fixed-point convergence.
- The softmax-collapse information argument (Section 1) remains a valid motivation; the tested remedy failed, not the diagnosis.

### Rejected claims (do not resurrect without new evidence)

- `E(s)=||F_theta(s,c)||^2` as a semantic correctness energy: its attractors do not decode to correct outputs (D40).
- Early-iteration contraction k as a predictor of endpoint convergence (D39 extrapolation falsified by D40).
- All pre-D40 "0% wrong-attractor" results (vacuous: converged fraction was 0).
- "Dynamics are necessary for addition" (weakened to parameter efficiency).
- The thinking-generating continuum (Section 4) as an empirical claim (confidence 2.5/10; dropped).
- "UESD has higher information capacity than AR" (false; both are L*log V at readout).

### Open only under a fresh preregistered hypothesis

- **Readout-coupled energy:** define the energy so that attractor = correct decoded output, not state self-consistency alone.
- **Anytime/transient-solver framing:** build on the D22 mechanism as the primary claim.

Do not continue the old fixed-point arc by inertia; see the update protocol in `STATUS.md`. (The original Open Questions and Phase 1-4 Research Plan that stood here are preserved in git history; they described work that is now complete or closed.)

### Semantic-ratchet successor gate

The separately preregistered semantic-ratchet direction remains distinct from the closed fixed-point arc. Its primary GSM8K task-band gate is complete: frozen `base-A` produced 18 correct, 238 valid extracted incorrect, and 0 extraction failures across the canonical 256-example cohort (7.03125% exact-answer accuracy). This is a valid **ABORT-AND-SWAP** because the correct count is below both the 26-example lower band edge and the 40-example minimum critic population, selecting the one allowed SVAMP fallback.

The canonical-mode SVAMP initial-parser attempt produced 66/256 correct (25.78125% of 256), 177/256 valid extracted incorrect (69.140625% of 256), and 13/256 model-empty non-answers and extraction failures (5.078125% of 256), for 190/256 total exact-answer failures (74.21875% of 256). Although the correct count is inside the 26–217 band and both response populations exceed 40, the extraction-failure rate is one example above the preregistered 5.000% ceiling. All 13 failed responses are the literal string `Answer:` with no numeric content. The parser-repair-scope adjudication therefore makes the base-A/SVAMP gate terminal **VOID** with reason `NO_RECOVERABLE_NUMERIC_CONTENT_FOR_PERMITTED_PARSER_REPAIR`: no eligible numeric content can be recovered, and no parser change or repeated generation is permitted. The immutable initial-miss artifact remains unchanged with its point-in-time status.

The single successor `base-B` task-band gate is **COMPLETE / PASS** on the preregistered deterministic disjoint GSM8K cohort. Its four mutually exclusive categories are 143/256 correct numeric (55.859375%), 104/256 valid extracted incorrect (40.625%), 0/256 model-empty non-answers (0%), and 9/256 parser-recognition failures (3.515625%). The correct count lies inside 26–217 and exceeds 40; the usable-incorrect population is 104; both independent failure counts are at most 12/256; and the frozen leakage, provenance, determinism, cohort, and accounting checks pass. This is band-placement evidence for frozen `base-B` only, not a capability claim or evidence for the semantic-ratchet mechanism, and it does not reinterpret, pool, or rescue either base-A result.

The immutable base-B artifact is `experiments/06_uesd/results/exp_e1_task_band_base_b.json` (checkout-stable canonical-LF SHA-256 `9c57a10f3aa64c43fa34819255f4cf4e004cc5ab74afaa838c02c4a06a271de2`). Total wall time was 21,412.46s (5h56m52.46s), generated-token throughput was 1.87186 tok/s, and peak allocated/reserved VRAM was 3,278,870,016/3,716,153,344 bytes. Variable power changed throughput only and did not alter the protocol.

The independent 30M E2 mechanics pilot is **COMPLETE / CONTROLLING POST-EVIDENCE VOID**. Its immutable artifact is `experiments/06_uesd/results/exp_e2_latch_mechanics.json` (checkout-stable canonical-LF SHA-256 `7842ca6f69ba3885fe7b03142b694e9c95950f195d31acd73f601d1e3f5a4075`). Both seeds completed the frozen 500,000-token common-controller and encoder-control training budgets, supplied 491,520 selector-training states and 16,384 selector-calibration states per seed, and passed generator, split, feature-boundary, accounting, and selector-fit integrity checks.

The artifact's runner-emitted reason is `PRETEST_SELECTOR_PROVENANCE_GATE_MISSED` in 2/2 seeds. Seed 42's critic AUROC was 25,183,159/50,331,648 positive-negative pair units = 0.5003444155 over 4,096 positive and 12,288 negative states; confidence and on-policy critic AUROCs over the same 50,331,648-pair population were 0.4999594887 and 0.5003444155. Seed 31415's critic AUROC was 25,275,202.5/50,331,648 = 0.5021731555 over the same 4,096/12,288 state populations; confidence and on-policy critic AUROCs over 50,331,648 pairs were 0.4996897976 and 0.5021731555. Critic-minus-confidence AUROC advantages were 0.0003849268 and 0.0024833580, below the preregistered 0.05 floor. Critic and confidence selectors each selected 256/1,024 calibration problems correctly in each seed, for a 0-point advantage against the required 3 points.

However, confidence-matched concordance was undefined in each seed because there were 0 qualifying pairs. The preregistration requires at least 200 qualifying pairs per seed before any `PROCEED`/`FAIL`, declares a floor miss `VOID`, and the config freezes `VOID > FAIL > PROCEED` precedence. The runner's pretest branch failed to apply that denominator gate before emitting `FAIL`. The JSON remains immutable, but its terminal classification is not preregistration-admissible. The controlling post-evidence token is therefore **`VOID`**, reason `INSUFFICIENT_CONFIDENCE_MATCHED_PAIRS_PER_SEED`, observed 0/200 in both seeds.

The calibration evidence also shows why no latch-mechanics endpoint claim is available: both seeds were exactly 256/1,024 correct at every candidate horizon `T=1..15`, so the shared tie-break froze `t*=1` at pooled 512/2,048 and the observed calibration gain was 0/1,024 per seed and 0/2,048 pooled. The informational hysteresis selector therefore froze `delta=0` with `calibration_constraint_miss=true` in both seeds. The pretest branch stopped before official-test inspection. No-latch regression, critic-latch regression and reduction, gain retention, endpoint critic-versus-confidence accuracy, matched-pair AUROC, competence floors, encoder-control test accuracy, arm-4 test accuracy/suppression, and `O/F/H` headroom are all not applicable, not zero. E2 supplies design-gate diagnostics only: the fitted latent critic missed every separately measurable provenance threshold and showed no meaningful pretest advantage over confidence-plus-schedule, while the zero-denominator matched population prevents a registered scientific `FAIL`. It does not establish that a latch failed at H=16/H=32, that overthinking was absent, or that latent-state selection is impossible in general.

The frozen consequence is that the 0.5B semantic-ratchet program does not launch. Because the controlling outcome is `VOID`, this attempt does not trigger the registered automatic Direction 2 route. Direction 2 causal isolation and E2-CERT remain possible subjects for a fresh steering decision, but neither is unlocked or authorized and neither can rescue, delay, or reinterpret E2. Any future work requires an explicit reopening decision, preregistration, and review rather than an E2 rerun or adaptive threshold change.

The bounded E2-DIAG post-mortem ran **Stage 0 only**. Its first seed-42 invocation reached all 3,000 updates but crashed during post-endpoint gradient-quantile serialization, leaving the immutable point-in-time artifact `experiments/06_uesd/results/exp_e2_diag.json` under operational **`VOID_NO_ROUTE`**. The owner then authorized one operational repair with the frozen seed, 128 examples, initialization, optimizer, horizon schedule, update limit, threshold, and access prohibitions unchanged. The repaired immutable artifact is `experiments/06_uesd/results/exp_e2_diag_stage0_instrumented.json` (canonical-LF SHA-256 `0ad4fc5fafe343b37944b462972d32ff201749280a7bdbb506fa3b62536620d7`). It completed 3,000 updates and 2,786,051 non-padding input-plus-answer tokens in 98.873 GPU-seconds.

The repaired run was exactly 32/128 correct at the initial evaluation and all 30 registered checkpoints. Full-set mean CE moved from 1.522216796875 to a minimum 1.38671875 at update 2,700—still above `ln(4)`—and ended at 1.388671875. From update 100 onward, every checkpoint assigned all 128 examples to one answer class. That class changed in 21/30 intervals (2,682 example-level flips), so predictions were not frozen even though balanced-label accuracy was. The registered Stage-0 result is **`STOP / FEWER_THAN_122_OF_128_AFTER_3000_UPDATES`** and the frozen route is **`KILL_FROM_SCRATCH_LINE`**; Stage 1 and the optimizer matrix did not launch.

Instrumentation falsifies the specific broken-downstream-gradient suspicion. At every 100-update sample, every tensor in the encoder, controller, plan-slot, prefix-projector, answer-decoder, and readout groups had a nonzero gradient. At update 3,000 their respective pre-clip norms were `1.810e-5`, `0.001126`, `9.392e-10`, `0.022183`, `0.501316`, and `12.546060`; the readout-head weight delta from initialization was `1.925192` L2. There is no `detach` or `no_grad` in the model path from encoded context through plan slots, prefix projector, answer decoder, and choice head. All 3,000/3,000 updates exceeded the clip norm, so optimization saturation is a live post-mortem diagnosis, not a localized wiring bug. This informational diagnosis cannot override the registered line-kill route or authorize an optimizer fork. No model bug was fixed, and no official-test, selector, critic, or latch path was accessed. The original E2 controlling `VOID` and blocked 0.5B launch remain unchanged.

Three causal hypotheses survive as questions, not conclusions or launch authorizations:

1. **Controller competence failure:** the current optimization/supervision may not teach the synthetic deduction task at this token budget, leaving a chance-level trajectory on which no selector can demonstrate value.
2. **Missing within-trajectory outcome variation:** the absence of horizon gain and qualifying opposite-correctness pairs may mean this controller/task pair does not generate the state transitions required to test best-state memory, even if a different competent controller could.
3. **Latent-evidence insufficiency:** after competence is causally isolated, the allowed latent-content and trajectory-geometry features may still lack outcome information; the separately registered certificate-verification follow-up is one explicit alternative but cannot revise this E2 adjudication.

## 11. Key References

### Continuous Reasoning (closest systems)
- COCONUT (Hao et al., 2024): arXiv 2412.06769
- MARCOS (2025): arXiv 2509.25020
- Soft Thinking (Zhang et al., NeurIPS 2025): arXiv 2505.15778
- STAR-LDM (Lovelace et al., 2026): arXiv 2602.20528
- Latent Thought Models (ICML 2025): arXiv 2502.01567
- Meta Large Concept Model (2024)

### Continuous Generation
- LangFlow (2026): arXiv 2604.11748
- CoDAR (2026): arXiv 2603.02547
- CDCD (Dieleman et al., 2022): arXiv 2211.15089
- CALM (2025): arXiv 2510.27688
- Difformer (Gao et al., NAACL 2024): arXiv 2212.09412
- SED (Strudel et al., 2022): arXiv 2211.04236

### Energy and Fixed-Point
- EDLM (Xu et al., ICLR 2025): arXiv 2410.21357
- CLLMs (ICML 2024): arXiv 2403.00835
- DEQ (Bai et al., 2019): arXiv 1909.01377
- pcDEQ (2024): arXiv 2402.04029
- Residual EBMs (Bakhtin et al., 2020): arXiv 2004.11714
- LaDiR (2025): arXiv 2510.04573

### Softmax Alternatives
- vMF Loss (Kumar & Tsvetkov, 2018): arXiv 1812.04616
- Softmax Bottleneck studies (2024): arXiv 2404.07647

### Theoretical Foundations
- Predictive Coding (Millidge et al., 2022): arXiv 2202.09467
- Free Energy Principle (Friston, 2010)
- LSGM (Vahdat et al., NeurIPS 2021)
- Neural ODEs (Chen et al., NeurIPS 2018)

### Internal References
- Nishimori cross-domain thesis — external sibling-repository provenance; not a local dependency or UESD evidence artifact.
- CTI universal-law thesis — external sibling-repository provenance; not a local dependency or UESD evidence artifact.
- Open Exploration synthesis — external sibling-repository provenance; not required to reproduce this repository.
- Cross-domain mechanisms note — external sibling-repository provenance; not required to reproduce this repository.
- Circuit viability: `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md`
