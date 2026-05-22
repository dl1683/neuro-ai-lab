# Unified Error-Space Dynamics (UESD): Eliminating the Softmax Collapse

**Status:** Formalization + literature positioning complete. Ready for proof-of-concept.
**Author:** Devansh (devansh@svam.com)
**Date:** 2026-05-22

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

### 3.2 Why E5 Is The Most Aligned With The Core Insight

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

### 6.4 Self-Consistency + Weak Readout (Recommended)
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
| **Nishimori Thesis** (from this lab's research) | Systems at criticality achieve optimal inference via ρ = tanh(1/2) in continuous error space. Nishimori identity: avg confidence = avg accuracy. | Provides the PHYSICS grounding: softmax collapses a system away from criticality. UESD keeps it there. See §8.3. |
| **CTI Law** (from this lab's research) | Networks at criticality behave as if performing Gaussian discrimination in d_eff ≈ 1–2 dimensions. SOC → critical point → Gaussian fluctuations → LDA is optimal. | Implies the network is ALREADY doing continuous discrimination internally. Softmax is an artificial post-hoc collapse. |

### 8.3 Physics Grounding: The Nishimori Connection

From this lab's existing research (see `_meta/inquiry/THESIS.md`, `_meta/research/nishimori-cross-domain.md`):

The Nishimori identity establishes that at the critical point of inference:
- Average confidence = average accuracy (perfect calibration)
- This holds in CONTINUOUS error space at βJ = 1/2
- Seven independent substrates confirm ρ = tanh(1/2) ≈ 0.462 with CV = 1%

**Connection to UESD:** Softmax forces the system OFF the Nishimori manifold.
The simplex Δ^{V-1} is a different geometry from the Fisher-Rao manifold where
optimal inference lives. By keeping generation in the same continuous space as
thinking (R^d with its natural Riemannian metric), UESD allows the system to
remain at criticality — where the Nishimori identity guarantees optimal
calibration and the CTI law guarantees efficient discrimination.

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
| **Edge of chaos** | Criticality maximizes sensitivity, dynamic range, information storage simultaneously. | Softmax forces away from criticality (toward order at low T, chaos at high T). Continuous dynamics can maintain criticality. |

### 8.5 The Novelty Gap

The literature has every PIECE of UESD. Nobody has the SYNTHESIS:

1. **No phase boundary between thinking and generating.** COCONUT, MARCOS,
   STAR-LDM all maintain explicit phase switches. UESD eliminates the boundary.

2. **Error-space framing for generation.** EDLM and CLLMs don't frame generation
   as error minimization in a continuous space shared with reasoning.

3. **No softmax at ANY point.** Every existing system uses softmax somewhere.
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

## 9. Open Questions (Ranked by Criticality)

### Must-solve for viability

**Q1: Variable-length output.** Autoregressive models handle variable length
via <EOS>. In continuous dynamics, how does the system produce different amounts
of content?
- Hypothesis: multi-resolution state, with different subspaces saturating at
  different rates. Or: chunk-based generation (each trajectory = one chunk).

**Q2: Compositionality.** Language is compositional. Can continuous dynamics
in R^d learn to represent and manipulate compositional structure without
discrete tokens as scaffolding?
- Hypothesis: structured state spaces (e.g., slot-based), or emergent structure
  through training on compositional data.

**Q3: Training efficiency.** Autoregressive models have O(1) forward passes per
token (teacher forcing). Iterative models need T forward passes.
- Mitigation: denoising objectives (single-step training), truncated unrolling,
  progressive step counts.

### Important for scaling

**Q4: Does this scale?** Transformers scale predictably with compute. Do
energy-based iterative systems? Unknown. Need scaling law experiments.

**Q5: Readout fidelity.** If the model never sees tokens during generation,
can it learn to produce precise, token-level-accurate output?
- Hypothesis: yes, if the embedding space is rich enough and readout training
  is sufficient. But empirical validation needed.

### Important constraint

**Q6: Dual-rate dynamics.** The Open Exploration archive identifies a universal
pattern: every persistent intelligent system develops two complementary
subsystems — one fast/rigid/cheap, one slow/flexible/expensive (hippocampus
vs. neocortex, System 1 vs. System 2). Does UESD's single error function
conflict with this?
- Resolution: The error function is unified, but the DYNAMICS can be dual-rate.
  Thinking = high-precision error minimization on slow timescale. Generating =
  low-precision error minimization on fast timescale. Both in the same S, both
  using the same E, but with different step sizes or precision levels.
  Formally: τ_think >> τ_generate, but E is the same.

### Interesting for theory

**Q7: What does the energy landscape look like?** Are there clean basins?
How many modes? What determines basin structure?

**Q8: Does the thinking–generating continuum actually emerge?** Or do trained
systems collapse to a trivial trajectory (one step, done)?

**Q9: Relationship to biological cognition.** Predictive coding and free energy
minimization are leading theories of brain function. Is UESD a good computational
model of how biological systems generate language?

---

## 10. Research Plan

### Phase 1: Proof of Concept (immediate)

**Goal:** Demonstrate that self-consistency dynamics (E5) in continuous embedding
space can converge to states that decode to correct tokens on a non-trivial task.

1. **Task:** Sequence-to-sequence transduction on a synthetic grammar.
   Small vocabulary (V=64–256), sequences of length 8–32.
   Something with compositional structure (e.g., reverse, sort, simple arithmetic).

2. **Architecture:** Weight-tied transformer block as F_θ (DEQ-style).
   d=128–256. Cross-attention to encoded input. Readout via nearest-neighbor
   to learned embeddings.

3. **Training:** Self-consistency + weak readout (§6.4).
   Compare against: (a) standard transformer with softmax, (b) single-step
   baseline (F_θ applied once), (c) diffusion in embedding space (E3).

4. **Measurements:**
   - Does the system converge? (||F_θ|| → 0)
   - How many steps to convergence?
   - Does readout accuracy match softmax baseline?
   - Does the thinking–generating continuum emerge? (plot ||F_θ|| vs. step)
   - What does the trajectory look like in PCA-reduced S?

**Success bar:** Readout accuracy within 5% of softmax baseline on the same
architecture and data budget. Convergence in <50 steps. Visible phase structure
in ||F_θ|| trajectory.

### Phase 2: Error Function Shootout

Implement E1–E5 on the same task from Phase 1. Compare:
- Convergence speed (steps to ||F_θ|| < ε)
- Training stability (loss variance, gradient norms)
- Output quality (exact match, BLEU if applicable)
- Trajectory structure (PCA visualization, ||F_θ|| profiles)
- Sensitivity to hyperparameters (η, λ_1/λ_2, number of steps)

### Phase 3: Scaling and Language

If Phase 1 and 2 succeed:
- Scale to natural language (small LM, ~10M params, WikiText-2)
- Compare perplexity against standard transformer and LangFlow
- Test whether Nishimori criticality metrics (ρ, calibration) are better
  preserved in UESD vs. softmax
- Investigate variable-length output via chunked generation

### Phase 4: Theory

- Formal proof that softmax violates a circuit viability condition
- Derive connection between E5 self-consistency and Nishimori identity
- Characterize the energy landscape: basin structure, mode connectivity
- Scaling laws for iterative systems vs. autoregressive

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
- Nishimori cross-domain thesis: `_meta/research/nishimori-cross-domain.md`
- CTI universal law: `_meta/inquiry/THESIS.md`
- Open Exploration synthesis: `_meta/research/open-exploration-synthesis.md`
- Cross-domain mechanisms: `_meta/insights/cross-domain-mechanisms.md`
- Circuit viability: `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md`
