# UESD Experiment Design — Tesla Mode System Design Document

## 1. Overview

This document specifies proof-of-concept experiments for Unified Error-Space
Dynamics (UESD). The goal: determine whether self-consistency dynamics in
continuous embedding space can match softmax-based generation on controlled
tasks, and whether the thinking-generating continuum emerges.

**Compute:** RTX 5090 (25.7 GB VRAM), 68 GB RAM, 24 CPUs.
**Budget:** Each experiment should train in <30 minutes. Total suite <4 hours.
**Kill chain:** Each experiment gates the next. Early failures terminate early.

---

## 2. The Kill Chain (Experiment Ordering)

```
Exp 0 (Math): Information bottleneck derivation
  ↓ (confirms theoretical motivation — no gate)
Exp A (Copy): Do dynamics converge to correct embeddings?
  ↓ YES → proceed    NO → UESD non-viable, study why, stop
Exp B (Reversal): Can dynamics solve non-trivial transformations?
  ↓ YES → proceed    NO → study architecture, don't proceed to C
Exp C (Error Shootout): E1 vs E3 vs E5 — does self-consistency matter?
  ↓ E5 advantage → core insight validated    No advantage → insight empty
Exp D (Sorting): Does adaptive compute emerge with varying difficulty?
  ↓ YES → thinking-generating continuum confirmed    NO → fixed compute
```

Each experiment answers ONE question. Failure at any step is informative,
not shameful — it tells us WHERE UESD breaks.

---

## 3. Shared Architecture

### 3.1 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| d_model | 128 | Small enough for fast training, large enough for V=64 |
| n_heads | 4 | head_dim=32, standard ratio |
| d_ff | 512 | 4× expansion |
| n_enc_layers | 2 | Sufficient for short-sequence encoding |
| V (vocab) | 64 | Small but non-trivial |
| L (seq len) | 8 | Short enough for fast iteration |
| T (dynamics steps) | 10 | Default unrolling depth |
| batch_size | 256 | Fits in VRAM easily at this scale |
| lr | 3e-4 | Standard Adam |
| training_steps | 20K | ~5 min at this scale |
| Total params | ~700K | Tiny model, fast experiments |

### 3.2 Context Encoder (shared, not novel)

Standard transformer encoder. Input tokens → embeddings + positional encoding
→ N self-attention layers → context vectors c ∈ R^{L_in × d}.

### 3.3 Dynamics Block F_θ (the core)

Weight-tied transformer decoder layer:
1. Pre-norm self-attention over L_out output positions
2. Pre-norm cross-attention to encoded context
3. Pre-norm FFN

The FULL dynamics layer computes: block(s_t, c) = s_t + SA(s_t) + CA(s_t, c) + FFN(...)
The UPDATE is: F_θ(s_t, c) = block(s_t, c) - s_t

Applied T times with shared parameters (DEQ-style weight tying).

**Convergence control:** Spectral normalization on FFN weights. Optional
learned gating: z = σ(W_g · [s_t, F_θ(s_t)]); s_{t+1} = s_t + z ⊙ F_θ(s_t).

### 3.4 State Initialization

s_0[l] = pos_embed_out[l] for l = 0, ..., L_out-1.
Learned position-specific vectors. Not random, not from encoder output.
This creates a clean "cold start" — the dynamics must do all the work.

### 3.5 Readout

Project state to embedding space:
  h = W_readout · s_T[l]

Cosine similarity to all vocabulary embeddings:
  sim(l, v) = cos(h[l], embed[v])

Training: softmax over similarities with temperature τ=0.1, cross-entropy loss.
Inference: argmax over similarities per position.

This softmax is POST-HOC. It does not participate in the dynamics loop.

### 3.6 Parameter Counting

| Component | Params |
|-----------|--------|
| Token embedding (V=64, d=128) | 8K |
| Position embedding (in + out) | 4K |
| Encoder (2 layers) | ~400K |
| Dynamics block (1 layer) | ~260K |
| Readout projection | 16K |
| **Total** | **~690K** |

---

## 4. Experiment 0: Information Bottleneck (Math Only)

### Claim
Softmax + sampling limits information flow to log₂(V) bits per step.
Continuous dynamics in R^d can carry O(d) bits.

### Derivation Sketch

1. At each AR step: h_t → softmax(Wh_t) → sample x_t
2. Mutual information: I(h_t; x_t) ≤ H(x_t) ≤ log₂(V)
3. For V=64: ≤ 6 bits per step
4. For d=128 at float32: state carries 128 × 32 = 4096 bits
5. Compression ratio: 4096/6 ≈ 683:1

The continuous dynamics preserve the full state at each step.
The softmax discards >99.8% of the representational capacity.

### Connection to circuit viability
The softmax step is an information bottleneck analogous to the SynFlow
bridge collapse. The pathway exists (the linear projection W maps to V dims)
but the bandwidth is crushed by sampling to a single token.

Formally: define the "bridge capacity" of a generation step as the
mutual information between consecutive internal states. For AR:
I(h_t; h_{t+1}) ≤ I(h_t; x_t) ≤ log₂(V) (data processing inequality)

For UESD:
I(s_t; s_{t+1}) = I(s_t; s_t + F_θ(s_t, c)) — depends on F_θ, but
no discrete bottleneck constrains it.

### Success bar
Clean derivation with tight bounds. No compute needed.

---

## 5. Experiment A: Convergence Test (Copy Task)

### Claim
Weight-tied dynamics can converge to fixed points in embedding space that
decode to correct tokens.

### Task
Identity function. Input: [a, b, c, d, e, f, g, h]. Output: same.
V=64, L=8. 50K training samples (generated on the fly). 10K test.

### Why identity first
Isolates convergence from task difficulty. If dynamics can't learn to
reproduce the input, nothing harder will work.

### Training
**E1 only** (simplest possible error):
```
L = ||s_T - Embed(y*)||²
```
where Embed(y*) is the target token embeddings at each position.

T=10 unrolling steps. Standard Adam, lr=3e-4.

### Baselines
1. Standard AR transformer (2-layer decoder, softmax, teacher forcing)
2. Single-step baseline (F_θ applied once, T=1)
3. Oracle: what accuracy does nearest-neighbor readout on raw embeddings achieve?

### Measurements
1. Token accuracy at each step t (should increase monotonically)
2. ||F_θ(s_t)|| at each step (convergence profile — should decrease)
3. Final exact-sequence accuracy at T=10
4. Accuracy vs. T at inference (test with T=1,2,5,10,20,50)

### Success bar
≥99% token accuracy. ||F_θ|| decreases monotonically. Accuracy improves
with T up to a plateau.

### Failure analysis
If convergence fails:
- Check spectral norm of Jacobian (is the map contractive?)
- Try stronger spectral normalization
- Try learned gating
- Try smaller learning rate

If readout fails (converged but wrong tokens):
- Check embedding space geometry (are embeddings well-separated?)
- Try vMF loss instead of L2
- Try larger d_model

---

## 6. Experiment B: Transformation Test (Reversal)

### Claim
Dynamics can converge to DIFFERENT fixed points depending on input,
producing correct non-trivial outputs.

### Task
Reverse. Input: [a, b, c, d, e, f, g, h]. Output: [h, g, f, e, d, c, b, a].
V=64, L=8. Same data generation as Exp A.

### Training
**E1** (embedding prediction): L = ||s_T - Embed(reverse(y*))||²
**E5** (self-consistency): L = λ₁||F_θ(s_T)||² + λ₂ CE(readout(s_T), y*) + λ₃ Σ_t ||F_θ(s_t)||²

Train both separately, same hyperparameters.
λ₁ = 0.1, λ₂ = 1.0, λ₃ = 0.01 (readout-dominated, convergence as regularizer).

### Baselines
Same as Exp A, but on reversal task.

### Measurements
Everything from Exp A, plus:
5. PCA visualization of trajectories s_0, ..., s_T for 100 test examples
6. Comparison of convergence profiles: reversal vs. identity (from Exp A)
   Does reversal take more steps to converge?
7. Per-position accuracy analysis: are edge positions (which move furthest) harder?

### Success bar
≥95% exact-sequence accuracy. Reversal trajectories should be longer
(more steps to converge) than identity trajectories.

---

## 7. Experiment C: Error Function Shootout

### Claim
E5 (self-consistency) produces qualitatively different trajectory structure
than E1 (embedding prediction) and E3 (denoising).

### Task
Reversal (same as Exp B).

### Methods
Three error functions, same architecture, same hyperparameters:

**E1 (Embedding Prediction)**:
```
L = Σ_l ||s_T[l] - embed(y*[l])||²
```

**E3 (Denoising in S)**:
```
s* = Embed(y*)
s_noisy = s* + σε,  σ ~ Uniform(0.1, 1.0)
L = ||F_θ(s_noisy, c) - s*||²
```
Note: E3 is single-step training but multi-step inference.

**E5 (Self-Consistency + Weak Readout)**:
```
L = λ₁||F_θ(s_T, c)||²              (convergence)
  + λ₂ CE(readout(s_T), y*)          (readout quality)
  + λ₃ Σ_t ||F_θ(s_t, c)||²          (path efficiency)
```
λ₁=0.1, λ₂=1.0, λ₃=0.01.

### Measurements
For each error function:
1. Token accuracy vs. training step (learning curves)
2. Token accuracy vs. inference steps T (1, 2, 5, 10, 20, 50)
3. ||F_θ(s_t)|| profile at inference (convergence shape)
4. PCA trajectory visualization (qualitative structure)
5. Gradient norm statistics during training (stability)
6. Training time per step

### Key comparison
The CRITICAL test: does E5 produce a DIFFERENT trajectory structure than E1?
- E1 should produce trajectories that converge directly to the target embedding
- E5 should produce trajectories that explore before converging (the continuum)
- If E5 trajectories look the same as E1, the self-consistency insight is empty

### Success bar
At least one method achieves >90% accuracy. E5 produces visibly different
trajectory structure from E1 (measured by trajectory length, curvature, or
PCA shape). E5 is at least as accurate as E1 (same or better).

---

## 8. Experiment D: Adaptive Compute (Sorting)

### Claim
Harder tasks require more dynamics steps to converge. The thinking-generating
continuum emerges from task difficulty, not from architecture design.

### Task
Sort. Input: random sequence of tokens. Output: sorted sequence.
V=32 (smaller vocab for cleaner sorting), L=4,8,12,16.

### Training
E5 (self-consistency). Train on mixed lengths. T=20 (higher for harder tasks).

### Measurements
For each length L:
1. Exact-sequence accuracy
2. Steps to convergence (first t where ||F_θ(s_t)|| < ε)
3. ||F_θ(s_t)|| profile shape
4. Whether longer sequences have longer "thinking" phases

### Key prediction
If the thinking-generating continuum is real:
- Steps-to-convergence should increase with L
- Short sequences (L=4) should converge fast (near one-step)
- Long sequences (L=16) should have extended exploration phases
- The ||F_θ|| profile should show a PLATEAU (thinking) then DECAY (generating)

### Success bar
Monotonic increase in steps-to-convergence with L. At least 2× difference
between L=4 and L=16. Visible phase structure in ||F_θ|| profiles.

---

## 9. Baselines (Fair Comparison Protocol)

### 9.1 Baseline A: Standard AR Transformer
- Same encoder
- 2-layer autoregressive decoder with causal attention
- Softmax + cross-entropy training
- Teacher-forced
- Params: ~660K (matched)
- FLOPs: ~1× (one forward pass)

### 9.2 Baseline B: Deep AR Transformer (FLOP-matched)
- Same encoder
- T-layer autoregressive decoder (T=10 → 10 layers)
- Params: ~2.6M (NOT matched — more params)
- FLOPs: ~10× encoder, T× decoder (matched to UESD)

### 9.3 Baseline C: Depth-Recurrent AR (param+FLOP matched)
- Same encoder
- Same weight-tied decoder layer as UESD
- Applied T times recurrently
- BUT: at the end, decode autoregressively with softmax
- Params: ~690K (matched)
- FLOPs: ~T× (matched)
- This isolates: does the BENEFIT come from weight tying + iterative
  refinement, or from the continuous error space?

### 9.4 Baseline D: SUNDAE-style (iterative refinement with argmax)
- Same architecture as UESD
- BUT: at each step, do argmax to tokens, re-embed, continue
- This tests: does the continuous dynamics (no argmax) help vs. discrete
  iterative refinement?

---

## 10. Diagnostics & Visualization Suite

Every experiment produces:

### 10.1 Convergence Diagnostics
- ||F_θ(s_t)|| vs. step t (per-example, averaged)
- Spectral norm of Jacobian ∂F_θ/∂s at convergence
- Fixed-point residual: ||s_T - s_{T+k}|| for k=1,...,10 (is it actually fixed?)

### 10.2 Trajectory Visualization
- PCA of s_0, ..., s_T for 100 test examples (2D and 3D)
- Color by: (a) step number, (b) current accuracy, (c) ||F_θ||
- Compare trajectory shapes across error functions

### 10.3 Readout Analysis
- Per-position accuracy
- Cosine similarity between s_T[l] and correct embedding
- Cosine similarity between s_T[l] and all other embeddings (confusion matrix)
- t-SNE of converged states colored by target token

### 10.4 Training Dynamics
- Loss components over training (for E5: convergence, readout, smoothness)
- Gradient norms per component
- Learning curves (accuracy vs. step)

### 10.5 Computational Cost
- Wall time per training step
- Wall time per inference (vs. baselines)
- Memory usage

---

## 11. Implementation Plan

### File Structure
```
experiments/06_uesd/
├── EXPERIMENT_DESIGN.md          (this document)
├── shared/
│   ├── model.py                  (UESDModel, baselines)
│   ├── data.py                   (task generators)
│   ├── training.py               (training loops for E1, E3, E5)
│   ├── diagnostics.py            (convergence, trajectory, readout analysis)
│   └── visualization.py          (PCA, convergence plots, etc.)
├── exp_0_bottleneck.py           (mathematical derivation)
├── exp_a_copy.py                 (convergence test)
├── exp_b_reversal.py             (transformation test)
├── exp_c_error_shootout.py       (E1 vs E3 vs E5)
├── exp_d_sorting.py              (adaptive compute)
└── results/                      (JSON artifacts + figures)
```

### Build Order
1. `shared/data.py` — task generators (copy, reverse, sort)
2. `shared/model.py` — UESDModel + 4 baselines
3. `shared/training.py` — training loops
4. `exp_a_copy.py` — run first experiment
5. (Gate: if Exp A fails, stop and diagnose)
6. `shared/diagnostics.py` + `shared/visualization.py`
7. `exp_b_reversal.py`, `exp_c_error_shootout.py`, `exp_d_sorting.py`

### What NOT to Build
- Natural language experiments (too early)
- Large models (stay ≤1M params)
- Fancy architectures (no mixture of experts, no sparse attention)
- Gradient-based energy dynamics (F_θ = -∇Ψ is interesting but adds
  complexity; test basic dynamics first)
- Variable-length output (fix L_out for now)
- Continuous-time ODE solver (discrete steps are simpler and sufficient)

---

## 12. Hypotheses & Predictions

### H1: Convergence is achievable
Weight-tied dynamics with spectral normalization will converge to fixed
points on the copy task within 10 steps.
**Prediction:** ||F_θ|| decreases by >90% from step 0 to step 10.
**Confidence:** High (8/10). DEQ literature shows this works.

### H2: Reversal requires more steps than copying
The thinking-generating continuum should emerge: reversal, being harder
than identity, should produce longer trajectories.
**Prediction:** Steps-to-convergence(reversal) > Steps-to-convergence(copy) by ≥2×.
**Confidence:** Medium (6/10). If the model has enough capacity, it might
solve reversal in ~the same number of steps as identity.

### H3: E5 produces different trajectory structure
Self-consistency energy should produce trajectories with an exploration
phase (high ||F_θ||, variable direction) followed by convergence, while
E1 should produce direct descent to the target embedding.
**Prediction:** E5 trajectories have higher total path length than E1.
**Confidence:** Medium (5/10). The smoothness penalty might suppress exploration.

### H4: Sorting length drives adaptive compute
Longer sequences should take more steps to converge when sorting.
**Prediction:** Steps-to-convergence is monotonically increasing with L.
**Confidence:** High (7/10). Sorting complexity is O(n log n); longer
sequences need more "work" regardless of method.

### H5: UESD matches softmax baselines
On these simple tasks, UESD should achieve accuracy within 5% of
softmax-based autoregressive generation.
**Prediction:** UESD token accuracy ≥ 95% of AR baseline accuracy.
**Confidence:** Medium (6/10). The readout might be a weak link.

### Risk Register
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dynamics don't converge | Medium | Fatal | Spectral norm, gating, smaller lr |
| Readout fails (converged but wrong) | Medium | Major | vMF loss, larger d, better embed geometry |
| E5 collapses to 1-step | High | Major | Harder tasks, tune λ₁/λ₃ ratio |
| Training unstable (exploding grads) | Medium | Major | Gradient clipping, progressive unrolling |
| No difference between E1 and E5 | Medium | Undermines thesis | Design more discriminating tasks |
| Trivial task doesn't test thesis | High | Minor | Tasks are staged by difficulty |
