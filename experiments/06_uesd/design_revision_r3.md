# UESD Minimal Build Spec — Round 3

This is the LOCKED build spec. Two experiments, two tracks, one baseline,
one ablation, exact thresholds. Nothing else ships in the first iteration.

## Scope

| In | Out (deferred) |
|----|----------------|
| Copy smoke test | Sorting / adaptive compute |
| Reversal main test | One-to-many / multimodal |
| Track A (E1 embedding regression) | Track C (learned energy) |
| Track B (E5 self-consistency + readout) | PCA/t-SNE trajectory viz |
| AR baseline (param-matched) | FLOP-matched baseline, SUNDAE baseline |
| Encoder-only ablation | Full lambda grid |
| lambda_1 in {0, 0.1, 1.0, 10.0} | Variable-length, hierarchy, Nishimori |

## Architecture (locked)

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| d_ff | 512 |
| n_enc_layers | 2 |
| vocab_size | 64 |
| seq_len | 8 |
| T (dynamics steps) | 10 |
| batch_size | 256 |
| lr | 3e-4 (Adam) |
| training_steps | 20K |

## Components

### 1. Data Generator
- Copy task: random sequence V=64, L=8. Output = input.
- Reversal task: random sequence V=64, L=8. Output = reversed input.
- Generated on the fly (no dataset files).

### 2. Context Encoder
Standard transformer encoder. 2 layers, d=128, 4 heads.
Input: token IDs -> embedding + positional encoding -> self-attention layers.
Output: context vectors c in R^{L_in x d}.

### 3. Dynamics Block F_theta
ONE transformer decoder layer (weight-tied across T steps):
- Pre-norm self-attention over L_out output positions
- Pre-norm cross-attention to encoded context c
- Pre-norm FFN (d_model -> d_ff -> d_model)
- Spectral normalization on FFN linear layers

The update map: G(s, c) = DecoderLayer(s, c)
The update vector: F_theta(s, c) = G(s, c) - s
New state: s_{t+1} = G(s_t, c) = s_t + F_theta(s_t, c)

### 4. State Initialization
s_0[l] = learned_pos_embed[l] for l = 0, ..., L_out - 1.
Deterministic, cold start.

### 5. Readout
h = readout_proj(s_T[l])  (linear projection, d -> d)
sim(l, v) = cos(h[l], embed_weight[v])  for all v in vocab
Inference: argmax_v sim(l, v) per position
Training: softmax(sim / tau) with tau=0.1, then CE against target tokens

### 6. Encoder-Only Ablation
Skip dynamics entirely. Encode input, apply a linear map from
encoder output to L_out x d, readout directly.
This tests: how much work is the ENCODER doing vs the DYNAMICS?

### 7. AR Baseline
Standard transformer encoder-decoder:
- Same encoder (2 layers)
- 2-layer autoregressive decoder with causal mask
- Softmax + CE loss, teacher-forced training
- ~660K params (matched to UESD)

## Error Tracks

### Track A: Embedding Regression (E1)
```
L = (1/L) * Sum_l ||s_T[l] - embed(y*[l])||^2
```
No convergence penalty. Direct regression to target embeddings.
The simplest possible continuous-space objective.

### Track B: Self-Consistency + Readout (E5)
```
L = lambda_1 * ||F_theta(s_T, c)||^2 + lambda_2 * CE(readout(s_T), y*)
```
lambda_2 = 1.0 always.
lambda_1 in {0, 0.1, 1.0, 10.0} (sweep).
lambda_1 = 0 is the "CE-only" control (no convergence pressure).
Warm-up: lambda_1 starts at 0, linearly ramps to target over steps 0-5K.

## Diagnostics (exact thresholds)

### D1: Token Accuracy
Per-position and per-sequence exact match.
GATE: Copy >= 99% token accuracy. Reversal >= 90% token accuracy.

### D2: Normalized Residual
r_norm = ||F_theta(s_T, c)|| / (sqrt(L_out * d_model))
Normalized per token and dimension for comparability.
REPORT: mean and std over test set at each dynamics step t.

### D3: Decoder Margin
m(l) = cos(s_T[l], e_{y*_l}) - max_{v != y*_l} cos(s_T[l], e_v)
GATE: mean decoder margin > 0.1 (positive margin = correct with confidence).

### D4: Wrong-Attractor Rate
Fraction of test examples where r_norm < 0.01 AND token accuracy < 100%.
"Converged but wrong."
GATE for E5 viability: wrong-attractor rate < 5%.

### D5: Basin Perturbation
Add Gaussian noise (sigma = 0.1 * ||s_T||) to s_T.
Re-run dynamics for 10 more steps. Measure fraction that return
to same readout.
REPORT: basin stability fraction. Higher = more robust attractors.

### D6: Spectral Radius
Estimate spectral radius of Jacobian dG/ds at s_T via 10 steps
of power iteration.
GATE: rho < 1.0 (locally contractive). REPORT: mean rho over test set.

## Experiment A: Copy Smoke Test

### Purpose
Gate: do dynamics converge to correct embeddings at all?

### Protocol
1. Train Track A (E1) on copy task. 20K steps.
2. Train Track B (E5, lambda_1=1.0) on copy task. 20K steps.
3. Train AR baseline on copy task. 20K steps.
4. Train encoder-only ablation on copy task. 20K steps.
5. Evaluate all on 10K test examples.
6. Report D1-D6 for each.

### Gates
- Track A copy accuracy >= 99%: PASS (dynamics converge)
- Track A copy accuracy < 90%: FAIL (dynamics fundamentally broken, stop)
- Track A between 90-99%: INVESTIGATE (readout or convergence issue)

### What we learn
- Does iterative refinement in embedding space produce correct readouts?
- How many steps T does copy need? (Should need very few — near 1-step)
- Is the encoder-only ablation nearly as good? (If yes, dynamics aren't needed for copy — expected, this is a smoke test)
- What is the spectral radius?

## Experiment B: Reversal Main Test

### Purpose
Core test: can dynamics solve non-trivial transformations?
E5 vs E1 comparison. Lambda sweep.

### Protocol
1. Train Track A (E1) on reversal. 20K steps.
2. Train Track B (E5) on reversal with lambda_1 in {0, 0.1, 1.0, 10.0}. 20K steps each.
3. Train AR baseline on reversal. 20K steps.
4. Train encoder-only ablation on reversal. 20K steps.
5. Evaluate all on 10K test examples.
6. Report D1-D6 for each.

### Gates
- Track A reversal accuracy >= 90%: PASS (UESD can transform)
- Track B wrong-attractor rate < 5%: E5 VIABLE
- Track B wrong-attractor rate > 20%: E5 DEAD
- Track A/B within 5% of AR baseline: COMPETITIVE
- Encoder-only ablation >> 80%: CONCERN (encoder doing the work)

### Key comparisons
- E1 vs E5: does self-consistency change trajectory structure?
  Measure: ||F_theta|| profiles, decoder margin trajectories.
- E5 lambda sweep: which lambda_1 gives best accuracy/convergence tradeoff?
- AR vs UESD: same accuracy? Same or different error patterns?
- Encoder-only vs UESD: how much value does the dynamics loop add?

### What we learn
- Is UESD competitive with AR on reversal?
- Does E5 produce more or fewer wrong attractors than E1?
- Does lambda_1 matter? (If lambda_1=0 is best, convergence pressure hurts)
- Is the dynamics loop actually doing useful work? (encoder-only ablation)

## Build Order

1. shared/data.py — copy and reversal generators
2. shared/model.py — UESDModel, ARBaseline, EncoderOnlyAblation
3. shared/diagnostics.py — D1-D6 implementations
4. shared/training.py — training loops for E1, E5, AR
5. exp_a_copy.py — run Experiment A, check gates
6. IF gates pass: exp_b_reversal.py — run Experiment B
7. Results to experiments/06_uesd/results/ as JSON

## What This POC Decides

| Outcome | Conclusion | Next step |
|---------|-----------|-----------|
| Exp A fails (copy < 90%) | Dynamics don't converge | Study Jacobian, try different arch |
| Exp A passes, Exp B fails (<70%) | Dynamics converge but can't transform | Study representation capacity |
| Exp B passes, E5 wrong-attractor > 20% | E5 is dead, E1 is the path | Drop E5, investigate learned energy (Track C) |
| Exp B passes, E5 wrong-attractor < 5% | E5 viable as proposed | Proceed to harder tasks, scaling |
| Exp B passes, encoder-only also passes | Dynamics aren't needed | Challenge: make tasks harder, or accept encoder is enough |
| Exp B UESD competitive with AR | Core thesis supported | Scale up |
| Exp B UESD >> AR | Unexpected win | Validate carefully, check for bugs |
