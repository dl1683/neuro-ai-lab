# Information-Theoretic Analysis of the Softmax Bottleneck

## Overview

This derivation formalizes the information bottleneck created by the
softmax + sampling step in autoregressive generation, and contrasts it
with the continuous-state refinement in UESD. The analysis addresses
Codex R1's critique that the original Exp 0 conflated storage capacity
(float32 bits) with mutual information (semantic bits).

---

## 1. The Data Processing Inequality Bound

### Setup

At each autoregressive generation step, the processing chain is:

    h_t -> z_t = W_h * h_t -> p_t = softmax(z_t / T) -> x_t ~ Cat(p_t) -> e_t = Embed(x_t)

where:
- h_t in R^d is the hidden state
- z_t in R^V is the logit vector
- p_t in Delta^{V-1} is the categorical distribution
- x_t in {0, 1, ..., V-1} is the sampled token
- e_t in R^d is the re-embedded token

### Theorem (MI Bound through Softmax + Sampling)

For any hidden state h_t and any processing chain ending in a discrete
sample x_t from a V-ary categorical:

    I(h_t; x_t) <= H(x_t) <= log_2(V)

**Proof.** By the data processing inequality (DPI), for the Markov chain
h_t -> p_t -> x_t:

    I(h_t; x_t) <= I(p_t; x_t) = H(x_t) - H(x_t | p_t)

Since x_t ~ Cat(p_t), H(x_t | p_t) >= 0, so I(p_t; x_t) <= H(x_t).
And H(x_t) <= log_2(V) for a V-ary random variable (maximum entropy
for the uniform distribution).

Note on tightness. If p_t is uniform, H(x_t) = log_2(V) but x_t
is nearly independent of h_t (no useful information transfer).
If temperature -> 0 (argmax regime), the output is deterministic:
x_t = argmax_v z_v. In this case, x_t carries exactly H(x_t) bits
of information about h_t (where H(x_t) depends on how many distinct
argmax values appear across the distribution of h_t). If the mapping
h_t -> argmax(W*h_t) is approximately injective, MI can approach
log_2(V) bits. The information loss in the argmax regime comes from
the DISCRETIZATION, not from randomness: continuous logit magnitudes
and relative rankings are discarded.

**Extending to re-embedding.** By DPI applied to x_t -> e_t = Embed(x_t):

    I(h_t; e_t) <= I(h_t; x_t) <= log_2(V)

This is the hard ceiling on information preserved through the
softmax bottleneck at each step.  QED.

---

## 2. Sequential Information Accumulation in AR

### Total Information in L Tokens

The full AR sequence (x_1, ..., x_L) carries at most:

    I(h_0; x_1, ..., x_L) <= sum_{t=1}^{L} I(h_0; x_t | x_1, ..., x_{t-1})

By DPI at each step:

    I(h_0; x_t | x_{<t}) <= H(x_t | x_{<t}) <= log_2(V)

Therefore:

    I(h_0; x_1, ..., x_L) <= L * log_2(V)

**Example:** For V = 64, L = 8: I <= 8 * 6 = 48 bits.
For V = 32000, L = 4096: I <= 4096 * 14.97 ~ 61,300 bits.

### Key Qualification (Codex R1 Critique)

This bound is on the mutual information between the initial hidden
state h_0 and the generated sequence. It is NOT the same as "storage
capacity" of the hidden state. A float32 vector in R^d has d * 32 bits
of storage, but its mutual information with any target is determined by
the signal-to-noise ratio and the encoding quality, not the raw
bit-width.

The correct comparison is: how much task-relevant information must the
output carry to solve the task, and how does each architecture deliver it?

---

## 3. Rate-Distortion Perspective

### Task-Required Information

For a sequence-to-sequence task mapping input x to output y*:

    I(S_T; y*) >= H(y*) - H(y* | S_T)

where S_T is the final state (UESD) or the generated sequence (AR).

For zero-error decoding of L tokens from vocab V:

    I(S_T; y*) >= L * log_2(V)   (= H(y*) for uniform targets)

### How Each Architecture Delivers It

**AR:** Delivers information sequentially. At each step t, the model
makes an irrevocable discrete choice x_t. The total information is
accumulated across L steps, each contributing at most log_2(V) bits.
The sequential bottleneck means errors compound: an incorrect x_t
provides wrong conditioning for all subsequent tokens.

**UESD:** Delivers information in parallel. The state S_T in R^{L x d}
carries L * d continuous values. The mutual information I(S_T; y*) is
bounded by the channel capacity of the readout:

    I(S_T; y*) <= I(S_T; readout(S_T)) <= H(readout(S_T)) <= L * log_2(V)

So the ULTIMATE information delivered to the output is the same:
L * log_2(V) bits. The difference is in HOW that information is
produced:

- AR: sequential sampling, each step irrevocable
- UESD: parallel refinement, all positions simultaneously, no commitment
  until final readout

### The Real Advantage

The advantage of UESD is NOT higher total information capacity.
Both AR and UESD must deliver the same L * log_2(V) bits to specify
the output.

The advantage is the PROCESS:

1. **No premature commitment.** AR must commit to x_1 before considering
   x_2. UESD refines all positions simultaneously, allowing cross-position
   interactions at every dynamics step.

2. **Continuous state preservation.** Between dynamics steps, UESD
   preserves the full R^{L*d} state. AR collapses to a discrete token
   and must reconstruct context from the sequence so far.

3. **Error correction.** UESD can correct partial solutions through
   continued refinement. AR must generate explicit correction tokens.

4. **Parallel throughput.** All L output positions are refined in a
   single dynamics step. AR requires L sequential steps.

---

## 4. The Softmax as a Channel

Model the softmax + sampling as an information channel:

    Channel input: z in R^V (logits)
    Channel output: x in {0, ..., V-1} (sampled token)
    Channel capacity: C = log_2(V) bits

This is a V-ary symmetric channel with capacity log_2(V). The key
properties:

1. **Irreversible.** Once x is sampled, the logit distribution z is
   lost. You cannot recover z from x.

2. **Lossy.** The confidence information (how peaked was the
   distribution?) is discarded.

3. **Amplifying.** Small differences in logits can lead to different
   tokens, creating discrete jumps in the continuation.

### Contrast: UESD's State Update

UESD's dynamics step:

    Input: s_t in R^{L*d}
    Output: s_{t+1} = G(s_t, c) in R^{L*d}

Unlike the softmax channel, this is a deterministic continuous map.
It preserves all L*d state dimensions without discretization. The
mutual information I(s_t; s_{t+1}) is bounded by H(s_t), which for
continuous variables depends on the differential entropy of the
state distribution (not the storage precision).

The dynamics step is:
1. **Invertible near fixed points** (if J = dG/ds has no zero eigenvalues).
2. **Dimension-preserving** (input and output have the same dimensionality).
3. **Smooth** (small state changes produce small output changes,
   given Lipschitz dynamics with bounded Jacobian).

Note: calling this "unbounded capacity" would be imprecise.
The effective information preserved depends on the Jacobian
structure and the noise floor of the computation.

---

## 5. Information-Theoretic Summary

| Quantity | AR | UESD |
|----------|-----|------|
| Information per step | <= log_2(V) bits | Unbounded (continuous) |
| Total output info | <= L * log_2(V) bits | <= L * log_2(V) bits (same) |
| Intermediate state | Discrete token x_t | Continuous state s_t in R^{L*d} |
| Error correction | Requires future tokens | Continuous refinement |
| Cross-position interaction | Through conditioning on x_{<t} | Through self-attention at each step |
| Commitment | Irrevocable at each step | Deferred until readout |

**Conclusion.** The softmax bottleneck is real but the framing matters.
It is not about total capacity (both deliver L*log_2(V) bits). It is
about the sequential commitment and information loss at each step.
UESD's advantage is process efficiency, not capacity.

---

## 6. Corrected Claim

**Original claim (Exp 0, overclaimed):**
"Continuous state carries d * 32 bits vs. log_2(V) bits through softmax."
This compares storage with information -- invalid.

**Corrected claim:**
"At each generation step, AR loses all non-selected information through
the softmax bottleneck (MI <= log_2(V) bits). UESD preserves the full
continuous state between dynamics steps, enabling parallel refinement
without premature commitment. The total output information is identical
(L * log_2(V) bits for both), but the generation PROCESS is
fundamentally different."

This is scientifically defensible and does not overclaim.
