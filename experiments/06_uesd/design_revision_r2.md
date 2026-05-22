# UESD Design Revision — Response to Codex Round 1

## Addressing the Core Critique: E5 as Stopping Condition vs. Error Function

Codex is right: ||F_theta(s,c)||^2 measures "dynamics stopped," not "answer is
right." The readout CE loss does the real work. This undermines the "single error"
philosophy.

### Resolution: Three-Track Error Function Design

Instead of betting on E5, test three fundamentally different error philosophies:

**Track A: Embedding Regression (E1) — the honest baseline**
```
L = Sum_l ||s_T[l] - embed(y*[l])||^2
```
No pretension of a novel error function. Direct regression in embedding space.
The dynamics learn to move states toward target embeddings.
Convergence = close to target = correct readout.
This IS a unified error (same space, same metric for thinking and generating).

**Track B: Self-Consistency + Readout (E5) — the original proposal**
```
L = lambda_1 ||F_theta(s_T, c)||^2 + lambda_2 CE(R(s_T), y*)
```
Test empirically whether convergence correlates with correctness.
Add wrong-attractor diagnostic to measure spurious fixed points.
If wrong-attractor rate > 10%, E5 is not viable as proposed.

**Track C: Learned Energy Gradient (E5b) — Codex's suggestion, refined**

The energy must be target-independent at inference time. Define:
```
E_psi(s, c) = MLP([s; c]) -> R  (learned scalar energy)
```

Training:
- Positive (correct) states: s+ = embed(y*)
- Negative (incorrect) states: s- = corrupted embeddings (wrong tokens, shuffled)
- Contrastive loss: E_psi(s+, c) < E_psi(s-, c) by margin
- Full training: L = E_psi(s_T, c) + margin_loss(E_psi, s+, s-, c)

Inference:
- Dynamics: s_{t+1} = s_t - eta * grad_s E_psi(s_t, c)
- Convergence: ||grad_s E_psi|| < epsilon
- The energy landscape has basins at correct states

This is the most principled: the energy measures task correctness,
the dynamics minimize the energy, convergence = correct.

### Why keep all three tracks

- Track A tells us if continuous dynamics work AT ALL
- Track B tests the original UESD thesis as stated
- Track C tests the refined thesis with proper energy grounding
- If A works but B and C don't, the dynamics idea is viable but E5 is wrong
- If C works better than B, the "learned energy" insight is validated

---

## Fixing the Bottleneck Analysis (Exp 0)

### What was wrong
The analysis compared float32 capacity (storage bits) with mutual information
(semantic bits). These are different quantities.

### Revised framing
The bottleneck is about PARALLELISM, not total information:

**AR:** processes tokens SEQUENTIALLY. At each step, the model must commit
to one token. To reconsider, it must generate correction tokens later.
The sequential bottleneck limits the model's ability to explore multiple
continuations simultaneously.

**UESD:** refines ALL output positions SIMULTANEOUSLY. The model can explore
multiple options in parallel because the state is continuous (no sampling).
Position interactions happen through self-attention at each dynamics step.

The rate-distortion framing:
- To decode L tokens from V vocab with zero error: need I(s_T; y*) >= L*log2(V)
- AR provides this through sequential conditioning (each step adds log2(V) bits)
- UESD provides this through parallel refinement (all positions at once)
- The advantage is THROUGHPUT, not total capacity

---

## Revised Diagnostics (addressing missing measurements)

### Added to EVERY experiment:

1. **Decoder margin**: m(l) = cos(s_T[l], e_{y*_l}) - max_{v != y*_l} cos(s_T[l], e_v)
   Positive = correct with margin. Track over dynamics steps.

2. **Wrong-attractor count**: examples where ||F_theta(s_T)|| < epsilon
   but readout != y*. This directly measures spurious fixed points.

3. **Basin perturbation**: add noise to s_T, re-run dynamics for K more steps.
   Measure: does it return to the same readout? What fraction of perturbations
   change the answer? This measures basin size.

4. **Spectral radius of G(s) = s + F_theta(s, c)**: compute at convergence
   via power iteration on the Jacobian. Must be < 1 for local stability.

5. **Correlation: ||F_theta|| vs token accuracy**: across test set, compute
   Spearman rank correlation between convergence quality and correctness.
   If rho > 0.7, E5 is a valid proxy. If rho < 0.3, E5 is decoupled.

6. **Decoder margin trajectory**: plot m(l) at each dynamics step t.
   Should increase monotonically if dynamics are doing useful work.

---

## Revised Loss Schedule

### Drop path smoothness penalty
Codex correctly notes lambda_3 term penalizes motion, conflicting with the
exploration thesis. Remove it.

### Warm-up convergence loss
- Steps 0-5K: L = lambda_2 CE(R(s_T), y*) only (learn useful dynamics first)
- Steps 5K-10K: linearly ramp lambda_1 from 0 to target
- Steps 10K-20K: full loss with final lambda values

### Revised lambda defaults
- lambda_1 = 1.0 (convergence — no longer a minor regularizer)
- lambda_2 = 1.0 (readout — equal weight)
- Run a lambda sweep: lambda_1 in {0.01, 0.1, 1.0, 10.0}

---

## Added Experiment: One-to-Many Task

### Motivation (from Codex Intuition 3)
Multimodal outputs test whether UESD can represent uncertainty.

### Task: Partial Sort
Input: sequence with TIES (some tokens equal). Output: any valid sort.
Multiple valid outputs exist. E.g., [3,1,2,1] -> [1,1,2,3] or [1,1,2,3]
(identical in this case, but for [3,1,2,1,2] -> both [1,1,2,2,3] is unique,
so use a task with genuine ambiguity).

Better task: **Constrained completion.**
Input: [a, _, c, _, e] where _ is a wildcard meaning "any token".
Output: any sequence matching the pattern.
Multiple valid outputs. Tests whether UESD converges to ONE valid output
vs. averaging over many (which would produce garbage).

### Measurement
- Does UESD converge to a SINGLE valid output? (check constraints match)
- Does it converge to DIFFERENT valid outputs from different random s_0?
- Compare: AR naturally samples different valid outputs. Can UESD do the same
  with noise injection in Langevin dynamics?

---

## Priority Directive Response: Mathematical Derivation

### Theorem (to derive): Convergence-Correctness Coupling

Given:
- G(s) = s + F_theta(s, c) is the update map
- rho = spectral radius of dG/ds at fixed point s*
- r = ||F_theta(s*, c)|| (residual at convergence)
- m = decoder_margin(s*) = min_l [cos(s*[l], e_{y*_l}) - max_{v!=y*_l} cos(s*[l], e_v)]

We want to prove (or disprove) that:
- Low r implies high m (convergence => correctness)
- High m implies low r (correctness => convergence)
- rho < 1 is necessary for both

### Sketch of the derivation

If the training loss L = lambda_1 r^2 + lambda_2 CE(R(s_T), y*) converges to
L -> 0, then:
- lambda_1 r^2 -> 0 => r -> 0 (convergence)
- lambda_2 CE -> 0 => R(s_T) = y* (correctness)

So at training convergence, r and m ARE coupled (both are forced by the loss).

But at TEST TIME on unseen inputs, the coupling depends on GENERALIZATION:
- Does the model generalize the coupling to new inputs?
- Are there regions of S where r = 0 but m < 0 (spurious fixed points)?

Empirical test: measure wrong-attractor rate on held-out data.

Theoretical bound (from contraction theory):
If rho(dG/ds|_{s*}) < rho_max < 1, then s* is locally stable with basin
of radius proportional to (1 - rho_max). If the training distribution
covers S well enough, the basins of correct fixed points should cover most
of the reachable state space.

This is NOT a proof of convergence-correctness coupling. It's a sufficient
condition: IF rho < 1 AND basins cover the space AND no spurious fixed points
exist in reachable regions, THEN convergence implies correctness.

The empirical test is: wrong-attractor rate. If it's low (<5%), the coupling
holds in practice even without a theoretical guarantee.
