# UESD Theoretical Framework: Consolidated Summary

## Purpose

This document provides a self-contained summary of the theoretical
framework for Unified Error-Space Dynamics (UESD), consolidating
results from six proof documents. It is intended as the reference
for the theory section of any future publication.

---

## 1. Architecture and Notation

UESD generates output by iterating a dynamics map in continuous
embedding space:

    s_0 = PosEmbed(output positions)     [initialization]
    s_t = G(s_{t-1}, c) for t = 1..T     [dynamics]
    y = argmax readout(s_T)              [readout]

where:
- c = Encode(input) is the encoded context
- G(s, c) = s + F_theta(s, c) is the update map
- F_theta is a weight-tied TransformerDecoderLayer (self-attn + cross-attn + FFN)
- readout(s) = cosine_sim(W_R s, E) / tau for embedding matrix E, temperature tau
- T = 10 dynamics steps (fixed, not adaptive)

The parameters theta are trained with loss:

    L = lambda_1 ||F_theta(s_T, c)||^2 + CE(readout(s_T), y*)

where the first term (self-consistency / E5) encourages fixed-point
convergence, and the second term (cross-entropy) drives correct readout.

---

## 2. Core Theoretical Results

### 2.1 Fixed-Point Stability (Theorems 1-3, convergence_correctness.md)

**Result:** If s* is a fixed point with spectral radius rho(J) < 1
where J = dG/ds|_{s*}, then s* is locally asymptotically stable:
trajectories starting near s* converge geometrically at rate rho.

**Result:** If the readout margin m(s*) > 0 (correct class logit
dominates), then the readout is stable under state perturbations
of magnitude < m(s*) / K, where K is the Lipschitz constant of
the margin function.

**Result:** Combining stability and margin: there exists a basin
B(s*) of radius min(eps_stability, m(s*)/K) such that any
trajectory starting in B(s*) converges to s* and produces
correct readout after sufficient steps.

[Proof: convergence_correctness.md, Theorems 1-3]

### 2.2 Wrong Attractors Exist (Theorem 4, convergence_correctness.md)

**Result:** Convergence (r -> 0) does NOT imply correctness (m > 0)
in general. Counterexample: identity dynamics G(s) = s has every
point as a fixed point, but only some produce correct readout.

[Proof: convergence_correctness.md, Theorem 4]

### 2.3 Training Creates the Coupling (Theorem 5, convergence_correctness.md)

**Result:** If the composite loss L = lambda_1 * ||F||^2 + CE converges
to delta, then:

    ||F(s_T, c)|| < sqrt(delta / lambda_1)    [approximate fixed point]
    p(y* | s_T) >= exp(-delta / lambda_2)       [high readout probability]

The decoder margin is positive when CE < log(2) (equivalently
p(y*) > 0.5). For well-trained models with CE << 0.1:

    m >= tau * (-CE - log(1 - exp(-CE)))

Numerical: CE < 0.01 gives m >= 0.461 (at tau = 0.1).

**Scope:** This coupling holds on the TRAINING distribution only.
Generalization depends on Lipschitz continuity of the dynamics and
readout (Proposition 6).

[Proof: convergence_correctness.md, Theorem 5]

### 2.4 Finite-Step Convergence (finite_step_convergence.md)

**Result:** After T steps with spectral radius rho:

    ||s_T - s*|| <= (rho + epsilon)^T * ||s_0 - s*|| + O(||s_0 - s*||^2)

where epsilon accounts for non-normality.

For T = 10:
- rho = 0.5: 99.9% reduction (essentially converged)
- rho = 0.8: 89% reduction (good)
- rho = 0.9: 65% reduction (borderline)
- rho = 0.95: 40% reduction (potentially insufficient)

**Minimum T for correct readout** (linear approximation):

    T >= log(K * ||s_0 - s*|| / m(s*)) / log(1/rho)

[Proof: finite_step_convergence.md, Theorems 1-3]

### 2.5 Fixed-Point Existence (fixed_point_existence.md)

**Result:** Global existence cannot be guaranteed for neural network
dynamics (Brouwer/Leray-Schauder conditions are hard to verify).
However:

1. **Training creates approximate fixed points:** Loss < delta implies
   ||F(s_T, c)|| < sqrt(delta/lambda_1).

2. **IFT bootstraps true fixed points:** If dF/ds is non-singular
   at s_T (guaranteed when rho(J) < 1), a true fixed point s*
   exists within distance ||F(s_T)|| / (1 - rho) of s_T.

3. **Smooth context dependence:** For nearby contexts c', fixed
   points s*(c') exist and depend smoothly on c via IFT.

[Proof: fixed_point_existence.md, Sections 2, 4]

### 2.6 Spectral Contraction (spectral_contraction.md)

**Result:** The dynamics Jacobian J = I + dF/ds has eigenvalues
lambda_i(J) = 1 + lambda_i(dF/ds). For rho(J) < 1, the eigenvalues
of dF/ds must lie in the open disk |z + 1| < 1 centered at -1.

Spectral normalization on FFN layers bounds ||dFFN/ds|| <= 1 but
does NOT guarantee contraction of the full map (attention can amplify).
The spectral radius must be measured empirically (D6, power iteration).

The basin of attraction scales as delta/M where delta = 1 - rho
(contraction margin) and M bounds the second derivative.

[Proof: spectral_contraction.md, Sections 1-4]

### 2.7 Non-Normal Stability (nonnormal_stability.md)

**Result:** For non-normal Jacobians (J^H J != J J^H), transient
growth is possible: ||J^k|| >> rho(J)^k for small k. The relevant
bound for finite-T dynamics is:

    ||s_T - s*|| <= sigma_max(J)^T * ||s_0 - s*||

where sigma_max is the largest singular value (not eigenvalue).

Self-attention is the primary source of non-normality. D6 measures
rho (eigenvalue), not sigma_max (singular value). D5 (basin
perturbation) empirically tests finite-step stability regardless
of non-normality.

The UESD dynamics are mathematically equivalent to forward Euler
integration of ds/dt = F(s,c) with step size h = 1. The stability
region is the same disk |1 + lambda| < 1.

[Proof: nonnormal_stability.md, Sections 1-5, 8]

### 2.8 Information Bottleneck (information_bottleneck.md)

**Result:** At each AR generation step, MI through the softmax
bottleneck is bounded by:

    I(h_t; x_t) <= H(x_t) <= log_2(V)

Total output information for both AR and UESD is L * log_2(V) bits
(identical). The advantage of UESD is NOT higher capacity but the
generation PROCESS:

1. No premature commitment (all positions refined simultaneously)
2. Continuous state preservation between steps
3. Error correction via continued refinement
4. Parallel throughput (single step refines all L positions)

[Proof: information_bottleneck.md, Sections 1-5]

---

## 3. Diagnostic-to-Theory Mapping

| Diagnostic | Measures | Theorems | Threshold |
|-----------|----------|----------|-----------|
| D1: Token accuracy | Does readout work? | — | >= 90% |
| D2: Normalized residual | How close to fixed point? | 2.4 (finite T) | < 0.01 |
| D3: Decoder margin | How confident is readout? | 2.1 (margin preservation) | > 0 |
| D4: Wrong-attractor rate | Convergence-correctness coupling? | 2.2, 2.3 | < 5% |
| D5: Basin perturbation | Basin size and non-normal stability? | 2.1, 2.7 | >= 90% |
| D6: Spectral radius | Is fixed point stable? | 2.1, 2.6 | < 1.0 |

---

## 4. What the Theory Does and Does Not Guarantee

### Guaranteed (with conditions)

1. Local stability of fixed points with rho < 1
2. Correct readout within the basin of attraction if margin > 0
3. Training creates approximate fixed points as loss -> 0
4. Nearby contexts have nearby fixed points (IFT)
5. MI through softmax bounded by log_2(V) per step

### NOT Guaranteed

1. Global existence of correct fixed points for arbitrary contexts
2. Absence of wrong attractors (only empirically testable via D4)
3. Generalization beyond training distribution (only Lipschitz bounds)
4. Convergence in T = 10 steps for rho > 0.9
5. Non-normal transient growth bounded (only D5 empirically tests)
6. Spectral radius staying < 1 during training (only at convergence)

### Open Questions

1. Does sorting require dynamics? (Exp C tests this)
2. Can UESD scale beyond L = 8, V = 64? (Future work)
3. Is there a natural energy function better than ||F||^2? (Deferred)
4. What is the optimal rho for given T? (Theory says rho ~ 0.5-0.8)
5. Can implicit dynamics (DEQ-style) remove the stability constraint?

---

## 5. Claim Calibration

The following claims are defensible based on the theory:

**STRONG claims (rigorous):**
- Softmax MI <= log_2(V) per step (DPI, standard)
- rho < 1 implies local stability (standard dynamical systems)
- Training loss -> 0 implies approximate convergence + correct readout
  on training data

**MODERATE claims (conditionally valid):**
- Basin size scales as (1-rho)/M (local linearization, holds near s*)
- Finite-T error bounded by rho^T * ||s_0 - s*|| (linearization)
- Generalization radius depends on Lipschitz constants (IFT)

**WEAK claims (directional, not rigorous):**
- Non-normal effects are empirically mild (D5 shows this, no proof)
- Optimal rho in [0.9, 1.0) (empirical, not theoretically derived)
- UESD's process advantage (parallel refinement) translates to
  practical gains (no benchmark evidence yet)

**OVERCLAIMS to avoid:**
- "UESD has higher information capacity than AR" (FALSE: both L*log(V))
- "Continuous state stores d*32 bits" (conflates storage with MI)
- "Self-consistency guarantees correctness" (FALSE: Theorem 4)
- "Spectral norm guarantees contraction" (FALSE: only bounds FFN)
