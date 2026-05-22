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

**Result (Normal case, Theorems 1-3):** After T steps with spectral
radius rho:

    ||s_T - s*|| <= rho^T * ||s_0 - s*|| + O(||s_0 - s*||^2)

For T = 10:
- rho = 0.5: 99.9% reduction (essentially converged)
- rho = 0.8: 89% reduction (good)
- rho = 0.9: 65% reduction (borderline)
- rho = 0.95: 40% reduction (potentially insufficient)

**Result (Non-normal case, Theorem 4):** The rigorous finite-T
bound uses sigma_max(J) (largest singular value), not rho(J):

    ||s_T - s*|| <= sigma_max^T * ||s_0 - s*|| * (1 + O(||s_0 - s*||))

For non-normal J, sigma_max > rho (possibly sigma_max >= 1 even
when rho < 1). The non-normality ratio kappa = sigma_max / rho
quantifies this gap:
- kappa = 1: normal, Theorems 1-3 exact
- kappa in (1, 1.5): mild, practical for T = 10
- kappa > 2: severe transient growth, D5 is the reliable test

**Minimum T for correct readout:**
- Normal: T >= log(K * ||s_0 - s*|| / m(s*)) / log(1/rho)
- Non-normal: T >= log(K * ||s_0 - s*|| / m(s*)) / log(1/sigma_max)

If sigma_max >= 1, no finite T guarantees contraction. D5 (basin
perturbation) empirically validates finite-T stability regardless
of non-normality.

[Proof: finite_step_convergence.md, Theorems 1-4]

### 2.5 Fixed-Point Existence (fixed_point_existence.md)

**Result:** Global existence cannot be guaranteed for neural network
dynamics (Brouwer/Leray-Schauder conditions are hard to verify).
However:

1. **Training creates approximate fixed points:** Loss < delta implies
   ||F(s_T, c)|| < sqrt(delta/lambda_1).

2. **IFT bootstraps true fixed points:** If dF/ds is non-singular
   at s_T (guaranteed when rho(J) < 1), a true fixed point s*
   exists nearby. The Kantorovich bound gives:
       ||s* - s_T|| <= 2 * beta * r_T / (1 + sqrt(1 - 2*alpha_K))
   where beta = ||A^{-1}||, r_T = ||F(s_T)||, alpha_K = beta * M * r_T.
   The simpler bound ||F(s_T)|| / (1 - rho) holds only for normal A.

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

### 2.8 Wrong-Attractor Risk Under Shift (Theorem 7, convergence_correctness.md)

**Result:** If W_1(P, Q) <= eta between training and test distributions,
the wrong-attractor rate on Q is bounded by:

    Q(m(s*(c)) < 0) <= alpha + (K * L_s * eta) / gamma

where alpha is the training-time failure rate, gamma is the margin
buffer, K is the margin Lipschitz constant, and L_s is the fixed-point
sensitivity. The bound is tight in the Lipschitz sense.

[Proof: convergence_correctness.md, Theorem 7]

### 2.9 Dynamics-Decoder Separation (Theorem 8, convergence_correctness.md)

**Result:** The E5 gradient decomposes into three terms:
(a) SC gradient shapes dynamics convergence (phi parameters)
(b) CE gradient shapes readout alignment (psi parameters)
(c) Coupling term (dCE/ds * ds/dphi) transmits readout signal to dynamics

E1 failure on addition illustrates coupling breakdown: with only 0.1*CE,
the coupling term is too weak. The dynamics converge but to wrong
attractors (WA = 100%). E5 with full CE provides sufficient coupling.

[Proof: convergence_correctness.md, Theorem 8]

### 2.10 Information Bottleneck (information_bottleneck.md)

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
| D5: Basin perturbation | Basin size and non-normal stability? | 2.1, 2.4 (Thm 4), 2.7 | >= 90% |
| D6: Spectral radius | Is fixed point stable? | 2.1, 2.6 | < 1.0 |
| D7: σ_max/ρ ratio | Non-normality severity? | 2.4 (Thm 4) | < 1.5 (CE-dyn: 1.45-1.57; E5: 1.85-2.12) |
| D8: Trajectory Lyapunov | Product-Jacobian trajectory stability? | 2.4, 2.7 | lambda_max < 0 for stability; actual: 0.045-0.199 (edge of chaos) |

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
4. Convergence in T = 10 steps when sigma_max >= 1 (even if rho < 1)
5. Non-normal transient growth bounded analytically (Theorem 4 gives
   sigma_max^T bound; D5 empirically validates; D7 quantifies severity)
6. Spectral radius staying < 1 during training (only at convergence)

### Open Questions

1. Does sorting require dynamics? (Exp C: NO at L=8, V=64 — encoder confound)
2. Does addition require dynamics? (Exp D: 2L encoder fails at 73%;
   Exp D2: 4L/8L encoders learn addition, so dynamics NOT strictly
   necessary. BUT UESD is more parameter-efficient: 694K params for
   100% vs 1.6M for 99.98%. CE-dynamics (no SC) most robust.)
3. Can UESD scale beyond L = 8, V = 64? (Future work)
4. Is there a natural energy function better than ||F||^2? (Deferred —
   D2 shows CE-only training outperforms SC+CE, questioning whether
   explicit SC is beneficial)
5. What is the optimal rho for given T? (Theory says rho ~ 0.5-0.8;
   Exp D shows E5 addition achieves rho ~ 0.49-0.51)
6. Can implicit dynamics (DEQ-style) remove the stability constraint?
7. Why does E5 have a 40% failure rate on addition? (Exp D2: 2/5 seeds
   stuck at CE=2.08 wrong attractors. SC drives convergence before CE
   guides to correct basins. Phase transition is initialization-dependent.)
8. Why does CE-dynamics avoid the wrong-attractor trap? (Hypothesis:
   without SC pressure, dynamics remain flexible during early training,
   allowing CE gradient to reshape the attractor landscape continuously.)
9. Can the seeding bug (model init not controlled) change D2 conclusions?
   (Need re-run with proper set_seed before model creation.)
10. Can Jacobian rotation be controlled or optimized? (D3 shows it's
    the dominant stabilization mechanism. Can a loss term explicitly
    encourage rotation? Would that improve training stability?)
11. Does edge-of-chaos self-organization persist at larger scale?
    (D3: lambda_max in 0.045-0.199 at L=8 V=64. Does this hold at
    L=16, L=32, or with harder tasks?)
12. Can a tighter trajectory stability bound replace Theorem 4?
    (D3 shows Theorem 4 is conservative by up to 718x (corrected).
    A bound accounting for Jacobian rotation would be far more useful.)
13. Why do CE-dynamics and E5 develop DIFFERENT stability mechanisms?
    (D4: CE-dynamics uses Jacobian rotation/diversity, E5 uses sigma
    compression. SC penalty prevents the "exploring" phase where
    alignment drops. Is this fundamental or architecture-dependent?)
14. Does the three-phase regime (untrained/exploring/task-aligned) in
    CE-dynamics persist at larger scale? (D4: phase transition at step
    ~3000 with alignment dip→recovery. Scale-dependent?)
15. Can the O/S ratio diagnostic predict training failure? (D4: E5
    O/S >1 = directional structure learned, CE-dynamics O/S <1 =
    state-dependent diversity. Does O/S <0.5 predict divergence?)

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
- Finite-T error bounded by sigma_max^T * ||s_0 - s*|| (Theorem 4,
  rigorous one-step bound, induction over T steps)
- Generalization radius depends on Lipschitz constants (IFT)
- Wrong-attractor rate under shift bounded by Wasserstein distance
  (Theorem 7, requires Lipschitz constants of fixed-point map)

**WEAK claims (directional, not rigorous):**
- Non-normal effects are empirically moderate per-step but trajectory
  stability is governed by Jacobian rotation, not per-step sigma_max.
  D3/D3b trajectory analysis: Theorem 4 bound conservative by 718x
  (corrected). Actual amplification 1.58-7.46x vs predicted 49-5,112x.
  Mechanism: state-dependent Jacobian diversity (D3b: temporal ordering
  accounts for only ~7% of cancellation).
- CE-dynamics and E5 develop FUNDAMENTALLY DIFFERENT stability mechanisms
  (D4): CE-dynamics uses Jacobian rotation (alignment drops to 0.068
  during exploring phase, recovers to 0.67); E5 uses per-step sigma
  compression (alignment stays >0.77 throughout). Same architecture,
  different loss, qualitatively different dynamics.
- Dynamics self-organize to edge of chaos (lambda_max in 0.045-0.199)
  without explicit regularization toward this regime. E5 approaches
  true marginal stability (lambda_max=0.060 at 20K steps, D4).
- Optimal rho in [0.9, 1.0) (empirical, not theoretically derived)
- UESD's process advantage (parallel refinement) translates to
  practical gains (no benchmark evidence yet)

**OVERCLAIMS to avoid:**
- "UESD has higher information capacity than AR" (FALSE: both L*log(V))
- "Continuous state stores d*32 bits" (conflates storage with MI)
- "Self-consistency guarantees correctness" (FALSE: Theorem 4)
- "Spectral norm guarantees contraction" (FALSE: only bounds FFN)
- "Dynamics are necessary for addition" (WEAKENED: 4L/8L encoders
  learn addition; dynamics are more parameter-efficient, not necessary)
- "E5 reliably learns addition" (FALSE: 40% failure rate in D2 sweep)
- "SC loss is essential" (FALSE: CE-dynamics outperforms E5 in D2)
