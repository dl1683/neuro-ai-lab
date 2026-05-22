# Non-Normal Stability Analysis for UESD Dynamics

## Motivation

The spectral radius rho(J) < 1 guarantees ASYMPTOTIC stability of
fixed points (Theorem 1, convergence_correctness.md). But UESD uses
fixed T = 10 iterations, so transient behavior matters. Non-normal
Jacobians (J^H J != J J^H) can exhibit transient growth where
||J^k|| >> rho(J)^k for small k, even when rho < 1.

Self-attention creates non-normal Jacobians because the attention
weight matrix A = softmax(QK^T / sqrt(d_k)) is generally non-symmetric
and depends on the state s. This document analyzes the implications
for UESD's finite-step dynamics.

---

## 1. Normal vs Non-Normal Matrices

### Normal Matrices

A matrix J is normal if J^H J = J J^H. For normal J:

    ||J^k|| = rho(J)^k   for all k >= 0

(where ||.|| is the spectral norm). This means the spectral radius
EXACTLY controls the convergence rate at every step — no transient
growth.

### Non-Normal Matrices

For non-normal J, the spectral norm of powers can exceed rho^k:

    ||J^k|| <= C_J * rho(J)^k

where C_J >= 1 depends on the eigenvalue condition numbers. The
Kreiss matrix theorem gives a refined bound:

    rho(J) <= K(J) <= e * n * rho(J)   (Kreiss theorem)

where K(J) is the Kreiss constant:

    K(J) = sup_{|z|>1} (|z| - 1) * ||(zI - J)^{-1}||

and n is the matrix dimension. For n = L*d = 8*128 = 1024 in our
experiments, the Kreiss bound allows ||J^k|| up to 1024 * e * rho^k
for the worst case, though this is rarely tight.

---

## 2. The Pseudospectrum Perspective

### Definition

The epsilon-pseudospectrum of J is:

    sigma_eps(J) = {z in C : ||(zI - J)^{-1}|| >= 1/eps}
                 = {z in C : z in spectrum(J + E) for some ||E|| <= eps}

For normal J, sigma_eps(J) is just the eps-neighborhood of the
spectrum. For non-normal J, sigma_eps(J) can extend much further.

### Pseudospectral Radius

    rho_eps(J) = max{|z| : z in sigma_eps(J)}

For transient behavior, rho_eps for small eps controls the initial
growth rate:

    sup_{k>=0} ||J^k|| = lim_{eps->0} rho_eps(J)^{-1}   (simplified)

More precisely, the resolvent norm ||R(z, J)|| = ||(zI - J)^{-1}||
near the unit circle determines whether transient growth occurs.

### Why This Matters for UESD

If rho(J) = 0.9 but rho_{0.01}(J) = 1.1 (the 0.01-pseudospectrum
extends outside the unit disk), then for early iterations the dynamics
can AMPLIFY perturbations even though they eventually contract.

For T = 10 fixed iterations:
- If transient growth peaks at step k < T and decays by step T, no
  practical problem.
- If transient growth peaks at step k ~ T, the state at T may be
  FURTHER from the fixed point than the initial state.

---

## 3. The Logarithmic Norm (Measure of a Matrix)

### Definition

The logarithmic norm (matrix measure) of A is:

    mu(A) = lim_{h->0+} (||I + hA|| - 1) / h

For the spectral norm (2-norm):

    mu_2(A) = lambda_max((A + A^H) / 2)

i.e., the largest eigenvalue of the Hermitian part of A.

### One-Step Growth Bound

For the ODE dx/dt = Ax, the logarithmic norm gives:

    ||e^{tA}|| <= e^{mu(A) * t}

For the discrete map s_{t+1} = J * s_t:

    ||s_{t+1} - s*|| <= ||J|| * ||s_t - s*||

The spectral norm ||J|| (not rho(J)) controls the one-step growth.
For normal J, ||J|| = rho(J). For non-normal J, ||J|| > rho(J).

### Application to UESD

For J = I + dF/ds, the one-step amplification factor is:

    ||J|| = ||I + dF/ds|| <= 1 + ||dF/ds||

But this can also be less than 1 if dF/ds has eigenvalues near -1:

    ||J|| = sigma_max(J)

The one-step bound is:

    ||s_{t+1} - s*|| <= sigma_max(J) * ||s_t - s*||

After T steps (without linearization update — using J at s*):

    ||s_T - s*|| <= sigma_max(J)^T * ||s_0 - s*||   (approximate)

This is TIGHTER than rho(J)^T when sigma_max > rho (non-normal),
and IDENTICAL when J is normal.

**Formal statement:** See finite_step_convergence.md, Theorem 4 for
the rigorous version with quadratic remainder, readout preservation
conditions, and the non-normality ratio κ = σ_max/ρ diagnostic.

---

## 4. Decomposing the UESD Jacobian

### Architecture of J

    J = dG/ds = I + d(SelfAttn)/ds + d(CrossAttn)/ds + d(FFN)/ds

(simplified; actual has layer norms and residual connections).

#### Self-Attention Contribution

For self-attention with Q = W_Q s, K = W_K s, V = W_V s:

    d(SelfAttn)/ds = W_O * [d(A * V * W_V)/ds]

The attention matrix A = softmax(QK^T / sqrt(d_k)) depends on s,
making d(SelfAttn)/ds depend on the current state. The Jacobian
involves terms like:

    dA/ds * V * W_V  +  A * W_V

The first term (attention gradient) is the primary source of
non-normality. It is rank-bounded by the number of heads and
creates asymmetric coupling between positions.

#### Cross-Attention Contribution

d(CrossAttn)/ds only involves the query gradient (context is fixed):

    d(CrossAttn)/ds = W_O * A_cross * W_V_cross * 0 + W_O * (dA_cross/ds) * V_cross

Since V_cross = W_V * context is independent of s:

    d(CrossAttn)/ds = W_O * (dA_cross/dQ * W_Q) * V_cross

This is lower rank than self-attention (no V gradient contribution).

#### FFN Contribution

d(FFN)/ds = W_2 * diag(ReLU'(W_1 * s)) * W_1

With spectral normalization: sigma(W_1) <= 1, sigma(W_2) <= 1.
So ||d(FFN)/ds|| <= 1. This is the best-controlled component.

### Non-Normality Sources (Ranked)

1. **Self-attention gradient** (primary) — state-dependent, asymmetric
2. **Cross-attention gradient** (secondary) — lower rank, state-dependent
3. **FFN** (minor) — spectral norm controlled, nearly normal
4. **Layer norm Jacobian** (minor) — introduces mild non-normality

---

## 5. Practical Bounds for T = 10

### Worst-Case Analysis

Let sigma_1 = sigma_max(J) be the largest singular value. If sigma_1 < 1:

    ||s_T - s*|| <= sigma_1^T * ||s_0 - s*||

This is the guaranteed contraction bound, holding for ANY J (normal or not).

For sigma_1 = 0.95: sigma_1^10 = 0.60 (40% reduction, same as rho case)
For sigma_1 = 1.05: sigma_1^10 = 1.63 (63% AMPLIFICATION — bad)
For sigma_1 = 1.10: sigma_1^10 = 2.59 (159% amplification — very bad)

So the SINGULAR VALUE (not eigenvalue) controls finite-step behavior.

### When rho < 1 But sigma_max > 1

This is the dangerous regime for non-normal dynamics:
- Spectral radius says "converges eventually"
- Singular values say "amplifies in the short term"

For T = 10, if sigma_max > 1 but sigma_max^10 * ||s_0 - s*|| is
still within the basin of correct readout (m(s*)/K from Theorem 2
in convergence_correctness.md), then the readout is still correct
despite transient growth.

**Corrected Readout Condition (Non-Normal Case):**

    m(s*) > K * sigma_max(J)^T * ||s_0 - s*||

Compare to the normal case:

    m(s*) > K * rho(J)^T * ||s_0 - s*||

The non-normal case requires a larger margin to absorb transient growth.

---

## 6. Empirical Detection via D5 and D6

### D6 (Power Iteration) Measures rho, Not sigma_max

The power iteration in diagnostics.py converges to the largest
EIGENVALUE (in magnitude), which is rho(J). It does NOT measure
sigma_max(J). To measure sigma_max, we would need:

    sigma_max(J) = sqrt(rho(J^H J))

i.e., power iteration on J^H J, which requires computing J^H v
(the adjoint-vector product). With finite differences this requires
an additional forward pass per iteration.

### D5 (Basin Perturbation) Tests the Actual Transient Behavior

D5 adds noise sigma * ||s_T|| to the converged state and checks
readout preservation. This EMPIRICALLY tests whether transient
growth (if any) is small enough to stay in the correct basin.

If D5 stability is high (>= 90%) despite rho near 1.0, then
transient growth is not a practical issue — either J is nearly
normal, or the basin is large enough to absorb it.

### Proposed Additional Diagnostic: D7 (Singular Value Ratio)

To directly measure non-normality, compute:

    kappa = sigma_max(J) / rho(J)

For normal J: kappa = 1.
For highly non-normal J: kappa >> 1.

If kappa > 1.5 and rho > 0.8, warn that finite-T dynamics may
exhibit transient growth. This diagnostic is deferred to future
work but noted here for completeness.

---

## 7. Spectral Normalization and Non-Normality

### What Spectral Norm Controls

Spectral normalization on the FFN layers ensures:

    sigma_max(W_1) = 1, sigma_max(W_2) = 1

This bounds the FFN contribution to J:

    ||d(FFN)/ds||_2 <= sigma_max(W_2) * sigma_max(W_1) = 1

But it does NOT control the attention contributions, which are the
primary source of non-normality.

### Reducing Non-Normality

Potential strategies (not implemented in POC):

1. **Attention weight regularization:** Penalize ||A - A^T|| to
   encourage symmetric attention (reduces non-normality but may
   hurt expressiveness).
2. **Spectral norm on attention projections:** Apply spectral
   normalization to W_Q, W_K, W_V (controls all contributions
   but may be too restrictive).
3. **Damped dynamics:** Use G(s,c) = (1-alpha)*s + alpha*F(s,c)
   with alpha < 1 (reduces effective step size, controls singular
   values but slows convergence).
4. **Direct sigma_max regularization:** Penalize sigma_max(J)
   during training (expensive — requires SVD or power iteration
   in the training loop).

For the POC, spectral normalization on FFN + empirical D5/D6
monitoring is sufficient. More sophisticated control is future work.

---

## 8. Connection to Numerical ODE Stability

### UESD as a Forward Euler Discretization

The UESD dynamics can be viewed as forward Euler integration of:

    ds/dt = F_theta(s, c)

with step size h = 1 (one dynamics step per unit time). The stability
region of forward Euler for the linear test equation ds/dt = lambda*s is:

    |1 + h*lambda| < 1

i.e., lambda must lie in the disk |z + 1| < 1 (same as the spectral
contraction condition!). This is not a coincidence — the UESD update
G(s) = s + F(s) IS forward Euler with h = 1.

### Stiffness and Non-Normality

If the Jacobian dF/ds has eigenvalues with large imaginary parts
(from attention — see Section 4), the forward Euler method may require
small step sizes for stability. With fixed h = 1:

- Purely real eigenvalues in (-2, 0) → stable
- Complex eigenvalues with |Im(lambda)| > sqrt(3) → potentially
  unstable even with Re(lambda) in (-2, 0)

This is the disk condition |1 + lambda| < 1 being violated by
large imaginary parts. The attention-induced imaginary eigenvalues
are the primary risk for instability in the UESD dynamics.

### Implication for Design

An implicit method (backward Euler or trapezoidal) would have
unconditional stability for Re(lambda) < 0, regardless of imaginary
parts. This corresponds to solving:

    s_{t+1} = s_t + F(s_{t+1}, c)   (backward Euler)

which is exactly the DEQ formulation. The fact that UESD uses
forward Euler (explicit dynamics) instead of backward Euler
(implicit/DEQ) is a deliberate tradeoff:

- **Forward Euler (UESD):** Cheap per step, parallelizable, but
  stability requires eigenvalues in the unit disk.
- **Backward Euler (DEQ):** Unconditionally stable for dissipative
  dynamics, but requires fixed-point solve at each step (expensive).

For the POC with T = 10, forward Euler is appropriate. For
production with harder tasks and larger T, implicit methods may
be necessary.

---

## Summary

| Concern | Severity for UESD POC | Mitigation |
|---------|----------------------|------------|
| Transient growth (sigma_max > 1 despite rho < 1) | MODERATE | D5 empirically tests; margin absorbs small growth |
| Attention-induced non-normality | LOW-MODERATE | FFN spectrally normed; attention is lower-rank |
| Forward Euler stability limits | LOW at T=10 | Disk condition checked by D6; eigenvalues trained into disk |
| Large imaginary eigenvalues | UNCERTAIN | Not directly measured; D5 is indirect test |

**Bottom line:** Non-normality is a theoretical concern that is
EMPIRICALLY mitigated by D5 (basin perturbation). If D5 stability
is high and D6 (rho) < 1, the dynamics are practically stable for
the POC. A future D7 diagnostic (sigma_max/rho ratio) would directly
quantify non-normality, but is not required for the current study.
