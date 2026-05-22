# Spectral Contraction Analysis for UESD Dynamics

## Motivation

The UESD dynamics iterate G(s, c) = s + F_theta(s, c) until convergence.
For this to produce reliable fixed points, we need the update map G to be
locally contractive (spectral radius rho < 1). This document derives the
relationship between the dynamics architecture, spectral normalization,
and the contraction properties.

---

## 1. The Dynamics Map

The UESD dynamics block is a single TransformerDecoderLayer applied T times:

    G(s, c) = s + SelfAttn(s) + CrossAttn(s, c) + FFN(s)

(simplified; actual computation involves pre-norm and residual connections
within each sub-layer). The Jacobian is:

    J = dG/ds = I + dSelfAttn/ds + dCrossAttn/ds + dFFN/ds

where I is the identity matrix.

The spectral radius rho(J) = max|lambda_i(J)|.

### Key Insight

Since J = I + dF/ds, the eigenvalues of J are:

    lambda_i(J) = 1 + lambda_i(dF/ds)

For rho(J) < 1, we need |1 + lambda_i(dF/ds)| < 1 for all
eigenvalues lambda_i of dF/ds. Geometrically, the eigenvalues of
dF/ds must lie in the open disk of radius 1 centered at -1 in the
complex plane. This is more restrictive than just requiring
Re(lambda_i) in (-2, 0) -- the imaginary part must also be bounded
so that the modulus stays below 1.

**Caveat for non-normal Jacobians.** If J is non-normal (J^H J != J J^H),
the spectral radius alone does not control transient behavior. The
pseudospectrum sigma_epsilon(J) or the logarithmic norm mu(J) may be
more appropriate. Transient amplification with ||J^k|| >> 1 for small k
is possible even when rho(J) < 1. For our fixed-T dynamics (T=10),
transient growth matters: we need ||J^T|| to be controlled, not just
rho(J)^T as T -> infinity. In practice, D6 measures the asymptotic
spectral radius, and D5 (basin perturbation) empirically tests the
finite-T behavior.

---

## 2. Effect of Spectral Normalization

### What Spectral Norm Does

Spectral normalization (Miyato et al., 2018) constrains the spectral
norm (largest singular value) of each weight matrix W:

    sigma(W) <= 1

Applied to the FFN linear layers (linear1: d -> d_ff, linear2: d_ff -> d):

    sigma(W_1) <= 1, sigma(W_2) <= 1

### What It Does NOT Do

Spectral normalization on individual layers does NOT guarantee that the
composed function has spectral norm <= 1. For the FFN:

    FFN(x) = W_2 * ReLU(W_1 * x)

    ||dFFN/dx|| = ||W_2 * diag(ReLU'(W_1 * x)) * W_1||
               <= sigma(W_2) * sigma(W_1)
               <= 1

So the FFN sub-block alone has spectral norm <= 1. But the full
dynamics include self-attention and cross-attention, which can amplify.

### What We Actually Need

For the full Jacobian J = I + dF/ds, we need rho(J) < 1. Since:

    rho(J) <= ||J|| = ||I + dF/ds|| <= 1 + ||dF/ds||

This bound is useless (rho could be up to 1 + ||dF/ds||). But if the
eigenvalues of dF/ds lie in Re(z) < 0 (the dynamics are dissipative),
then rho(J) < 1.

The self-attention mechanism can create both dissipative and amplifying
modes. Spectral normalization on the FFN helps but does not guarantee
global contraction.

---

## 3. Empirical Measurement via Power Iteration

### Algorithm

Given state s and context c, estimate rho(J) where J = dG/ds|_s:

    1. Initialize random direction v ~ N(0, I), normalize to ||v|| = 1
    2. For k = 1 to K (power iterations):
       a. Compute J*v using finite differences:
          J*v ~ (G(s + eps*v, c) - G(s - eps*v, c)) / (2*eps)
       b. Compute lambda_k = ||J*v|| / ||v||
       c. Normalize: v <- J*v / ||J*v||
    3. Return lambda_K as estimate of rho(J)

### Convergence

Power iteration converges to the largest eigenvalue (in magnitude) at
rate |lambda_2 / lambda_1| per iteration. For K = 10 iterations, the
estimate is accurate to within |lambda_2 / lambda_1|^10 relative error.

### Interpretation of Results

| rho(J) | Interpretation |
|--------|---------------|
| < 0.9 | Strongly contractive. Fast convergence. May collapse dynamics. |
| 0.9 - 0.99 | Well-contractive. Good convergence with rich dynamics. |
| 0.99 - 1.0 | Weakly contractive. Slow convergence, possible near-critical behavior. |
| 1.0 | Critical. Fixed point is marginally stable. |
| > 1.0 | Unstable. Dynamics diverge from the fixed point. |

**Expected behavior:** For well-trained UESD on copy/reversal, we expect
rho in [0.9, 1.0). If rho > 1.0, the fixed point is unstable and the
system may oscillate or diverge.

---

## 4. Contraction and Basin Size

### Theorem (Basin of Attraction Radius)

If G is C^2 in a neighborhood of s* and rho(J|_{s*}) = rho < 1, then
the basin of attraction B(s*) contains a ball of radius:

    r >= delta / (M * (1 + delta))

where:
- delta = 1 - rho (contraction margin)
- M = sup_{s near s*} ||d^2G/ds^2|| (second derivative bound)

**Proof sketch.** For ||s - s*|| = r:

    ||G(s) - s*|| = ||J(s - s*) + O(r^2)||
                  <= rho * r + M * r^2 / 2

For contraction (||G(s) - s*|| < r):

    rho * r + M * r^2 / 2 < r
    M * r / 2 < 1 - rho = delta
    r < 2 * delta / M

So the basin radius scales as delta/M. Tighter contraction (smaller
rho) and smoother dynamics (smaller M) give larger basins.

### Practical Implication

For the basin perturbation test (D5), we add noise
sigma = 0.1 * ||s_T|| and check if readout is preserved. If the basin
radius r ~ delta/M is larger than this perturbation, the readout should
be stable. The fraction of stable examples measures the effective basin
coverage.

---

## 5. Spectral Radius and Training Dynamics

### During Training

The spectral radius rho changes during training as theta updates. Early
in training (random weights), rho is typically > 1 (no convergent fixed
points). As training progresses:

1. The CE loss drives the readout toward correct tokens.
2. The self-consistency loss (in E5) penalizes ||F(s_T)||^2, which
   encourages the dynamics to have fixed points.
3. Spectral normalization bounds the FFN contribution, stabilizing
   the eigenvalue spectrum.

### Lambda_1 and Spectral Radius

The self-consistency penalty lambda_1 * ||F(s_T)||^2 does not directly
control the spectral radius. It penalizes the residual MAGNITUDE, not
the STABILITY of the fixed point. However:

- Lower residual often correlates with lower rho (the dynamics are
  "more settled" near the fixed point).
- High lambda_1 can force the dynamics to collapse (rho << 1, trivial
  fixed points with little useful computation).
- The optimal lambda_1 balances convergence quality with dynamics
  richness.

This is tested empirically in the lambda sweep {0, 0.1, 1.0, 10.0}.

---

## 6. Connection to DEQ Literature

Deep Equilibrium Models (Bai et al., 2019) find fixed points of
weight-tied layers using Anderson acceleration and implicit
differentiation. Key differences from UESD:

| Aspect | DEQ | UESD |
|--------|-----|------|
| Fixed-point finding | Anderson accel / Broyden | Fixed T iterations |
| Backprop | Implicit (adjoint) | Explicit (BPTT through T steps) |
| Convergence guarantee | Requires contraction | Not required (empirical) |
| Target | Fixed point s* s.t. G(s*, c) = s* | State that reads out correctly |

UESD does not require convergence to a true fixed point. It requires
that after T steps, the readout is correct. The residual may be small
but non-zero. This is a weaker requirement that may allow richer
dynamics than strict DEQ convergence.

DEQ's implicit differentiation (Jacobian-free backprop at the fixed
point) could be applied to UESD for memory efficiency, but is not
needed at the POC scale (T=10, d=128).

---

## Summary

1. Spectral normalization on FFN layers bounds ||dFFN/ds|| <= 1 but
   does not guarantee contraction of the full update map.
2. The spectral radius rho of the full Jacobian J = dG/ds must be
   measured empirically (D6, power iteration).
3. rho < 1 implies local stability. The basin of attraction scales
   as (1-rho) / M where M bounds the second derivative.
4. The self-consistency loss and spectral radius are correlated but
   not causally linked. Lambda_1 controls residual magnitude,
   not eigenvalue structure.
5. The optimal regime for UESD is rho in [0.9, 1.0) — contractive
   enough for stability, rich enough for useful dynamics.
