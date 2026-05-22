# Fixed-Point Existence and Uniqueness for UESD Dynamics

## Motivation

The convergence proofs in convergence_correctness.md and
finite_step_convergence.md assume a fixed point s* exists near the
trajectory. This document derives conditions under which fixed points
exist, are unique, and are reachable from the initial state.

---

## 1. The Contraction Mapping Theorem (Banach)

### Statement

Let (X, d) be a complete metric space and G: X -> X a contraction
mapping, i.e., there exists q in [0, 1) such that for all x, y in X:

    d(G(x), G(y)) <= q * d(x, y)

Then G has a unique fixed point x* in X, and for any x_0 in X:

    x_n = G^n(x_0) -> x* as n -> infinity

with error bound:

    d(x_n, x*) <= q^n / (1 - q) * d(x_0, G(x_0))

### Application Attempt to UESD

For UESD, X = R^{L*d} (the state space), G(s) = s + F_theta(s, c)
(dropping c notation for clarity). The contraction condition requires:

    ||G(s) - G(s')|| <= q * ||s - s'||   for all s, s' in X

This is equivalent to:

    ||dG/ds|| = ||I + dF/ds|| <= q < 1

Since ||I + dF/ds|| >= 1 - ||dF/ds|| by reverse triangle inequality,
we need ||dF/ds|| >= 1 - q > 0 AND all eigenvalues of I + dF/ds to
have modulus < 1. This is a STRONG condition — it requires dF/ds to
have eigenvalues near -1 everywhere, not just at the fixed point.

**Verdict:** Global contraction is too strong for UESD. The dynamics
need not be globally contractive; they only need local contractivity
near the fixed point (which is what Theorem 1 in
convergence_correctness.md provides).

---

## 2. Local Existence via the Implicit Function Theorem

### Setup

A fixed point s* satisfies H(s*, c) = 0 where H(s, c) = G(s, c) - s = F_theta(s, c).

### Theorem (Local Existence and Uniqueness)

If F_theta is C^1, and at a point (s_0, c_0) with F_theta(s_0, c_0) = 0,
the Jacobian dF/ds|_{s_0, c_0} is non-singular (i.e., dF/ds is invertible),
then there exist neighborhoods U of c_0 and V of s_0 such that for each
c in U, there exists a unique s*(c) in V with F_theta(s*(c), c) = 0.

Moreover, s*(c) is C^1 in c with:

    ds*/dc = -(dF/ds)^{-1} * dF/dc

**Proof.** Direct application of the implicit function theorem to
H(s, c) = F_theta(s, c) = 0, noting that dH/ds = dF/ds.  QED.

### Conditions for Non-Singularity

dF/ds = J - I where J = dG/ds. So dF/ds is non-singular iff J has no
eigenvalue equal to 1. Since rho(J) < 1 (our stability condition),
all eigenvalues of J are inside the unit disk, so no eigenvalue equals
1, and dF/ds is indeed non-singular.

**Corollary.** If training finds a fixed point s* with rho(J) < 1,
then the implicit function theorem guarantees that nearby contexts c'
also have unique fixed points s*(c') near s*.

This is the formal underpinning of Proposition 6 in
convergence_correctness.md.

---

## 3. Existence Without a Priori Knowledge of s*

The implicit function theorem requires starting from a known fixed
point. But does a fixed point exist at all for arbitrary context c?

### Brouwer's Fixed-Point Theorem

If S is a compact convex set in R^n and G: S -> S is continuous,
then G has at least one fixed point.

**Application attempt:** If the UESD dynamics map G(s, c) sends a
compact ball B_R = {s : ||s|| <= R} into itself, then a fixed point
exists. The condition is:

    ||G(s, c)|| <= R for all ||s|| <= R

i.e., ||s + F_theta(s, c)|| <= R.

For this to hold:

    ||F_theta(s, c)|| <= R - ||s||

At the boundary ||s|| = R: ||F_theta(s, c)|| <= 0, which is impossible
unless F = 0 there. So Brouwer does not apply directly.

### Leray-Schauder Alternative

A more practical approach: G(s, c) = s + F_theta(s, c) has a fixed
point (i.e., F_theta(s, c) = 0) if there exists R > 0 such that:

    For all s with ||s|| = R: s != lambda * G(s, c) for any lambda in [0, 1]

(The topological degree argument.) This is hard to verify analytically
for neural networks. Instead, we take the training-based approach.

---

## 4. Training-Guaranteed Existence (Practical Approach)

### Theorem (Training Creates Fixed Points)

If the training loss

    L(theta) = lambda_1 * ||F_theta(s_T, c)||^2 + lambda_2 * CE(R(s_T), y*)

converges to L < delta on training example (c, y*), then s_T is an
APPROXIMATE fixed point with residual:

    ||F_theta(s_T, c)|| < sqrt(delta / lambda_1)

**This is not a true fixed point** (F is not exactly zero) but it is
close. By the inverse function theorem, if dF/ds is non-singular
at s_T, then there exists a true fixed point s* near s_T with:

    ||s* - s_T|| <= ||dF/ds|_{s_T}^{-1}|| * ||F(s_T, c)||

### Bounding the True Fixed-Point Distance (Kantorovich Form)

Let A = dF/ds|_{s_T}. Assume:
- F is C^2 in a neighborhood of s_T
- A is non-singular with beta = ||A^{-1}|| = 1 / sigma_min(A)
- M = sup ||d^2F/ds^2|| in a neighborhood (second-derivative bound)
- r_T = ||F(s_T, c)|| (the residual)

**Theorem (Newton-Kantorovich).** If alpha = beta * M * r_T < 1/2,
then there exists a unique fixed point s* (i.e., F(s*, c) = 0) in the
ball B(s_T, r_+) where:

    r_+ = (1 - sqrt(1 - 2*alpha)) / (beta * M)

and the distance satisfies:

    ||s* - s_T|| <= r_+ <= 2 * beta * r_T / (1 + sqrt(1 - 2*alpha))

For small alpha (well-conditioned case), r_+ ~ beta * r_T = r_T / sigma_min(A).

**Proof.** Standard Kantorovich theorem applied to the Newton map
N(s) = s - A^{-1} F(s). The contraction condition ||I - A^{-1} dF/ds||
< 1 holds in B(s_T, r_+) because ||dF/ds - A|| <= M * r_+ and
M * beta * r_+ < 1.  QED.

**Non-normality caveat.** For non-normal A, the singular values
and eigenvalue moduli can differ significantly. The naive bound
sigma_min(A) >= 1 - rho(J) only holds for normal matrices. For
non-normal A:

    sigma_min(A) = 1 / ||A^{-1}||

which must be estimated directly (e.g., via SVD or randomized
methods). The bound beta = ||A^{-1}|| is the correct quantity,
not 1/(1 - rho(J)).

**Relation to prior bound.** The simpler bound

    ||s* - s_T|| <= ||F(s_T, c)|| / (1 - rho(J))

is a special case that holds when A is normal and rho(J) < 1
(since sigma_min(A) = 1 - rho for normal matrices with eigenvalues
in the real interval). For the general (non-normal) case, the
Kantorovich form with explicit beta is more appropriate.

### Numerical Example (Updated)

With lambda_1 = 0.1, delta = 0.01 (training loss):
- r_T = ||F(s_T, c)|| < sqrt(0.01 / 0.1) = sqrt(0.1) = 0.316
- If sigma_min(A) = 0.05 (i.e., beta = 20): ||s* - s_T|| <= 6.32
- If sigma_min(A) = 0.5 (i.e., beta = 2): ||s* - s_T|| <= 0.632
- Kantorovich condition: need beta * M * r_T < 0.5
  For beta = 20, r_T = 0.316: need M < 0.079

The Kantorovich condition provides a verifiable certificate: if it
holds, a unique nearby fixed point is guaranteed.

---

## 5. Uniqueness in the Reachable Basin

### Local Uniqueness

By the implicit function theorem (Section 2), if rho(J) < 1 at s*,
the fixed point is locally unique (in some neighborhood V of s*).

### Global Non-Uniqueness

Multiple fixed points can coexist. The UESD dynamics may have:

1. **Correct attractors:** s* with F(s*) = 0 and m(s*) > 0
2. **Wrong attractors:** s' with F(s') = 0 and m(s') < 0
   (Theorem 4 in convergence_correctness.md)
3. **Repellers:** Fixed points with rho(J) > 1

Training pushes the dynamics toward correct attractors on the
training distribution, but cannot eliminate wrong attractors globally.

### Basin Partition

The state space partitions into basins of attraction for different
fixed points. The training-time basin coverage determines the
practical fixed-point landscape. D4 (wrong-attractor rate) measures
the fraction of initial states that converge to wrong attractors.

---

## 6. Context-Dependence of Fixed Points

### Smooth Dependence on Context

From Section 2, if s*(c_0) is a stable fixed point (rho < 1), then
s*(c) exists and is C^1 for c near c_0. The sensitivity is:

    ds*/dc = -(dF/ds)^{-1} * dF/dc = -(J - I)^{-1} * dF/dc

This means:
- Smoother dynamics (smaller ||dF/dc||) → fixed point moves less with context
- Tighter contraction (smaller rho) → larger (1 - rho)^{-1} prefactor,
  but more robustly stable basin
- The net sensitivity ||ds*/dc|| <= ||dF/dc|| / (1 - rho)

### Bifurcation Risk

If rho(J) approaches 1 for some context c, the fixed point s*(c) can:

1. **Lose stability** (eigenvalue crosses unit circle)
2. **Bifurcate** (split into multiple fixed points)
3. **Disappear** (saddle-node bifurcation)

These are dynamical systems bifurcations. At the POC scale, they
manifest as:
- Sudden accuracy drops on certain inputs (bifurcation)
- High rho values on specific contexts (near-critical)
- Wrong-attractor formation (competing fixed points)

D6 (spectral radius) measured per-example can detect these.

---

## Summary

| Question | Answer | Condition |
|----------|--------|-----------|
| Does a fixed point exist? | Yes (approximately) | Training loss → 0 (Section 4) |
| Is it unique? | Locally, yes | rho(J) < 1 (Section 5) |
| Is it the right one? | Depends on training | CE → 0 ∧ margin > 0 (Theorem 5 in convergence_correctness.md) |
| Does it persist for nearby contexts? | Yes | IFT + rho < 1 (Section 2) |
| Can wrong fixed points exist? | Yes, always | Theorem 4 in convergence_correctness.md |
| How close is s_T to s*? | ||s* - s_T|| ≤ r / (1 - rho) | r = residual, rho = spectral radius |

The practical conclusion: UESD does not need a priori existence
guarantees. Training creates approximate fixed points, and rho < 1
bootstraps local existence/uniqueness via the implicit function
theorem. The remaining risk is wrong attractors, which D4 measures.
