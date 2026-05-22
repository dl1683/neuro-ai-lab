# Convergence-Correctness Coupling in UESD

## Motivation

The central question from Codex review: does low fixed-point residual
||F_theta(s,c)||^2 imply correct readout? This document derives the
mathematical relationship between three quantities:

1. **r** = ||F_theta(s*, c)|| -- residual at convergence
2. **m(s*)** = decoder margin at the converged state
3. **rho** = spectral radius of the Jacobian dG/ds at s*

---

## Setup and Notation

- S = R^{L x d} is the state space (L output positions, d dimensions each)
- C = R^{L_in x d} is the context space (encoded input)
- G(s, c) = s + F_theta(s, c) is the full update map (one dynamics step)
- F_theta: S x C -> S is the dynamics function (residual update)
- s* is a fixed point of G for context c: G(s*, c) = s*, i.e., F_theta(s*, c) = 0
- J = dG/ds|_{s*,c} is the Jacobian of the update map at the fixed point
- rho(J) = spectral radius of J = max|eigenvalue(J)|
- R: S -> R^{L x V} is the readout function (cosine similarity / tau)
- e_v in R^d is the embedding of token v
- m_l(s) = cos(s[l], e_{y*_l}) - max_{v != y*_l} cos(s[l], e_v) is the per-position decoder margin
- m(s) = min_l m_l(s) is the worst-case decoder margin

---

## Theorem 1: Local Stability of Fixed Points

**Statement.** If rho(J) < 1, then s* is a locally asymptotically stable
fixed point. There exists epsilon > 0 and constant C >= 1 such that for
all s_0 with ||s_0 - s*|| < epsilon:

    ||G^t(s_0, c) - s*|| <= C * rho(J)^t * ||s_0 - s*||

where G^t denotes t-fold composition.

**Proof.** Standard result from discrete dynamical systems theory (see
e.g., Strogatz, Nonlinear Dynamics and Chaos, Ch. 10). Since rho(J) < 1,
all eigenvalues of J lie strictly inside the unit disk. By the Gelfand
formula and continuity of the spectrum, there exists a submultiplicative
matrix norm ||.||_* such that ||J||_* = rho(J) + delta for arbitrarily
small delta > 0. In a neighborhood of s*:

    G(s, c) - s* = J(s - s*) + O(||s - s*||^2)

So for ||s_0 - s*|| sufficiently small, the iterates contract
geometrically at rate rho(J) + delta.  QED.

**Caveat: non-normal Jacobians.** When J is non-normal (J^T J != J J^T),
the constant C in the bound can be large — iterates may transiently
GROW before eventually contracting. The pseudospectral radius or
logarithmic norm gives tighter transient bounds, but rho(J) < 1 still
guarantees eventual convergence. For UESD with T=10 fixed steps, the
relevant question is whether T is large enough for transient growth
to decay. See finite_step_convergence.md for the T-sufficiency analysis.

**Consequence for UESD.** Gate D6 requires rho < 1.0. If satisfied,
the UESD dynamics locally converge to the fixed point. This is necessary
but not sufficient for correctness.

---

## Theorem 2: Margin Preservation Under State Perturbation

**Statement.** Let R: R^{L x d} -> R^{L x V} be the readout function,
and define the margin function m(s) = min_l m_l(s). Suppose R is
K-Lipschitz in the following sense: for all s, s' in S:

    |m(s) - m(s')| <= K * ||s - s'||

If m(s*) > 0 (correct readout at the fixed point), then for any state s
with ||s - s*|| < m(s*) / K, we have m(s) > 0 (also correct readout).

**Proof.** Direct from Lipschitz continuity:

    m(s) >= m(s*) - |m(s) - m(s*)| >= m(s*) - K||s - s*||

If ||s - s*|| < m(s*)/K, then m(s) > 0.  QED.

**Estimating K.** The readout is:

    sim(l, v) = cos(h[l], e_v) / tau, where h = W_R * s

The margin m_l = sim(l, y*_l) - max_{v != y*_l} sim(l, v). Since
cosine similarity is 1-Lipschitz in each argument (when inputs have
bounded norm), and W_R is a linear map with operator norm ||W_R||:

    K <= 2 * ||W_R|| / (tau * min_l ||s[l]||)

For our configuration (tau = 0.1, ||W_R|| ~ O(1) after spectral norm):

    K ~ O(20 / min ||s[l]||)

This means the "safe radius" around a correct fixed point is:

    r_safe = m(s*) / K ~ m(s*) * tau * min||s[l]|| / (2 * ||W_R||)

---

## Theorem 3: Basin of Correct Convergence

**Statement.** Combining Theorems 1 and 2: if s* is a fixed point with
rho(J) < 1 and m(s*) > 0, then there exists a basin B(s*) such that for
any s_0 in B(s*):

    (a) The dynamics G^t(s_0, c) converge to s* geometrically.
    (b) There exists T_0 such that for all t >= T_0, the readout at
        G^t(s_0, c) is correct.
    (c) The readout at the converged state s* is correct.

The basin radius is:

    r_basin = min(epsilon_stability, m(s*) / K)

where epsilon_stability is the stability radius from Theorem 1.

**Proof.** Part (a) follows from Theorem 1. For part (b), choose T_0
such that C * rho^{T_0} * ||s_0 - s*|| < m(s*)/K. Then for t >= T_0:

    ||G^t(s_0) - s*|| < m(s*)/K

By Theorem 2, m(G^t(s_0)) > 0. Part (c) is immediate since
m(s*) > 0 by assumption.  QED.

---

## Theorem 4: No General Coupling (Wrong Attractors Exist)

**Statement.** There exist dynamics F_theta, readout R, and context c
such that a fixed point s* has F_theta(s*, c) = 0 but m(s*) < 0.

**Construction.** Consider a 1D toy example with V=2, L=1, d=2.
Let target y* = 1 with embedding e_1 = [1, 0]. Let e_0 = [0, 1].

Define G(s) = s (identity -- every point is a fixed point).
Then F(s) = 0 everywhere, and r = 0 everywhere.

But the readout m(s) = cos(s, e_1) - cos(s, e_0), which can be
positive or negative depending on s. For s = e_0, we have
m(s) = cos(e_0, e_1) - cos(e_0, e_0) = 0 - 1 = -1 < 0.

So s = e_0 is a "wrong attractor": converged (r = 0) but incorrect
(m < 0).  QED.

**Interpretation.** This construction is degenerate (identity dynamics)
but proves the point: convergence alone does not guarantee correctness.
The coupling must come from training or structural properties. In
practice, the training loss L = lambda_1 * r^2 + lambda_2 * CE pushes
the model away from wrong attractors on the training distribution.

---

## Theorem 5: Training-Time Coupling

**Statement.** If the training loss

    L(theta) = lambda_1 * ||F_theta(s_T, c)||^2 + lambda_2 * CE(R(s_T), y*)

converges to L < delta on training example (c, y*), then:

    (a) ||F_theta(s_T, c)||^2 < delta / lambda_1
    (b) CE(R(s_T), y*) < delta / lambda_2

and in particular, if delta is small enough:

    (c) The residual r = ||F_theta(s_T, c)|| < sqrt(delta / lambda_1)
    (d) The readout probability p(y* | s_T) >= exp(-delta / lambda_2)
    (e) The decoder margin m(s_T) > 0 when delta < lambda_2 * log(2)
        (since CE < log(2) implies p(y*) > 0.5, which requires
        the correct logit to dominate)

**Proof.** Since L = lambda_1 * A + lambda_2 * B with A, B >= 0
and lambda_1, lambda_2 > 0: L < delta implies A < delta/lambda_1
and B < delta/lambda_2. Parts (a)-(c) follow directly. For (d),
from standard properties of cross-entropy:

    CE(p, y*) = -log p(y*)

so CE < delta/lambda_2 implies p(y*) >= exp(-delta/lambda_2).
This is a WEAKER bound than the naive 1 - delta/lambda_2.

For (e), CE -> 0 requires p(y*) -> 1. Since
p(y*) = exp(sim(y*)/tau) / sum_v exp(sim(v)/tau), this requires
sim(y*)/tau to dominate all other logits. The margin satisfies:

    m = tau * log(p(y*) / max_{v!=y*} p(v))

**CE-to-margin bound (tight).** If CE < eps (where eps = delta/lambda_2),
then p(y*) >= exp(-eps). In the worst case, all remaining probability
concentrates on one wrong class: max_{v!=y*} p(v) = 1 - p(y*).

    m >= tau * log(exp(-eps) / (1 - exp(-eps)))
     = tau * (-eps - log(1 - exp(-eps)))

For small eps: 1 - exp(-eps) ~ eps, so m ~ tau * (-eps - log(eps)).

Numerical examples (tau = 0.1):
- CE < 0.01: m >= 0.1 * (0.01 + 4.60) = 0.461
- CE < 0.1:  m >= 0.1 * (0.1 + 2.35)  = 0.245
- CE < 0.5:  m >= 0.1 * (0.5 + 0.47)  = 0.097
- CE < 0.693: m >= 0 (p(y*) = 0.5 = max wrong)
- CE > 0.693: m not guaranteed positive

So the coupling requires CE < log(2) ~ 0.693 per token (equivalently,
p(y*) > 0.5) for positive margin. In practice, well-trained models
achieve CE << 0.1, giving strong margin guarantees.  QED.

**Scope.** This coupling holds only on the training distribution.
At test time, generalization depends on the smoothness of the
learned dynamics and the coverage of the training distribution.

---

## Proposition 6: Generalization of Coupling

**Statement (informal).** If the dynamics F_theta and readout R are
Lipschitz continuous (which spectral normalization encourages), and the
training distribution has sufficient coverage, then the coupling between
convergence and correctness extends to nearby test inputs.

**Argument.** Let (c_train, y*) be a training example with converged
fixed point s*_train, and let c_test be a test input with
||c_test - c_train|| < epsilon_c.

If G(s, c) is L_c-Lipschitz in c, then the fixed point s*_test (if it
exists near s*_train) satisfies:

    ||s*_test - s*_train|| <= L_c * epsilon_c / (1 - rho)

(from the implicit function theorem applied to G(s*, c) - s* = 0,
assuming rho < 1).

By Theorem 2, the decoder margin at s*_test:

    m(s*_test) >= m(s*_train) - K * ||s*_test - s*_train||
               >= m(s*_train) - K * L_c * epsilon_c / (1 - rho)

So the margin stays positive as long as:

    epsilon_c < m(s*_train) * (1 - rho) / (K * L_c)

**Interpretation.** Tighter contraction (smaller rho), larger margin,
and smoother dynamics (smaller L_c) all increase the generalization
radius. Spectral normalization on the FFN layers helps by bounding
the Lipschitz constant. The warmup schedule for lambda_1 helps by
first learning useful dynamics (large margin) before adding
convergence pressure.

---

## Summary: The Convergence-Correctness Status

| Claim | Status | Evidence |
|-------|--------|----------|
| r -> 0 implies m > 0 in general | FALSE | Theorem 4 (construction) |
| r -> 0 implies m > 0 on training data | TRUE if loss -> 0 | Theorem 5 |
| r -> 0 implies m > 0 on test data | CONDITIONAL | Proposition 6 (depends on Lipschitz, rho, coverage) |
| rho < 1 is necessary for stability | TRUE | Theorem 1 |
| rho < 1 is sufficient for correctness | FALSE | Need margin + coverage too |

**Bottom line.** E5 (self-consistency) is a stopping condition, not a
semantic error function. The convergence-correctness coupling is an
empirical property that depends on:

1. Training loss reaching near-zero (Theorem 5)
2. Spectral radius < 1 at trained fixed points (Theorem 1)
3. Decoder margin > 0 at trained fixed points (Theorem 3)
4. Sufficient training coverage (Proposition 6)
5. Absence of wrong attractors in reachable state space (Theorem 4)

**The right empirical test is D4 (wrong-attractor rate).** If it's
below 5% on held-out data, the coupling holds in practice even
without a theoretical guarantee.

---

## Diagnostic Implications

| Diagnostic | What it measures | Relevant theorem |
|-----------|------------------|-----------------|
| D1 (token accuracy) | Does readout work? | -- |
| D2 (normalized residual) | How close to fixed point? | Theorem 1 |
| D3 (decoder margin) | How confident is readout? | Theorem 2, 3 |
| D4 (wrong-attractor rate) | Are convergence and correctness coupled? | Theorem 4 |
| D5 (basin perturbation) | How large are the basins? | Theorem 3 |
| D6 (spectral radius) | Is the fixed point stable? | Theorem 1 |

The six diagnostics form a complete empirical test of the
convergence-correctness relationship. If D1 is high, D2 is low,
D3 is positive, D4 is low, D5 is high, and D6 is < 1, then the
UESD dynamics are producing correct, stable, robust attractors.
