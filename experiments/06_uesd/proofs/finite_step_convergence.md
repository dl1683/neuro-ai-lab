# Finite-Step Dynamics: Convergence After T Iterations

## Motivation

The UESD proofs in convergence_correctness.md mostly assume behavior
at a true fixed point (r = ||F(s*)|| = 0). But experiments use a fixed
T = 10 iterations, so s_T is NOT generally a fixed point. This document
derives error bounds for finite-T dynamics.

---

## Setup

- G(s, c) = s + F_theta(s, c) is the update map
- s_0 is the initial state (learned positional embeddings)
- s_t = G(s_{t-1}, c) for t = 1, ..., T
- s* is the nearest fixed point: G(s*, c) = s*
- J = dG/ds|_{s*} with spectral radius rho < 1

---

## Theorem 1: Finite-T Distance to Fixed Point

**Statement.** If G is L_G-Lipschitz in a neighborhood of s* containing
the trajectory {s_0, s_1, ..., s_T}, and rho(J) < 1, then for the
induced operator norm ||J|| = rho + epsilon (where epsilon depends on
the non-normality of J):

    ||s_T - s*|| <= ||J||^T * ||s_0 - s*|| + O(||s_0 - s*||^2)

More precisely, using the linearization:

    s_t - s* = G(s_{t-1}, c) - G(s*, c)
             = J * (s_{t-1} - s*) + R(s_{t-1}, s*)

where ||R(s, s*)|| <= M * ||s - s*||^2 for second-derivative bound M.

By induction:

    ||s_T - s*|| <= (rho + epsilon)^T * ||s_0 - s*||
                  + M * ||s_0 - s*||^2 * sum_{k=0}^{T-1} (rho + epsilon)^k

              = (rho + epsilon)^T * ||s_0 - s*||
                  + M * ||s_0 - s*||^2 * (1 - (rho + epsilon)^T) / (1 - rho - epsilon)

For T = 10 and rho = 0.95:

    Linear term: 0.95^10 * ||s_0 - s*|| ~ 0.60 * ||s_0 - s*||
    (still 60% of initial distance -- slow convergence!)

For rho = 0.8:

    Linear term: 0.8^10 * ||s_0 - s*|| ~ 0.11 * ||s_0 - s*||
    (89% reduction -- good)

For rho = 0.5:

    Linear term: 0.5^10 * ||s_0 - s*|| ~ 0.001 * ||s_0 - s*||
    (99.9% reduction -- essentially converged)

**Implication:** T = 10 is sufficient only if rho is well below 1.
If D6 measures rho > 0.9, the state may still be far from the fixed
point after T = 10 steps.

---

## Theorem 2: Readout Error After T Steps

**Statement.** If the readout function R is K-Lipschitz (as defined in
convergence_correctness.md Theorem 2), then the decoder margin at s_T
satisfies:

    m(s_T) >= m(s*) - K * ||s_T - s*||

Using Theorem 1:

    m(s_T) >= m(s*) - K * (rho + epsilon)^T * ||s_0 - s*||
            - K * M * ||s_0 - s*||^2 * (1 - (rho + epsilon)^T) / (1 - rho - epsilon)

For the readout to be correct at step T, we need m(s_T) > 0:

    (rho + epsilon)^T < m(s*) / (K * ||s_0 - s*||)  (ignoring quadratic term)

**Minimum T for correct readout (linear approximation):**

    T >= log(K * ||s_0 - s*|| / m(s*)) / log(1 / (rho + epsilon))

For concrete values: K ~ 20 (from convergence_correctness.md estimate),
||s_0 - s*|| ~ 10 (typical initialization distance), m(s*) = 0.5
(moderate margin), rho = 0.9:

    T >= log(20 * 10 / 0.5) / log(1/0.9) = log(400) / 0.105 ~ 57

This suggests T = 10 may be insufficient for rho = 0.9!

For rho = 0.5:

    T >= log(400) / log(2) ~ 8.6

So T = 10 suffices if rho ~ 0.5. The D6 measurement directly informs
whether T = 10 is enough.

---

## Theorem 3: Residual at Finite T

**Statement.** The residual r_T = ||F(s_T, c)|| at step T satisfies:

    r_T = ||G(s_T, c) - s_T|| = ||s_{T+1} - s_T||

Using the linearization near s*:

    s_{T+1} - s_T = G(s_T, c) - s_T
                   = (G(s_T, c) - s*) - (s_T - s*)
                   = J(s_T - s*) - (s_T - s*) + O(||s_T - s*||^2)
                   = (J - I)(s_T - s*) + O(||s_T - s*||^2)

So:

    r_T ~ ||(J - I)|| * ||s_T - s*||

Since J = dG/ds and dF/ds = J - I:

    r_T ~ ||dF/ds|| * ||s_T - s*||

If dF/ds has operator norm ||dF/ds|| ~ delta (the "speed" of the
dynamics near the fixed point), then:

    r_T ~ delta * (rho + epsilon)^T * ||s_0 - s*||

This means the residual r_T is proportional to the distance from the
fixed point. Low residual genuinely indicates proximity to the fixed
point (up to the ||dF/ds|| scaling factor).

**This partially validates D2 as a convergence measure:** low r_T implies
small ||s_T - s*||, which by Theorem 2 implies correct readout (if
m(s*) > 0 and K * ||s_T - s*|| < m(s*)).

---

## Practical Implications

1. **T = 10 is sufficient if rho < 0.8.** The state will be within 11%
   of the fixed point. For rho > 0.9, consider increasing T or checking
   that the residual is indeed small.

2. **D2 (normalized residual) validates convergence at finite T.** If
   r_T / sqrt(L*d) < 0.01, the state is close to a fixed point. Combined
   with D3 (margin > 0) and D4 (wrong-attractor rate < 5%), this gives
   strong empirical evidence for correct convergence.

3. **Copy task should converge fast.** Copy is near-identity, so rho
   should be small and T = 10 is more than enough. Reversal requires
   more computation and may have higher rho.

4. **Adaptive T could help.** If D2 > threshold at T = 10, running more
   steps could improve accuracy. This is deferred to future work but the
   current fixed-T design is appropriate for the POC.

---

## Theorem 4: Non-Normal Finite-T Stability (σ_max Bound)

**Motivation.** Theorems 1-3 use ρ(J) + ε as the convergence rate,
where ε accounts for non-normality but is left unquantified. For
non-normal Jacobians (J^H J ≠ J J^H), the relevant quantity is
σ_max(J) = ||J||_2 (the largest singular value), not ρ(J). This
theorem provides the rigorous finite-T bound.

**Setup.** Let J = dG/ds|_{s*} with:
- σ_max = σ_max(J) (largest singular value)
- ρ = ρ(J) (spectral radius)
- κ = σ_max / ρ (non-normality ratio; κ = 1 for normal J)
- M = sup_{s ∈ B(s*,r)} ||d²G/ds²|| (second-derivative bound)
- K = Lipschitz constant of the margin function m

**Statement.** For s_0 in a neighborhood of s* where ||s_0 - s*|| < r
with r ≤ (1 - σ_max) / M when σ_max < 1:

(a) **One-step bound (exact, no approximation):**

    ||s_{t+1} - s*|| ≤ σ_max · ||s_t - s*|| + (M/2) · ||s_t - s*||²

Proof: s_{t+1} - s* = G(s_t, c) - G(s*, c) = J(s_t - s*) + R(s_t, s*)
where ||R(s_t, s*)|| ≤ (M/2)||s_t - s*||² by Taylor remainder. Then
||J(s_t - s*)|| ≤ ||J|| · ||s_t - s*|| = σ_max · ||s_t - s*||. □

(b) **T-step bound:** If σ_max < 1 and ||s_0 - s*|| < 2(1 - σ_max)/M:

    ||s_T - s*|| ≤ σ_max^T · ||s_0 - s*||
                    + (M/2) · ||s_0 - s*||² · σ_max^{T-1} · (1 - σ_max^T) / (1 - σ_max)

The leading term σ_max^T · ||s_0 - s*|| dominates when the quadratic
remainder is small. This simplifies to:

    ||s_T - s*|| ≤ σ_max^T · ||s_0 - s*|| · (1 + O(||s_0 - s*||))

(c) **Readout preservation under non-normality:** Readout is correct
at step T if:

    m(s*) > K · σ_max^T · ||s_0 - s*|| · (1 + (M · ||s_0 - s*||) / (2(1 - σ_max)))

Compare to the normal case (Theorem 2) where σ_max = ρ.

(d) **When σ_max ≥ 1 but ρ < 1 (transient growth regime):**

σ_max^T grows initially. Asymptotic contraction requires ρ < 1, but
finite-T behavior is governed by σ_max:

    ||s_T - s*|| ≤ σ_max^T · ||s_0 - s*||    (leading order)

If σ_max = 1.05 and T = 10: amplification factor 1.63 (63% GROWTH).
Correct readout requires m(s*) > 1.63 · K · ||s_0 - s*||.

The crossover step T* where asymptotic contraction dominates:

    T* ≈ log(κ) / log(σ_max / ρ)

For T < T*, the dynamics amplify; for T > T*, they contract.

**Minimum T for correct readout (non-normal case):**

    T ≥ log(K · ||s_0 - s*|| / m(s*)) / log(1 / σ_max)

Compare to Theorem 2: the denominator uses log(1/σ_max) instead of
log(1/ρ). When κ > 1, σ_max > ρ, so log(1/σ_max) < log(1/ρ), and
MORE steps are required.

For concrete values: σ_max = 1.0, ρ = 0.9, κ = 1.11:
- Theorem 2 (spectral): T ≥ log(400)/log(1/0.9) ≈ 57
- Theorem 4 (singular): T = ∞ (σ_max = 1 never contracts!)

This illustrates why the σ_max bound is strictly more conservative
and correct: if σ_max ≥ 1, no amount of iteration guarantees
contraction, even though ρ < 1 promises eventual convergence.

---

## Corollary: Non-Normality Ratio as Diagnostic (D7)

**Definition.** The non-normality ratio is κ = σ_max(J) / ρ(J).

**Interpretation:**
- κ = 1.0: J is normal. σ_max = ρ. Theorems 1-3 are exact.
- κ ∈ (1, 1.5): Mild non-normality. Finite-T bounds are within
  50% of the spectral prediction. Practical for T = 10.
- κ ∈ (1.5, 2): Moderate non-normality. The effective convergence
  rate is significantly slower than ρ^T. May need T > 10.
- κ > 2: Severe non-normality. Transient growth dominates for
  practical T values. D5 basin perturbation is the reliable test.

**Measurement.** σ_max(J) = sqrt(ρ(J^H J)) can be computed via
power iteration on J^H J (requiring adjoint-vector products). In
practice, use D5 (basin perturbation) as the empirical proxy: if
perturbed states return to the same readout, σ_max^T is effectively
bounded by the basin radius.

**Relation to D5.** D5 stability at perturbation scale σ implies:

    σ_max^T · σ · ||s_T|| < basin radius ≈ m(s*) / K

So high D5 stability empirically upper-bounds σ_max^T even without
computing it directly.

---

## Caveat: Kreiss Constant and Worst-Case Transient Growth

The Kreiss constant K(J) = sup_{|z|>1} (|z|-1) · ||(zI-J)^{-1}||
bounds worst-case transient growth:

    sup_{k≥0} ||J^k|| / ρ(J)^k ≤ e · n · K(J)    (Kreiss theorem)

where n = dim(J). For normal matrices K(J) = 1 (no transient
growth). For highly non-normal J, K(J) can be large.

Self-attention creates non-normal Jacobians (attention weights break
symmetry). The practical test is D5 (basin perturbation): if perturbed
states return to the same readout, the dynamics are empirically stable
regardless of the Kreiss constant.

**Key insight:** The σ_max bound (Theorem 4) is tighter than the
Kreiss bound for T-step dynamics because it directly uses the
one-step amplification factor rather than the worst-case over all k.
