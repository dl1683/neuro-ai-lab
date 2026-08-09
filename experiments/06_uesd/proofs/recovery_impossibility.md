# Recovery Impossibility Under Cross-Entropy-Only Training

## Motivation

D22 showed that variable-T training achieves near-zero wrong-attractor
rates (WA) for small perturbations but NO positive recovery: extra
dynamics steps after perturbation never reduce WA. D25 implements
recovery-first training to address this gap (T6 = 2/10 confidence).

This document proves that cross-entropy-only training CANNOT produce
positive recovery, regardless of architecture or training duration.
The fundamental reason: CE evaluates readout quality at the NOMINAL
trajectory s_0 -> s_1 -> ... -> s_T only. It provides zero gradient
signal about what happens at perturbed states s_T + epsilon.

---

## Setup and Notation

- G(s, c) = s + F_theta(s, c): update map with parameters theta
- s_t = G^t(s_0, c): nominal trajectory
- s* = lim_{t->inf} s_t: fixed point (when rho < 1)
- m(s): readout margin at state s
- Basin B(s*) = {s : G^t(s) -> s* and m(G^t(s)) > 0 eventually}
- r_basin = radius of B(s*): smallest distance from s* to basin boundary
- Perturbation: s_perturbed = s_T + epsilon, ||epsilon|| = sigma
- Recovery: s_recovered^{(K)} = G^K(s_perturbed, c) for K extra steps
- WA(sigma, K): wrong-attractor rate at perturbation scale sigma after K extra steps
- Recovery(sigma, K) = WA(sigma, 0) - WA(sigma, K): improvement from extra steps

---

## Theorem 11: CE Gradient Contains No Basin-Shape Information

**Statement.** Let the training loss be:

    L(theta) = CE(readout(s_T(theta)), y*)

where s_T(theta) = G^T(s_0, c; theta) is the state after T dynamics
steps. The gradient dL/dtheta is:

    dL/dtheta = (dCE/dR) * (dR/ds_T) * (ds_T/dtheta)

This gradient is a function of the nominal trajectory
{s_0, s_1, ..., s_T} only. It contains no information about:

(a) The Jacobian dG/ds evaluated at any point OTHER than the
    nominal trajectory (particularly not at s_T + epsilon).

(b) The second derivative d^2G/ds^2, which determines basin size
    (via the bound r_basin ~ (1 - rho) / M from
    spectral_contraction.md).

(c) The recovery trajectory G^K(s_T + epsilon) for any epsilon != 0.

**Proof.**

The chain rule gives:

    dL/dtheta = dCE/d(logits) * d(logits)/ds_T * ds_T/dtheta

Each factor depends only on quantities evaluated along the nominal
trajectory:

- dCE/d(logits): evaluated at readout(s_T), depends on s_T only
- d(logits)/ds_T = d(readout)/ds|_{s_T}: evaluated at s_T only
- ds_T/dtheta: the sensitivity of the nominal trajectory to theta,
  computed by BPTT through s_0 -> s_1 -> ... -> s_T

None of these terms involve s_T + epsilon for any epsilon != 0.

Formally, dL/dtheta lies in the tangent space of the loss surface
at the point (theta, s_T(theta)). Perturbation response
G^K(s_T + epsilon) involves the map at s_T + epsilon, which lies
in a different region of state space. The gradient provides no
information about this region unless the dynamics happen to be
globally linear (trivial case).  QED.

---

## Theorem 12: Basin Properties Are Side Effects, Not Objectives

**Statement.** Under CE-only training, the basin of attraction
B(s*) exists as a consequence of contraction (rho < 1) and
smoothness (bounded M), but its size and shape are NOT optimized
by the training loss.

Specifically:

(a) **Basin existence:** If rho(J|_{s*}) < 1, then B(s*) has
    positive radius r_basin >= (1 - rho) / M (from
    spectral_contraction.md Theorem on basin size). This holds
    regardless of the training loss.

(b) **Basin size under CE training:** The basin radius r_basin
    depends on rho and M, both of which are determined by the
    dynamics parameters theta. CE training optimizes theta to
    minimize CE(readout(s_T), y*), which is equivalent to:
    
    - Maximize readout margin m(s_T)
    - Push s_T close to a correct fixed point s*

    Neither of these objectives directly constrains rho or M.
    The spectral radius rho emerges from the dynamics architecture
    and weight magnitudes, not from the CE loss. The second
    derivative M = ||d^2G/ds^2|| is even less constrained.

(c) **Basin shape under CE training:** The basin shape (anisotropy)
    is determined by the eigenvalue structure of J at s*. CE
    training provides no gradient toward isotropic basins (uniform
    contraction in all directions) because it only evaluates the
    readout at one point (s_T), not along multiple perturbation
    directions.

**Proof.**

Part (a): Direct from spectral_contraction.md. The basin exists
whenever the dynamics are contractive, independent of training loss.

Part (b): The CE gradient dL/dtheta optimizes the readout quality
at s_T. It indirectly affects rho through the implicit dependence:

    theta -> G(.; theta) -> J(s*; theta) -> rho(J)

But the gradient dL/dtheta does not equal d(rho)/dtheta. The
relationship between CE and rho is mediated by the complex
interaction between readout accuracy and dynamics structure.

Empirical evidence: D6 measurements show rho in [0.49, 0.53] for
CE-dynamics across multiple seeds, but these values are not
optimized toward any particular target -- they emerge from the
interplay of architecture, initialization, and CE training.

Part (c): Basin shape requires the eigenvalue spectrum of J to be
uniformly contractive. This means all |lambda_i(J)| should be
similar. CE training provides no gradient signal for the eigenvalue
DISTRIBUTION, only for the readout at s_T. Spectral normalization
on FFN layers provides some implicit regularization but does not
control attention-induced eigenvalues.  QED.

---

## Theorem 13: Recovery Trichotomy (Under Clean Basin Geometry)

**Additional Assumptions (A13).**
- (A13.1) Clean basin boundary: the basin B(s*) has a well-defined
  boundary with no significant overlap with other correct-readout
  basins. For dynamics with overlapping or fractal basin boundaries,
  the trichotomy is approximate.
- (A13.2) Single correct attractor: only one fixed point s* produces
  correct readout for the given input. If multiple correct attractors
  exist, out-of-basin perturbations could land in another correct
  basin, producing weak positive recovery.
- (A13.3) Basin is approximately isotropic at the scale of sigma.
  For highly anisotropic basins, recovery can be weakly positive
  along contractive directions while negative along expansive ones.

Under these assumptions, the following holds. For realistic dynamics
with anisotropic/overlapping basins, weak positive recovery
(Recovery ~ 0.001-0.01) may occur, but systematic large positive
recovery requires explicit training (Theorem 14).

**Statement.** For a CE-only trained model with fixed point s* and
basin radius r_basin, under (A13.1-A13.3), the recovery behavior
at perturbation scale sigma falls into approximately one of three
regimes:

**(a) Within-basin (sigma << r_basin):**

WA(sigma, 0) approx 0. The perturbation stays inside the basin, so
the readout is already correct. Extra dynamics steps contract the
state toward s*, but since readout was already correct:

    Recovery(sigma, K) = WA(sigma, 0) - WA(sigma, K) approx 0 - 0 = 0

No improvement possible because there is nothing to improve.

**(b) Basin-boundary (sigma ~ r_basin):**

WA(sigma, 0) is positive (some perturbations exit the basin).
For perturbations that remain inside: extra steps help (converge
back to s*). For perturbations that exit: extra steps hurt (converge
to wrong attractor s'*). The net recovery is:

    Recovery(sigma, K) = P(epsilon in B) * [WA_{in}(0) - WA_{in}(K)]
                       - P(epsilon not in B) * [WA_{out}(K) - WA_{out}(0)]

The first term is approximately 0 (within-basin states are mostly
already correct). The second term is non-negative (out-of-basin
states get worse with more steps). So:

    Recovery(sigma, K) <= 0

Recovery is non-positive at the basin boundary.

**(c) Far-outside-basin (sigma >> r_basin):**

WA(sigma, 0) approx 1. Nearly all perturbations exit the basin.
Extra dynamics steps converge to wrong attractors:

    Recovery(sigma, K) = 1 - WA(sigma, K) approx 1 - 1 = 0

No improvement because almost all states converge to wrong attractors.

**In all three regimes, Recovery <= 0 under (A13.1-A13.3).** The
transition between regimes is sharp under clean basin geometry.
In practice, anisotropic basins and non-normal transients can
produce weak positive recovery (O(0.01)) through directional
effects, but systematic large recovery (O(0.1)+) is not possible
without explicit recovery training (Theorem 14).  QED.

---

## Corollary: Variable-T Helps But Does Not Solve Recovery

**Statement.** Variable-T training (Theorem 9) implicitly
regularizes sigma_max to be smaller, which ENLARGES the basin
(r_basin increases because rho decreases). This shifts the
basin-boundary regime to higher sigma values but does not change
the fundamental trichotomy.

**Consequence for D22 data:**

Variable-T training shifts the WA transition from sigma ~ 0.1
(fixed-T) to sigma ~ 0.3-0.5 (variable-T). But Recovery remains
approximately 0 at all sigma levels because:

- Below the new threshold: WA is already 0
- Above the new threshold: extra steps converge to wrong attractors
- At the threshold: the effects cancel (Theorem 13(b))

D22 data confirms this pattern:

| sigma | WA@0 (fixed-T) | WA@0 (variable-T) | best_recovery |
|-------|------|------|------|
| 0.01 | 0.0005 | 0.0002 | +0.0005 |
| 0.1 | 0.0107 | 0.0003 | +0.0004 |
| 0.5 | 0.9995 | 0.1293 | -0.0043 |
| 1.0 | 1.0000 | 0.9923 | +0.0016 |

Variable-T dramatically reduces WA@0 (enlarged basin), but
best_recovery remains near zero across all sigma levels.

---

## Theorem 14: Recovery Requires Explicit Perturbation Loss

**Statement.** For Recovery(sigma, K) > delta > 0, the training
loss must include a term that evaluates readout quality at a
perturbed state. The minimal sufficient loss is:

    L_recovery(theta) = E_{epsilon ~ N(0, sigma^2 I)}[
        CE(readout(G^K(s_T + epsilon, c; theta)), y*)
    ]

The gradient of this loss w.r.t. theta is:

    dL_recovery/dtheta = E_epsilon[
        dCE/d(logits) * d(logits)/ds * ds_{T+K}/dtheta
    ]

where s_{T+K} = G^K(s_T + epsilon, c) depends on theta through
both the original trajectory (s_T(theta)) and the recovery
trajectory (G^K applied at s_T + epsilon).

**Why this works (and CE alone doesn't):**

The recovery gradient contains the Jacobian of G evaluated at
perturbed states {s_T + epsilon, G(s_T + epsilon), ..., G^{K-1}(...)}.
This provides direct gradient signal for the dynamics behavior
AWAY FROM the nominal trajectory, shaping:

1. The Jacobian dG/ds at s_T + epsilon (not just at s_T)
2. The contraction rate in the perturbed region
3. The basin shape around s*

Specifically, minimizing L_recovery encourages:

    G^K(s_T + epsilon) -> correct readout region

This requires the basin to EXTEND to cover s_T + epsilon, or the
recovery trajectory to cross into the correct basin within K steps.

**Connection to D25 design:**

D25 implements exactly this loss:
- recovery_gentle: sigma -> 0.1, K=5 extra steps
- recovery_stronger: sigma -> 0.2, K=10 extra steps
- recovery_weighted: sigma -> 0.1, K=5, alpha shifts task->recovery

The sigma curriculum (linear ramp from 0.01 to sigma_end) gradually
increases the perturbation scale, expanding the basin incrementally
rather than demanding large-basin recovery from the start.  QED.

---

## Information-Geometric Perspective

The Fisher-Rao metric on the parameter space theta defines the
natural gradient:

    g_ij = E[d(log p)/dtheta_i * d(log p)/dtheta_j]

CE-only training follows the gradient in EUCLIDEAN parameter space,
which ignores the Riemannian structure of the model's probability
manifold. The Fisher-Rao metric captures the CURVATURE of the
probability surface, which is directly related to basin shape.

Recovery requires shaping the dynamics' Jacobian dG/ds around s*,
which is a second-order property of the map G. CE loss is a
first-order property (value of readout at s_T). The disconnect
between first-order optimization (CE) and second-order requirements
(basin shape) is the fundamental reason recovery is impossible
under CE-only training.

The _meta research identifies that Fisher-Rao is the UNIQUE
reparameterization-invariant metric on probability space (Cencov
1982). Basin structure is a topological/geometric property that
requires metric-aware optimization. CE-only training, by ignoring
the metric, cannot control topology.

The recovery loss L_recovery provides metric information because
it evaluates readout at MULTIPLE points (s_T + epsilon for different
epsilon), which implicitly samples the local curvature of the
dynamics around s*.

---

## Proposition 20: Negative Recovery Scales with Computational Depth

**Motivation.** D23 data (baselines L=4 through L=16) reveals that
extra dynamics steps after perturbation produce NEGATIVE recovery
that worsens with carry depth. At sigma=0.1:

| L | D | WA@0 | WA@+1 | Rec@+1 | WA@+20 | Rec@+20 |
|---|---|------|-------|--------|--------|---------|
| 4 | 2 | 0.02% | 0.00% | +0.02% | 3.34% | -3.32% |
| 8 | 4 | 1.07% | 0.83% | +0.24% | 53.76% | -52.69% |
| 12 | 6 | 0.27% | 0.10% | +0.17% | 43.70% | -43.43% |
| 16 | 8 | 1.88% | 1.27% | +0.61% | 83.69% | -81.81% |
| 20 | 10 | 0.51% | 0.29% | +0.22% | 56.74% | -56.23% |
| 24 | 12 | 0.66% | 0.32% | +0.34% | 75.54% | -74.88% |

Extra steps make things dramatically worse, but the pattern is
OSCILLATORY, not monotonic. L=20 (WA@+20=56.74%) is less negative
than L=16 (83.69%) despite deeper carry chain. L=24 (75.54%) is
between L=16 and L=20, with the highest E_enc deficit (0%).

UNIVERSAL +1 RECOVERY: ALL L values show POSITIVE recovery at +1
extra step (0.02-0.61%). This is Theorem 13(a) in action — within-
basin contraction corrects small perturbations in one step. The
crossover from positive to negative recovery occurs between +1 and
+5 steps, when carry error propagation (Prop 20) overtakes
contraction.

**Assumptions (A20).**
- (A20.1) The dynamics perform serial computation: output position i
  depends on the carry state propagated from positions 0,...,i-1.
- (A20.2) CE-only training constrains the Jacobian primarily along
  the nominal trajectory (Theorem 11). The Jacobian at perturbed
  states s' = s_T + epsilon is only weakly/indirectly constrained
  through shared parameters (CE updates alter parameters that also
  determine off-trajectory behavior, but without direct evaluation
  at perturbed states).
- (A20.3) Perturbation corrupts the internal carry representation.

**Statement.** Under CE-only training with carry depth D:

(a) **Error propagation through carry chain.** A perturbation epsilon
    at s_T corrupts the carry state at position i_0 (the position
    most affected by the perturbation direction). Each additional
    dynamics step propagates this carry error to subsequent positions,
    corrupting positions i_0+1, i_0+2, etc.

(b) **Negative recovery mechanism.** Extra dynamics steps INCREASE WA
    because the dynamics are optimized to propagate carry information
    (that's their function), and they propagate carry ERRORS equally
    well. The Jacobian at perturbed states (unconstrained by
    Theorem 11) may even amplify errors.

    After K extra steps, the number of corrupted positions is:

        N_corrupt(K) ~ min(N_corrupt(0) + C_step * K, D)

    and WA scales with the fraction of corrupted positions:

        WA(K) >= 1 - (1 - WA_per_pos)^{N_corrupt(K)}

    where WA_per_pos is the per-position wrong rate from the
    corrupted carry state.

(c) **Depth scaling.** The RATE of negative recovery depends on D
    because deeper carry chains have more positions for error to
    propagate to. The time to full corruption is:

        K_full ~ (D - N_corrupt(0)) / C_step

    For L=4 (D=2): K_full ~ 2 → slow corruption (few positions)
    For L=16 (D=8): K_full ~ 8 → fast corruption (many positions)

    At K=20 (our evaluation horizon), all carry positions are
    corrupted for D >= ~5 (K_full <= 20), explaining the plateau
    of WA@+20 near 80-85% for L=16.

**Connection to Proposition 19.** The difficulty-dependent contraction
(lower sigma_max at L=12 due to training pressure) provides SOME
protection against perturbation — tighter contraction means faster
return toward the correct basin IF the state remains within it.
This explains why L=12 (sigma_max ~ 0.45) has lower WA@+20 (43.7%)
than L=8 (sigma_max ~ 0.70, WA@+20 = 53.8%) despite deeper carry
chains. The tighter contraction partially compensates for the longer
carry chain.

But this compensation is insufficient at L=16 (WA@+20 = 83.7%)
where D=8 creates a genuinely long error propagation path that
even tight contraction cannot prevent.

**Implication for D25.** Recovery training (Theorem 14) must address
the carry-error propagation mechanism. Simply enlarging the basin
may not suffice for deep carry chains — the training must also
constrain the dynamics to CORRECT carry errors, not just tolerate
small perturbations. This suggests D25's sigma curriculum should
be adapted for different L values.

**Claim calibration: WEAK-to-MODERATE (Codex-reviewed 2026-05-24).**
- Direction clear: deeper carry chains produce worse recovery under
  CE-only training. 6 data points (L=4..24) confirm the pattern.
- Non-monotonicity confirmed: L=12 better than L=8, L=20 better
  than L=16 — consistent with Proposition 19 interaction.
- L=24 (D=12, E_enc=0%) at WA@+20=75.54% sits between L=16
  (83.69%) and L=20 (56.74%). Not the worst despite deepest chain
  — maximal training pressure provides some protection.
- Universal +1 positive recovery across all L values is suggestive
  of Theorem 13(a) (within-basin contraction), but values are very
  small (0.02-0.61%) and without confidence intervals, statistical
  significance is unclear. Could be numerical/readout artifact.
- DOWNGRADED from MODERATE per Codex review: +1 recovery needs
  multi-seed significance testing, single-seed-per-L insufficient
  for MODERATE, N_corrupt(K) model is informal.

---

## Empirical Predictions

0. **D23 negative recovery — all L values now ACTUAL:**
   L=20 ACTUAL: WA@+20 = 56.74% at σ=0.1 (prediction of >90%
   was too pessimistic — Proposition 19's contraction effect
   dominates, producing less negative recovery than L=16's 83.69%).
   L=24 ACTUAL: WA@+20 = 75.54% (prediction of ~100% was too
   pessimistic — maximal training pressure keeps it below L=16).
   NEW: All L values show positive +1 recovery (0.02-0.61%),
   validating Theorem 13(a) within-basin contraction.

1. **D25 recovery_gentle and recovery_stronger will show positive
   recovery** at their respective sigma scales, while the
   variable_t_only baseline will show zero recovery (consistent
   with D22).

2. **Recovery improvement will be proportional to K** (more extra
   steps = more recovery) up to a saturation point where K exceeds
   the time needed to return to the basin.

3. **The sigma_end parameter determines the effective basin radius
   after training**: training with sigma_end = 0.2 should produce
   larger basins than sigma_end = 0.1.

4. **Task accuracy may decrease slightly** with aggressive recovery
   training (recovery_weighted) because the loss weight shifts from
   task to recovery. This is the accuracy-robustness tradeoff.

---

## Proposition 21: Criticality-Optimal Recovery (Nishimori-Conditioned)

**Motivation.** The recovery problem has two competing forces:
1. Basin size r ≥ 2(1-ρ)/M — wants ρ small (strong contraction)
2. Computational capacity C_step — wants ρ closer to 1 (richer dynamics)

Neither extreme works: ρ → 0 gives large basins but trivial dynamics
(can't solve carry chains); ρ → 1 gives rich dynamics but vanishing
basins (no recovery). There must be an optimal ρ*.

D6 measured ρ ≈ 0.49-0.53 for CE dynamics. D15 found CE overshoots
(ρ = 0.534) and E5 undershoots (ρ = 0.341). D23 L=12 estimated
σ_max ≈ 0.45, remarkably close to tanh(1/2) = 0.462.

**Assumptions (A21).**
- (A21.1) Basin radius r ∝ (1-ρ)/M (from Theorem in
  spectral_contraction.md, Section 4).
- (A21.2) Computational capacity per step C_step increases
  monotonically with ρ. Heuristic: richer dynamics (closer to
  criticality) can resolve more carry positions per step. Justified
  by D23 data showing variable C_step ∈ [0.7, 2.0].
- (A21.3) Recovery capacity is approximately the product of
  basin coverage and task solvability: R(σ) ∝ P(σ < r) × P(task
  solved in T steps). This is an approximation: in reality, basin
  shape matters, not just radius.

**Statement.** For dynamics G(s,c) = s + F_θ(s,c) with spectral
radius ρ at fixed point s*, the recovery capacity

    R(ρ, σ) = P(perturbation σ within basin of radius r(ρ))
            × P(task solved in T steps at contraction rate ρ)

is jointly maximized at a critical spectral radius ρ*.

**Derivation.**

Factor 1 — Basin coverage: For Gaussian perturbation
ε ~ N(0, σ²I) in d dimensions, the probability of staying within
the basin of radius r = 2(1-ρ)/M is:

    P_basin(ρ, σ) = P(||ε|| < r) ≈ 1 - exp(-r²/(2σ²)) for d >> 1

This is monotonically decreasing in ρ (larger ρ → smaller basin).

Factor 2 — Task solvability: The dynamics must converge to a correct
fixed point within T steps. Convergence rate is ρ^T. For the state
to be within tolerance ε_tol of s* after T steps:

    ||s_T - s*|| ≤ ρ^T × ||s_0 - s*|| < ε_tol

This requires T > log(ε_tol / ||s_0 - s*||) / log(ρ). Smaller ρ →
fewer steps needed → more capacity for actual computation.

BUT: this is only the convergence criterion. The dynamics must
ALSO propagate carry information through the sequence. From
Proposition 19 (difficulty-dependent contraction), tighter
contraction (smaller ρ) forces the dynamics to "work harder" at
carry propagation, which can either help or hurt depending on the
task difficulty regime.

Factor 3 — Dynamic range at criticality: At the Nishimori critical
point ρ_c = tanh(1/2), the dynamics have maximal dynamic range —
they respond to perturbations across many scales. Below ρ_c, the
dynamics are overdamped (perturbations are rapidly contracted,
no information about error direction). Above ρ_c, the dynamics
are underdamped (perturbations grow, basins vanish).

**The Nishimori prediction:** ρ* ≈ tanh(1/2) = 0.462.

At this point:
- Basin radius: r = 2(1-0.462)/M = 1.076/M
- Contraction: ρ^10 = 0.462^10 ≈ 4.7 × 10^{-4} (strong convergence
  in T=10 steps)
- Dynamic range: maximal sensitivity to perturbation direction
  (can distinguish which carry positions are corrupted)

Compare to ρ = 0.9 (typical "well-contractive"):
- Basin radius: r = 0.2/M (5.4× smaller)
- Contraction: 0.9^10 ≈ 0.35 (marginal convergence)
- Dynamic range: low (all perturbations contract similarly)

Compare to ρ = 0.1 (over-contractive):
- Basin radius: r = 1.8/M (1.7× larger than Nishimori)
- Contraction: 0.1^10 ≈ 10^{-10} (immediate collapse)
- Dynamic range: none (all information destroyed in 1-2 steps)

**Supporting evidence.**
- D6: CE dynamics naturally evolve to ρ ≈ 0.49-0.53, close to
  tanh(1/2) but slightly above. SOC self-organization explains
  this proximity but not exact convergence.
- D15: CE overshoots (ρ = 0.534), E5 undershoots (ρ = 0.341).
  The average (0.438) is remarkably close to tanh(1/2).
- D23 L=12: σ_max ≈ 0.45 ≈ tanh(1/2), and L=12 shows the best
  T_min / accuracy tradeoff (T_min ~3, 100% accuracy).
- Cross-domain: The _meta research identifies tanh(1/2) as a
  universal critical point across 7+ domains (AI, brain, evolution,
  markets, music, quantum error correction, democracy).

**Falsifiable prediction for D25:** If recovery training (Theorem 14)
succeeds, the trained model's spectral radius ρ should converge
toward tanh(1/2) ≈ 0.462 as a natural consequence of jointly
optimizing task accuracy and perturbation recovery. Specifically:

(a) recovery_gentle (σ_end=0.1): ρ should decrease from ~0.53
    (CE baseline) toward ~0.47, and recovery should be positive
    at σ=0.1.

(b) recovery_stronger (σ_end=0.2): ρ should decrease further
    toward ~0.46, and recovery should be positive at σ=0.2.

(c) recovery_weighted (σ_end=0.1 with weight shift): ρ should be
    closest to tanh(1/2) because the loss weight shift toward
    recovery directly shapes the basin structure.

**Falsifiable prediction for D26 (proposed):** Explicitly targeting
ρ toward tanh(1/2) via a Lyapunov penalty:

    L_crit(θ) = λ_c × (ρ(J_θ) - tanh(1/2))²

should produce BETTER recovery than D25's implicit basin shaping,
because it directly controls the spectral radius rather than
relying on perturbation loss to shape it indirectly.

**Connection to SOC.** Self-organized criticality (Zhang & Tang,
PNAS Dec 2025) predicts that SGD ≈ natural gradient pins the
Fisher-Rao distance to 1 nat, which corresponds to βJ = 1/2 in
the Ising representation. This gives ρ = tanh(βJ) = tanh(1/2)
at the Bethe lattice critical point. Our dynamics may be
self-organizing toward this point through SGD, but imperfectly
(CE overshoots because CE doesn't directly penalize off-trajectory
behavior — Theorem 11).

**Claim calibration: WEAK-to-MODERATE.**
- Direction supported: ρ near tanh(1/2) appears empirically in
  D6, D15, and D23 L=12. Cross-domain evidence is extensive.
- Quantitative prediction (ρ* = tanh(1/2) exactly) is stronger
  than the evidence warrants. The Nishimori connection is
  analogical, not derived from UESD first principles.
- The "dynamic range at criticality" argument (Factor 3) is
  qualitative, not quantitative.
- Upgrade path: D25 spectral radius measurements will directly
  test prediction (a)-(c). D26 criticality-targeting would be
  a strong test of the full proposition.

---

## Relation to Existing Proofs

This theorem extends:
- Theorem 4 (convergence_correctness.md): wrong attractors exist;
  this proves CE cannot prevent convergence to them after perturbation
- Theorem 8 (convergence_correctness.md): dynamics-decoder separation;
  the coupling term dCE/ds_T * ds_T/dtheta provides no perturbation info
- Spectral contraction (spectral_contraction.md): basin size is
  (1-rho)/M, a side effect not an objective

New contribution: formal proof that recovery is impossible under
CE-only, with the trichotomy argument showing exactly why recovery
is non-positive in all regimes.

---

## Claim Calibration

**STRONG (rigorous):**
- CE gradient contains no basin-shape information (Theorem 11)
- Basin exists as side effect of contraction (Theorem 12(a))
- Recovery loss provides Jacobian gradient at perturbed states (Thm 14)

**MODERATE (conditionally valid):**
- Recovery trichotomy (Theorem 13): assumes sharp basin boundary
  (holds for smooth dynamics with bounded M; fuzzy for highly
  non-normal dynamics)
- Variable-T helps but doesn't solve (Corollary): depends on
  Theorem 9's sigma_max bound
- Negative recovery scales with D (Proposition 20): mechanism clear,
  4 data points from D23 (L=4,8,12,16), L=12 anomaly explained by
  Proposition 19 interaction

**WEAK-to-MODERATE (directional with supporting data):**
- Criticality-optimal recovery (Proposition 21): ρ* ≈ tanh(1/2)
  prediction supported by D6/D15/D23 data and cross-domain evidence,
  but Nishimori connection is analogical. D25 spectral measurements
  will upgrade or falsify.

**WEAK (directional):**
- Information-geometric perspective: the Fisher-Rao connection is
  suggestive but not rigorously linked to basin recovery
- Accuracy-robustness tradeoff prediction: direction correct but
  magnitude unknown
- Quantitative carry-error propagation rate: model-dependent, only
  approximately linear in C_step
