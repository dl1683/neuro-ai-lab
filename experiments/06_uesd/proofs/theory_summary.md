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

### 2.11 Variable-T Spectral Regularization (variable_t_spectral_stability.md)

**Result (Theorem 9):** Training with T sampled from {T_1,...,T_k}
implicitly regularizes sigma_max:

    sigma_max(J) < (m(s*) / (K * d_0))^{1/T_1}

This is strictly tighter than the fixed-T bound at T_k by a factor
of (m/(Kd_0))^{(T_k-T_1)/(T_1*T_k)}. The compute window is
guaranteed to span at least [T_1, infinity) when rho < 1.

**Result (Theorem 10):** Variable-T training suppresses oscillatory
eigenvalues (large imaginary parts) because readout must be correct
at multiple non-commensurate T values, forcing eigenvalues toward
the real axis.

[Proof: variable_t_spectral_stability.md, Theorems 9-10]

### 2.12 Recovery Impossibility Under CE-Only (recovery_impossibility.md)

**Result (Theorem 11):** The CE gradient dL/dtheta contains no
information about basin shape — it evaluates readout ONLY along
the nominal trajectory, never at perturbed states s_T + epsilon.

**Result (Theorem 13, Recovery Trichotomy):** For CE-only trained
models, recovery is non-positive in all regimes:
- Within-basin: WA already 0, no improvement possible
- Basin-boundary: within-basin improvement cancels out-of-basin
  worsening
- Far-outside: converges to wrong attractors, extra steps hurt

**Result (Theorem 14):** Positive recovery requires an explicit
perturbation loss that evaluates readout at G^K(s_T + epsilon),
providing gradient signal for the Jacobian at perturbed states.

[Proof: recovery_impossibility.md, Theorems 11-14]

### 2.13 Bottleneck-Depth Scaling Law (bottleneck_depth_scaling.md)

**Result (Theorem 15):** T_readout = O(log V / log(1/sigma_max)).
The readout precision requirement grows logarithmically with V.

**Result (Theorem 16):** T_depth >= D / C_step where D is
computational depth and C_step is per-step capacity. For serial
carry in small models: T_depth ~ D.

**Result (Theorem 17):** T_min = max(T_readout, T_depth). This
decomposes the minimum step count into independent readout and
depth components, predicting a phase transition in the D23 carry-
depth experiment at L ~ 16-20 where the system transitions from
readout-limited to depth-limited.

[Proof: bottleneck_depth_scaling.md, Theorems 15-18]

### 2.14 Difficulty-Dependent Contraction (bottleneck_depth_scaling.md)

**Result (Proposition 19):** T_min is non-monotonic in carry depth D
because harder tasks force stronger training gradients on the dynamics,
driving sigma_max lower. At moderate D, the improved contraction rate
more than compensates for the increased computational depth, producing
a local MINIMUM in T_min.

D23 data confirms: L=12 (D=6) has T_min≈3, while L=8 (D=4) has
T_min≈5, despite L=12 having 50% deeper carry chain. The encoder-
only baseline drops from 89.7% at L=8 to 70.3% at L=12, creating
stronger gradient pressure on the dynamics.

Cross-domain connections: dissipative adaptation (England 2013),
dynamic range at criticality (Kinouchi & Copelli 2006), non-monotonic
dose-response in latent reasoning (_meta repo).

[Proof: bottleneck_depth_scaling.md, Proposition 19]

### 2.15 Criticality-Optimal Recovery (recovery_impossibility.md)

**Result (Proposition 21):** Recovery capacity R(ρ, σ) — the joint
probability that a perturbation remains in-basin AND the task is
solved — is maximized at a critical spectral radius ρ* ≈ tanh(1/2)
= 0.462.

Three competing forces determine ρ*:
1. Basin size r ∝ (1-ρ)/M — wants ρ small
2. Computational capacity C_step — wants ρ closer to 1
3. Dynamic range at criticality — maximized at Nishimori point

Supporting data: D6 measured ρ ≈ 0.49-0.53, D15 found CE/E5
average ≈ 0.438, D23 L=12 σ_max ≈ 0.45. All cluster near tanh(1/2).

Falsifiable D25 prediction: recovery training should push ρ toward
tanh(1/2). Proposed D26: explicit criticality targeting via Lyapunov
penalty L_crit = λ_c(ρ - tanh(1/2))² should outperform implicit
basin shaping.

[Proof: recovery_impossibility.md, Proposition 21]

### 2.16 Signal Amplification via Channel-Coding Dynamics (bottleneck_depth_scaling.md)

**Result (Proposition 22):** UESD dynamics implement an iterative
decoding process analogous to turbo/LDPC belief propagation. The
encoder acts as a noisy channel with per-token MI I_enc, and the
dynamics amplify this signal by exploiting inter-position constraints
(carry chain structure).

D23 L=24 demonstrates the extreme case: E_enc_seq = 0% but
dynamics reach 99.61% accuracy, amplifying ~0.9 bits/token encoder
signal to 6 bits/token output (6.7× amplification).

The Nishimori condition (ρ = tanh(1/2)) corresponds to the optimal
decoding threshold where information extraction per iteration is
maximized at I_step = ln(cosh(1/2)) ≈ 0.173 bits.

Minimum iterations lower bound: T_min ≥ [H(y*) - C_enc] / I_step.

[Proof: bottleneck_depth_scaling.md, Proposition 22]

**Corollary 22.1: Depth-Dependent Re-Read Necessity (REVISED).**
As computational depth D increases, the PROBABILITY that the optimizer
finds a re-read-dependent strategy increases, but the transition is
stochastic, not deterministic. The original sigmoid fit (D*=4.95)
is DEFUNCT — L=12 seed=1337 achieves 89% without re-reading.

D27 3-seed evidence: L=8 delta = 15.6% ± 6.7% SE (3 seeds, all B),
L=12 delta = 40.3% ± 23.8% SE (3 seeds: 86.0%, 11.0%, 23.8%).
Bimodal split confirmed: Strategy A (86.0%) vs Strategy B (11-24%),
gap 62pp >> within-B variance 13pp. P_A(D=4)=0/3, P_A(D=6)=1/3.
Strategy B correlates with better noise robustness.

Status: PARTIALLY CONFIRMED (direction real, bifurcation CONFIRMED).
Quantitative P_A estimates need n>=8 seeds.

[Proof: bottleneck_depth_scaling.md, Corollary 22.1]

### 2.17 Spin-Glass Decoding Isomorphism (bottleneck_depth_scaling.md)

**Result (Proposition 23):** Via the Sourlas construction (Nature 1989),
UESD energy minimization E(s) = ||F_theta(s,c)||^2 maps to ground-state
search in a spin glass, where:
- Codeword (correct answer) = spin-glass ground state
- Encoder noise = random bond perturbation
- Dynamics iterations = simulated annealing / gradient descent toward ground state
- Channel capacity = critical temperature T_c of the spin glass

This is a formal isomorphism, not analogy. Three consequences:
(a) The decoding threshold (minimum encoder MI for dynamics to succeed)
    equals the spin-glass critical temperature: beta_c = 1/(2T_c)
(b) At the Nishimori temperature (beta = beta_N = 1/2), the decoder's
    prior matches the true channel noise — Bayes-optimal decoding
(c) The step-accuracy "waterfall" in D23 maps to the spin-glass
    phase transition: sharp jump from disordered (wrong) to ordered
    (correct) near T_c

Additional evidence from Usatyuk et al. (2025): the smallest
Bethe-Hessian eigenvalue vanishes at the Nishimori temperature,
providing a computable criterion for optimal decoding.

L=24 waterfall: T=1:0%, T=3:64%, T=5:99% matches the LDPC/turbo
waterfall profile, with steepness increasing with L (coding theory
prediction: longer codeword = sharper waterfall).

[Proof: bottleneck_depth_scaling.md, Proposition 23]
[Calibration: WEAK — formal isomorphism via Sourlas, but mapping
 to UESD's specific architecture requires additional steps:
 identification of the factor graph, computation of code rate,
 and verification that the dynamics' convergence profile matches
 BP's iteration-vs-BER curve]

### 2.18 T_min Saturation via Parallel Attention Dynamics (bottleneck_depth_scaling.md)

**Result (Proposition 24):** For UESD dynamics with unrestricted
self-attention, T_99 (first T achieving ≥99% seq_acc) is empirically
constant at ≈5 across carry depths D∈{2,4,6,8,10,12}. All 6 D values
reach >99% by T=5.

The mechanism hypothesis: full self-attention allows O(1) information
ACCESS between positions, enabling the learned dynamics to resolve carry
chains without sequential propagation. C_step increases with D (observed
{0.67,0.80,1.2,1.6,2.0,2.4}) but the functional form (proportional or
otherwise) is not determined from 6 points.

CAVEAT: "O(1) access ⇒ O(1) computation" is a capacity hypothesis,
not derived from first principles (Codex review, 2026-05-24). The T=5
constant could also reflect T=10 training bias (T_99 ≈ T_train/2).

Implications:
(a) T_min dominated by T_readout ≈ 5, not carry depth D
(b) Weakens SPARSE LOCAL BP interpretation — consistent with dense
    factor graphs or learned global solvers
(c) Consistent with attractor/dynamical-systems interpretation

Key falsifiable predictions (Codex-augmented):
1. Local attention (window w) → T_min should scale with D/w (strongest)
4. Out-of-range D>12 test: T_99 constant or increases?
5. Fixed-T training ablation: T_99 tracks T_train/2 or stays constant?

[Proof: bottleneck_depth_scaling.md, Proposition 24]
[Codex review: proofs/_codex_prop24_review.md]
[Calibration: MODERATE — 6 data points across D=2-12, theoretical
 mechanism plausible but not rigorous, single seed/architecture/task,
 T=10 training bias not ruled out]

### 2.19 Banach Contraction Convergence Rate (bottleneck_depth_scaling.md)

**Result (Proposition 25, REJECTED):** Predicted k ≈ 0.4 contraction
ratio from T_99=5 universality. D28 measured k = 0.988 (FLAT), NOT 0.4.
rho > 1 across all 6 tested models (range 1.002-1.051). There is NO
Banach contraction in full state space.

The original argument (k < 0.01^{1/5} ≈ 0.398) assumed convergence to
a fixed point in Frobenius norm. D28 showed this is wrong: the dynamics
converge in READOUT SPACE (T_99 = 2-4 steps) but NOT in full state space.
The mechanism is manifold reachability (Prop 28), not point contraction.

**Corollary 25.1: Distance-Dependent Contraction (REJECTED).** Predicted
U-shaped k_t profile. D28 measured FLAT k_t = 0.988 at ALL steps. No
strong-contraction zone exists. The non-monotonic D25 recovery (+10 peak,
+20 degradation) must be explained by a different mechanism than U-shaped
contraction — likely manifold topology (state overshoots the readout-
correct region of M at +20 steps).

[Proof: bottleneck_depth_scaling.md, Proposition 25 + Corollary 25.1]
[Status: REJECTED — D28 L=4 k=0.988 FLAT, rho>1 universal (6/6 models).
 Pending: do k and rho change qualitatively at L=12+?]

**Corollary 25.2: Two-Step Acceleration via Contraction Halving.**
Variable-T training universally reduces T_99 by exactly 2 steps (CONFIRMED
at L=4,8,12 in D23). The acceleration is robust because it lives in a wide
stability region: k_eff(vt)/k_eff(base) ∈ [0.35, 0.55] all map to the same
2-step ceil-function jump. Quantitative prediction for D28: k_eff(fixed_t) ≈ 0.4,
k_eff(variable_t) ≈ 0.22, ratio ≈ 0.54.

[Proof: bottleneck_depth_scaling.md, Corollary 25.2]
[Calibration: PRE-EMPIRICAL (D23 T_99 data; D28 will directly measure k_eff)]

### 2.20 Canalization of Attractor Basins (bottleneck_depth_scaling.md)

**Result (Proposition 26, partially confirmed):** Variable-T training
acts as Waddington canalization on the UESD dynamics landscape. By
requiring correct readout at EVERY T in {4,...,16}, the dynamics must
converge fast AND stay converged, which geometrically deepens and widens
the basin of attraction.

Variable-T provides implicit basin-shape gradient (evaluating readout at
multiple states along the trajectory), functioning as a soft form of
Theorem 14's perturbation loss. This explains D25's +27.5% implicit
recovery WITHOUT explicit recovery training: the widened basin catches
σ=0.2 perturbations that would escape the narrower fixed-T basin.

D22 denoising failure (σ=0.3) matches the canalization prediction:
gentle, structured perturbation (temporal: variable T) succeeds where
aggressive, random perturbation (spatial: noise injection) fails.

Coherence gap prediction (Rozum et al. 2025, untested): variable-T models
should be robust to noise but equally sensitive to structural perturbations.

[Proof: bottleneck_depth_scaling.md, Proposition 26]
[Calibration: MODERATE (partially confirmed) — D22 breakthrough + D25
 +27.5% recovery + D22 denoising failure all consistent with canalization.
 Coherence gap prediction untested. Single task/architecture/1-2 seeds.]

### 2.21 Directed Amplification via Structured Jacobian Rotation (bottleneck_depth_scaling.md)

**Result (Proposition 27, preliminary):** D6 random-matrix null model
reveals that trained dynamics amplify MORE than matched-spectrum random
rotation (factor alpha = 2.0-2.6), not less. The 718x conservatism from
D3b is entirely explained by spectrum shape (most SVs << sigma_max).

The inter-step singular vector alignment exhibits a temporal profile:
- Early steps (1-5): a_t near 0 (orthogonal rotation between subspaces)
- Late steps (6-10): a_t near 1 (coherent contraction)

This reveals a two-phase computation strategy:
1. Exploration: diverse transformation through orthogonal subspaces
2. Convergence: consistent contraction toward fixed point

The transition at step ~5 coincides with T_min from D19/D23, confirming
that Phase 1 = computation and Phase 2 = refinement. Connects to D4
(CE uses rotation, E5 uses compression) and D5 (CE lower alignment).

E5 prediction: alpha closer to 1.0, uniformly high alignment (pure
convergence, less exploration needed due to per-step contraction).

[Proof: bottleneck_depth_scaling.md, Proposition 27]
[Calibration: MODERATE (preliminary) — 3/8 CE-dynamics samples, alpha>1
 consistent across all. E5 track pending. Single seed/task/architecture.]

**Corollary 27.1: Alignment-Contraction Duality.** The alignment profile
(Prop 27) anti-correlates with the contraction profile (Corollary 25.1).
Both arise from linearization near s*: far from s*, dynamics are nonlinear
(diverse Jacobians, strong contraction, low alignment); near s*, dynamics
linearize (stable Jacobian, weak contraction, high alignment). The transition
step t* should be the same for both profiles. Cross-prediction: D28 k_t
should anti-correlate with D6 a_t (Pearson r < -0.7).

[Proof: bottleneck_depth_scaling.md, Corollary 27.1]
[Calibration: PRE-EMPIRICAL — architectural reasoning, testable with D28]

### 2.22 Readout-Stable Manifold (bottleneck_depth_scaling.md)

**Result (Proposition 28, EMPIRICALLY CONFIRMED at L=4):** D28 full data
reveals the dynamics do NOT converge to a fixed point. Instead:
- k_frob = 0.9882 FLAT (not U-shaped). Cor 25.1 REJECTED.
- Spectral radius rho = 1.0018 (SUPERCRITICAL — no stable FP)
- Readout T_99 = 4, but state never converges (moves 7 units/step forever)
- Geometric decomposition: 58% contractional / 82% rotational early,
  shifting to 81% / 59% late. Readout converges during balanced phase.

**Revised interpretation:** Dynamics reach a MANIFOLD M of correct-readout
states in ~4 steps, then evolve along M indefinitely. M is transversally
stable (readout perturbations corrected) but longitudinally neutral
(state continues evolving on M). This explains the 97x T_99 discrepancy.

**Theory impact:** Prop 25 (k~0.4) REJECTED at L=4 easy regime. Cor 25.1
(U-shaped k_t) REJECTED at L=4. Banach framework may still apply at
harder tasks (L>=8) — PENDING D28 higher-L data. "Falsify-by-regime":
the manifold picture holds for easy tasks; Banach may hold for harder ones.

**Corollary 25.2: Two-Step Acceleration.** Variable-T training reduces
T_99 by exactly 2 steps (confirmed at L=4,8,12,16 in D23). Universal
for all tested depths D=2-8.

**Cross-experiment supercriticality (2026-05-24, UPDATED):** 8 models
measured. Summary:

| Model | L | Variant | rho | Source |
|-------|---|---------|-----|--------|
| D28 fixed_t | 4 | fixed_t | 1.002 | D28 |
| D28 variable_t | 4 | variable_t | 0.999 | D28 |
| D25 s42 vt | 8 | variable_t | 1.028 | D25 |
| D25 s1337 vt | 8 | variable_t | 1.051 | D25 |
| D25 s2024 vt | 8 | variable_t | 1.048 | D25 |
| D25 s42 rg | 8 | recovery_gentle | 1.018 | D25 |
| D25 s1337 rg | 8 | recovery_gentle | 1.048 | D25 |

7/8 models show rho > 1 (range 1.002-1.051). Exception: D28 variable_t
at L=4 has rho=0.999 (barely subcritical). KEY: rho scales with problem
complexity — trivial L=4 stays near criticality, moderate L=8 goes
supercritical. Variable-T L=8 mean rho=1.042, fixed-T L=4 rho=1.002.
This is consistent with Edge-of-Stability self-organization applied
to forward dynamics.

[Proof: bottleneck_depth_scaling.md, Proposition 28 + Corollary 25.2]
[Calibration: MODERATE-HIGH for rho>1 at L>=8 (7/7 models).
 Open: k_t shape and manifold geometry at L=12+. D28 L=8+ pending.]

### 2.23 Computational Strategy Bifurcation (bottleneck_depth_scaling.md)

**Result (Proposition 29, EMPIRICAL — 3 seeds at L=12, CONFIRMED).** At
sufficient computational depth (D >= 6), the optimization landscape
bifurcates into multiple basins implementing different strategies:

- Strategy A (re-read-dependent): delta = +86.0% (L=12 seed=42)
- Strategy B (self-attn-dominant): delta = +11.0% (L=12 seed=1337),
  +23.8% (L=12 seed=2024)

All 3 achieve >=99.93% with full re-reading. Bimodal gap: 62+ pp between
A and B clusters, vs ~13pp within-B variance (5:1 separation ratio).

P_A(D=4) = 0/3, P_A(D=6) = 1/3. Bifurcation is minority outcome.
Strategy B correlates with better noise robustness (σ=0.1 seq_acc: s2024
24.7% vs s42 1.3%).

[Proof: bottleneck_depth_scaling.md, Proposition 29]
[Calibration: LOW→MODERATE — 3 seeds confirm bimodal split. Need n>=8 for
 statistical power on P_A estimate.]

### 2.24 Complexity-Dependent Criticality (bottleneck_depth_scaling.md)

**Result (Proposition 30, REVISED — Training Horizon Strain):** The
spectral radius rho of the Jacobian at the readout-stable manifold
depends on BOTH task depth D AND the ratio of D to training horizon T:

    ln(rho(D, T)) = f_complex(D) + g_strain(D/T)

where:
- f_complex(D): complexity-dependent term, peaks at D≈4, declines
  beyond (readout dimensionality squeeze — original mechanism)
- g_strain(eta): horizon strain term, peaked at eta = D/T ≈ 1

The quadratic model (V_crit saturation) is **FALSIFIED** by D=10 FT
data (rho=1.0042, highest observed, vs predicted ≈1.000). The Training
Horizon Strain model replaces it, explaining rho as a two-component
function of depth AND training-horizon ratio.

**Physical mechanism:** When D/T → 1 (FT) or D/T_min → 1 (VT), the
carry computation fills all available steps. Gradients shift from
contraction-favoring (convergence margin) to expansion-favoring
(information flow), increasing rho. At D/T >> 1 (unsolvable minimum-T
batches), gradient coherence degrades and the effect attenuates.

Evidence (D28, within-experiment):
- FT rho: D=2→1.0018, D=4→1.0026, D=6→1.0024, D=8→1.0016, D=10→1.0042
  TWO-PHASE: decline D=4→8 (complexity effect), SURGE D=10 (strain).
- VT rho: D=2→0.9994, D=4→1.0001, D=6→0.9996, D=8→1.0030
  Δρ sign reversal at D=8 (strain from D/T_min=2.0 exceeds regularization).
- ALL FT rhos are supercritical (>1), VT near/below 1 for D≤6.

**PREDICTIONS (from strain model):**
- D28 L=20 VT (D=10): rho_VT > 1.004, likely 1.005-1.007 (HIGHEST in D28)
- D28 L=24 FT (D=12): rho_FT ∈ [1.003, 1.006] (past strain peak)
- D30 Config D (T_min=8): rho ≈ 1.001-1.002 (weak regularization)
- D30 Config E (T=10 fixed): rho ≈ 1.0026 (recover D28 L=8 FT)

**Alternative models (Codex-suggested, equally plausible with 9 points):**
- Model A: piecewise linear with kink at D/T=0.8 (no peak claim)
- Model B: logistic phase boundary at D/T=1 (smooth step function)
- Discriminating test: D28 L=24 FT (eta=1.2) — Model A predicts
  continued increase, Model B predicts plateau, Strain model predicts
  decline from peak.

[Proof: bottleneck_depth_scaling.md, Proposition 30]
[Calibration: LOW — 3/10. Quadratic model DOUBLY FALSIFIED. Training
 Horizon Strain model proposed as replacement — explains all 9 data
 points qualitatively but under-identified (Codex review). Three
 competing model classes fit equally. D28 L=24 FT is the discriminator.]

**Result (Corollary 30.2, NEW): Solvability Boundary (Props 30-32
Unification).** The T_99 saturation (Prop 32) and rho strain (Prop 30)
are governed by the same solvability fraction q(D, T_min):
- q ≈ 1 → T_99 = T_min (binding constraint) and Δρ < 0 (coherent reg)
- q << 1 → T_99 = D_intrinsic (task binding) and Δρ > 0 (gradient conflict)
The transition q ≈ 0.5 at T_min ≈ D_intrinsic simultaneously explains
T_99 saturation at D_intrinsic, Δρ sign reversal at D=8, and rho surge
at D=10. q is dynamic (increases during training via C_step adaptation).
[Proof: bottleneck_depth_scaling.md, Corollary 30.2]
[Calibration: LOW-MODERATE — 4/10. Unifies three observations but q
 not directly measured. Upgrade path: per-sample T_99 during training.]

**Result (Corollary 30.1, REVISED — CONDITIONAL):** Variable-T training
reduces the spectral radius by approximately constant delta, BUT ONLY
when T_min ≥ D_intrinsic:
  ρ(VT, D) ≈ ρ(FT, D) - Δρ_VT,  Δρ ≈ 0.0026  (if T_min ≥ D)
  ρ(VT, D) > ρ(FT, D)                            (if T_min << D)

Evidence (D28, within-experiment, T_min=4 for all VT):
  D=2:  Δρ = -0.0024 (T_min > D) ✓
  D=4:  Δρ = -0.0025 (T_min = D) ✓
  D=6:  Δρ = -0.0028 (T_min < D, borderline) ✓
  D=8:  Δρ = +0.0014 (T_min << D) **LOCALIZED ANOMALY**
  D=10: Δρ = -0.0025 (T_min << D) **CONSTANT RESTORED** (2026-05-24)

D=10 VT (rho=1.0017, acc=95.7%, k=0.987) restores constant-Δρ for
4/5 depths. D=8 is a 20σ outlier from mean Δρ=-0.0026±0.0002. May
be seed-dependent fluctuation rather than a phase transition.

**Prediction (revised):** D=12 VT: Δρ ≈ -0.0025, acc < 93%.

[Proof: bottleneck_depth_scaling.md, Corollary 30.1]
[Calibration: MODERATE — 6/10 (upgraded from 5/10 after D=10 VT confirms
 constant Δρ). D28 12/12 COMPLETE: THREE regimes discovered.
 Constant Δρ≈-0.0025 at D={2,4,6,10}. Anomaly +0.0014 at D=8.
 VT suppression VANISHES at D=12 (Δρ=0.0000). D31 multi-seed D=8
 replication in progress (8 seeds). Updated model: Δρ ∝ -q(D,T_min).]

### 2.24b Gradient-Dynamics Coherence Profile (bottleneck_depth_scaling.md)

**Result (Proposition 34, NEW 2026-05-24):** The Training Horizon Strain
mechanism (Prop 30) predicts a measurable per-step gradient alignment profile
A(t) = |cos(dL/ds_t, v_1(J_t))| that distinguishes the competing rho models.

Three regimes governed by eta = D/T_train:
- Sub-critical (eta << 1): Bimodal A(t) — high during computation, low
  during convergence steps → gradient drives contraction.
- Critical (eta ~ 1): Uniformly high A(t) — no convergence margin →
  every step's gradient aligns with expansion → rho surges.
- Super-critical (eta >> 1): Noisy A(t) — unsolvable batches produce
  incoherent gradients that average out → strain attenuates.

Model discrimination:
- Model C (Peaked Strain) uniquely predicts A_mean PEAKS at eta~1 then
  DECLINES, while sigma_A DIPS at eta~1 then RISES.
- Models A (Piecewise) and B (Logistic) make no alignment predictions.

Connection to Prop 31: A(t) profile determines the FTLE partition.
High A(t) → selective lambda_R contraction; low/random A(t) → uniform
lambda_perp effects. Predicts |lambda_R|/lambda_perp should be SMALLER
at higher eta (more uniform gradient alignment).

[Proof: bottleneck_depth_scaling.md, Proposition 34]
[Calibration: LOW — 3/10 (downgraded from 4/10). Three-regime model
 COLLAPSED by D31 (8 seeds at D=8 show mean Δρ=-0.001, three-regime sign
 reversal was seed=42 artifact). The gradient coherence profile A(t)
 prediction remains untested but the motivating observation is gone.
 HOWEVER: The Δρ(D,T_min) ≈ -A·q(D,T_min) formulation may still hold
 if reframed in terms of contraction rate k rather than rho — D31 shows
 Δk is the primary mechanism (p=0.000017), not Δρ (p=0.083).
 D33 (IN PROGRESS) will provide independent q(T_MIN) measurement.
 **STATUS: NEEDS REFORMULATION** — replace rho-centric prediction with
 k-centric version. See Proposition 35 for the empirical k mechanism.]

### 2.25 Anisotropic Readout Convergence via FTLE Decomposition (bottleneck_depth_scaling.md)

**Result (Proposition 31):** The readout-stable manifold (Prop 28) and
supercritical dynamics (rho > 1, Prop 30) are reconciled by anisotropic
contraction. The Jacobian contracts rapidly in readout-relevant directions
(FTLE < 0) while expanding in readout-orthogonal directions (FTLE > 0).

Key results:
(i)   Readout converges at rate exp(lambda_R * t) independent of rho
(ii)  Manifold stability: transversal corrections at rate exp(lambda_R),
      longitudinal expansion at rate exp(lambda_perp)
(iii) rho = exp(lambda_1) where lambda_1 = max FTLE, which lives in V_perp.
      T_99 depends on lambda_R << lambda_1, resolving the rho>1 paradox.

The readout function R acts as a "contraction observer" that only sees
contracting directions. This leverages the non-normality of the
transformer Jacobian (Kreiss theorem allows w(J) to exceed rho(J)
by up to e*n, but FTLE decomposition gives direction-dependent rates).

Connects: Prop 25 rejection (global k ≈ 0.99 is blended average, not
useful), Prop 27 (two-phase alignment = FTLE evolution), Prop 28
(manifold = readout-contracting FTLEs already acted), Prop 30
(higher D → more readout-orthogonal computation → higher lambda_perp).

**D28 D=8 three-way decoupling (key Prop 31 confirmation):**
VT at D=8 has rho=1.003 (HIGHER than FT 1.002) but T_99=3 (LOWER than
FT T_99=6) and k=0.988 (LOWER than FT 0.991). This triple decoupling
is precisely what Prop 31 predicts: rho measures max(lambda_perp),
T_99 measures lambda_R, and k measures the blended average. VT can
worsen lambda_perp (raising rho) while improving lambda_R (lowering
T_99). The readout function sees only lambda_R, confirming it as a
"contraction observer" blind to null-space expansion.

Testable via D29: SVD of product Jacobian at manifold, partition
singular vectors by readout alignment, measure lambda_R and lambda_perp.

[Proof: bottleneck_depth_scaling.md, Proposition 31]
[Calibration: MODERATE-STRONG — 7/10 (D29b directional support, Codex
 evidence review 6/10). 72/72 measurements show lambda_R < 0 AND
 lambda_null > 0 (sign p < 0.003). Separation 10x in magnitude.
 CAVEATS (Codex): (1) P_crit @ Phi @ P_crit is not standard restricted
 FTLE (should be Q^T Phi Q), (2) missing readout normalization Jacobian
 in critical directions, (3) FP not converged (residual ~14%). Single
 config only (VT, L=8, seed=42) — needs multi-config replication.
 IMPORTANT: lambda_R = -0.004 measures manifold STABILITY, not
 convergence speed. T=1→T=2 accuracy jump (15%→98%) is nonlinear,
 far exceeding exponential rate at lambda_R. FTLE characterizes
 perturbation rejection, not approach dynamics.]

**Corollary 31.1: Manifold Stability Timescale and U-Shaped Update Norms**

The update norm trajectory ||G(s_t) - s_t|| is U-shaped when lambda_R < 0
(approach phase decreases) AND lambda_perp > 0 (manifold phase increases).
The U-shape minimum occurs at T_min ~ T_99 + O(1/lambda_perp). Manifold
escape time T_escape ~ T_99 + (1/lambda_perp) * ln(r_basin/delta_perp).

**Empirical verification (D28, 2026-05-24):**
- L=4 fixed_t: monotonic decrease (12.5→5.7) — lambda_perp too small for
  observable U-shape in 30-step window. rho=1.0018.
- L=8 fixed_t: U-SHAPED! Minimum at step 18 (17.9→11.5→12.2). Readout
  degrades 99.7%→83.4% from step 13-30. Effective finite-time
  lambda_perp ~ 0.005 per step (~2x the asymptotic ln(1.0026)=0.0026).

**Predictions:** L=12+ should show U-shape minimum shifting EARLIER
(higher lambda_perp from Prop 30). Finite-time lambda_perp should
exceed asymptotic value by ~2x (non-normal transient amplification,
Trefethen & Embree 2005).

[Proof: bottleneck_depth_scaling.md, Corollary 31.1]
[Calibration: MODERATE — 6/10. D29b directly measures: mean lambda_R ≈ -0.005
 (contracting), mean lambda_null ≈ 0.064 at T=5 (expanding). Finite-time
 lambda_null is ~25x larger than asymptotic ln(rho)=0.003, confirming
 transient amplification prediction. U-shape at L=8 (D28) is consistent.]

**Corollary 31.2: Three-Way Metric Decoupling (k, rho, T_99)**

Three commonly measured diagnostics capture orthogonal aspects of the
FTLE spectrum:

    k_frob ≈ exp(mean_i lambda_i)     [average over ALL FTLEs]
    rho    = exp(max_i lambda_i)       [maximum FTLE, in V_perp]
    T_99   ≈ ceil(ln(0.01) / lambda_R) [readout-aligned FTLE]

When lambda_R < 0 < lambda_perp (the anisotropic regime from Prop 31),
these metrics can move INDEPENDENTLY:
- k < 1 while rho > 1 (contractive average, expansive maximum)
- T_99 can decrease while rho increases (readout improves, null space worsens)
- VT training can adjust lambda_R (via T_min gradient) without improving
  lambda_perp, or vice versa

**Empirical confirmation (D28, D=8):**
    | Metric | FT    | VT    | VT effect |
    |--------|-------|-------|-----------|
    | k      | 0.991 | 0.988 | ↓ better  |
    | rho    | 1.002 | 1.003 | ↑ WORSE   |
    | T_99   | 6     | 3     | ↓ better  |

VT improves readout (lambda_R) and average contraction (mean lambda)
while worsening null-space stability (lambda_perp / max FTLE). This
is possible ONLY because the dynamics are anisotropic. An isotropic
system (all lambda_i equal) would have k = rho^{1/d} and T_99 fully
determined by rho — no decoupling.

**Implication for UESD evaluation:** rho alone is not a reliable
diagnostic of VT effectiveness. T_99 (readout convergence) is the
functionally relevant metric. rho measures null-space behavior that
does not affect readout accuracy. Cor 30.1 "failure" at D=8 is a
failure of the rho diagnostic, not of the VT mechanism itself.

[Proof: Follows directly from Prop 31 FTLE decomposition]
[Calibration: STRONG — 7/10. D28 D=8 is the cleanest example of all
 three metrics moving independently. Consistent with D29b FTLE data
 (lambda_R and lambda_null measured independently). The mathematical
 relationship k ≈ exp(mean(lambda_i)) is approximate but directionally
 confirmed across 8 D28 configs.]

**Proposition 32: Variable-T Training Tightens Readout FTLE Bound**

Variable-T training (T sampled from {T_min, ..., T_max}) tightens the
readout FTLE bound compared to fixed-T training:

    lambda_R(VT) <= ln(epsilon) / T_min  <  ln(epsilon) / T_0 = lambda_R(FT)

The gradient from T = T_min batches imposes the binding constraint:
readout must converge within T_min steps. This explains why T_99 = 3
universally for variable_t (with T_min = 4) across D = 2-10.

Key predictions:
- lambda_R(VT) ~ -1.5 (from T_99 = 3)
- lambda_R(FT) ~ -0.5 to -0.9 (from T_99 = 5-6)
- lambda_perp is training-independent (architectural, via Prop 30)
- Changing T_min changes T_99 proportionally (D30 tests this)

Empirical evidence: T_99 = 3 at L=4,8,12,16,20 (D23, 5 data points).
**EXCEPTION: L=24 (D=12) gives T_99=5, matching baseline.** The VT
overshoot diminishes when intrinsic computation depth exceeds T_min.
Refined bound: T_99 ≤ max(T_min, D_intrinsic).
k_frob(FT) ≈ k_frob(VT) at L=4 (0.988 vs 0.990) — orthogonal
dynamics are architectural, readout contraction is training-tunable.

**D30 DIRECT TEST (T_min control at D=4, 2/5 configs done):**
Config A: T_min=2 → T_99=2, rho=0.9992. PASS.
Config B: T_min=4 → T_99=4, rho=1.0001. PASS. **H1 CONFIRMED.**
Configs C-E training (T_min={6,8}, fixed T=10).

**D30 RESULT: T_99 = min(T_min, D_intrinsic)**
Two competing hypotheses were pre-registered:
  H1 (training-determined): T_99 = T_min (gradient pressure sets speed)
  H2 (architecture-determined): T_99 = const ≈ 3
Config B (T_min=4) was the decisive test: T_99=4 → **H1 PARTIALLY CONFIRMED.**
Config C (T_min=6) REFINES: T_99=4 (not 6) → T_99 saturates at D_intrinsic.

Four data points establish T_99 = min(T_min, D_intrinsic):
  D30-A (T_min=2, D=4): T_99=2 = min(2,4) ✓
  D23 standard VT (T_min≈3, D=4): T_99=3 = min(3,4) ✓
  D30-B (T_min=4, D=4): T_99=4 = min(4,4) ✓
  D30-C (T_min=6, D=4): T_99=4 = min(6,4) ✓

~~REVISED PROP 32: T_99 = max(T_min, D_intrinsic)~~ FURTHER REVISED:
T_99 = min(T_min, D_intrinsic). VT training compresses readout
convergence to T_min steps when T_min ≤ D, but cannot push T_99
beyond D_intrinsic — readout convergence requires solving the task.

**rho DEPENDS ON T_min (3 data points now):**
Config A: rho=0.9992 (Δρ from FT=0.0034)
Config B: rho=1.0001 (Δρ from FT=0.0025)
Config C: rho=1.0006 (Δρ from FT=0.0020)
rho increases monotonically with T_min: ≈ 0.999 + 0.00035*T_min.
Lower T_min → larger Δρ → stronger spectral radius suppression.
This refines Cor 30.1: Δρ = f(T_min), not a universal constant.
**D28 D=8 FALSIFICATION:** Δρ reverses sign when T_min << D_intrinsic.
D30 varies T_min at fixed D=4; D28 varies D at fixed T_min=4. Together
they reveal Δρ = f(T_min, D) with a phase boundary near T_min ≈ D.

Remaining predictions (updated after Config C):
  Config D (T_min=8): T_99 = 4 (=D), rho ≈ 1.001
  Config E (fixed T=10): T_99 = 5-6 (D23 baseline), rho ≈ 1.003
  rho should continue increasing: A < B < C < D < E

[Proof: bottleneck_depth_scaling.md, Proposition 32]
[Calibration: MODERATE — 6/10 (was 7/10 but Config C falsifies strict
 T_99=T_min equality. 4 data points for revised min(T_min,D) form.
 rho monotonicity confirmed at 3 points.
 **D31 UPDATE (2026-05-25):** D31 provides the strongest evidence for
 Prop 32 via contraction rate k, not rho. VT Δk=-0.0023 with p=0.000017
 (8/8 seeds, d=-3.9). This is the empirical signature of tighter readout
 FTLE. The k mechanism is the QUANTITATIVE backbone of this proposition.
 Prop 32 should be reframed to predict Δk directly:
   k(VT) ≈ exp(lambda_R_VT) and k(FT) ≈ exp(lambda_R_FT)
   Δk = k(VT) - k(FT) ≈ -0.0023 ± 0.0006
 D32 confirms k suppression persists even WITHOUT task learning (geometric).)]

**Corollary 32.1: Complexity-Bound Crossover.** The VT overshoot
(T_99 < T_min) holds iff D_intrinsic < T_min. At D=12, D_intrinsic ≈ 5
exceeds T_min=4, so T_99(VT)=5=T_99(FT). The binding constraint is
effective only when the task fits within the training horizon. Note that
high-T robustness (canalization, Prop 26) is a SEPARATE mechanism that
persists regardless (T=48: 99.95% at D=12).

[Evidence: D23 complete — 6 data points L=4..24, crossover at D=12.
 D30 config A: T_min=2 at D=4 gives T_99=2 (D_intrinsic=4 > T_min=2,
 so T_99=max(T_min,?)=2 → D_intrinsic<T_min NOT required here, just
 T_99=T_min. Consistent with bound.]
[Calibration: 5/10. D_intrinsic estimates are post-hoc. D30 confirms
 T_min mechanism but doesn't directly test crossover.]

### 2.26b Primary VT Mechanism: Contraction Rate Suppression (NEW 2026-05-25)

**Result (Proposition 35, NEW 2026-05-25):** Variable-T training primarily
improves the generalized contraction ratio k through T_min-dependent gradient
pressure on readout-relevant directions, NOT through spectral radius
suppression.

Statement:
  k(VT) < k(FT) with Δk ≈ -0.0023 ± 0.0006 (at D=8)
  rho difference is secondary: Δρ ≈ -0.001 ± 0.002 (not significant)
  Effect sizes: Cohen d(k) = -3.92, Cohen d(rho) = -0.59

Mechanism: Gradient from T=T_min batches imposes the tightest convergence
constraint. To minimize CE loss at T_min, the dynamics must contract the
readout-relevant subspace faster, which directly suppresses k. The rho
suppression (if any) is a secondary consequence of this contraction
tightening — rho measures the worst-case (null-space) expansion, while k
measures the average contraction including readout-relevant directions.

Key distinction (k vs rho):
- rho = spectral radius of Jacobian at fixed point = max eigenvalue
  magnitude = local stability indicator at s*
- k = average contraction ratio along trajectory = exp(mean FTLE) =
  GLOBAL convergence behavior during iteration
- Relationship: k captures readout-relevant contraction (via lambda_R);
  rho captures null-space expansion (via lambda_perp). When readout is
  low-dimensional (~0.4% of state space per Prop 33), k is dominated
  by the orthogonal bulk, while readout-relevant contraction is a small
  correction. VT's effect concentrates on this small correction.

Predictions:
1. Δk scales with T_min: lower T_min → larger |Δk| (tighter binding)
2. k suppression persists without task learning (D32 confirms: geometric)
3. k suppression is universal across depths where T_min ≤ D_intrinsic
4. At D=12 where VT suppression vanishes (D28), Δk should also vanish

Relationship to existing propositions:
- Unifies Props 30 (rho strain), 31 (FTLE decomposition), 32 (VT readout
  FTLE tightening) into a k-centric framework
- Spectral radius (rho) is relegated to a diagnostic of null-space behavior,
  not the primary control variable for task performance
- Corollary 31.2 (metric decoupling) is ELEVATED: it correctly identified
  that k, rho, and T_99 respond to different FTLE components

[Evidence: D31 (8 seeds at D=8, 28/28 total runs). k: paired t=-10.36,
 p=0.000017, d=-3.92, 8/8 seeds unanimous. rho: paired t=-1.55,
 p=0.083, d=-0.59. T_99 (clean): paired t=-6.58, p=0.0006.
 D32: VT k suppression preserved even without task learning (24/24 runs).
 D28: k measured across D=4-12 (12 configs).]
[Calibration: HIGH — 8/10. The 8-seed unanimous result with d=-3.9 is
 the strongest statistical evidence in the entire UESD program. The
 k vs rho distinction resolves multiple theoretical puzzles (why rho>1
 works, why D28 VT suppression was noisy). Remaining: (a) T_min scaling
 prediction untested at varied T_min, (b) D=12 Δk prediction untested,
 (c) single architecture family.]

### 2.27 Necessary Anisotropy Theorem (bottleneck_depth_scaling.md)

**Result (Proposition 33, THEOREM — proven, then REFINED after D29):** Any
dynamical system with supercritical spectral radius (rho > 1) and stable
readout MUST have contracting margin-critical FTLE. This is a necessary
geometric consequence, not an empirical observation.

**REFINED STATEMENT (2026-05-24):** The proof applies to V_CRIT (margin-
critical subspace, dim ~L/2), NOT the full readout row space V_R (dim ~Ld/2).
D29 revealed that dim(V_R) = 512/1024 = 50% of state space, so V_R is
too large for the original "V_R contracts" claim. The refined version:

(i)   lambda_CRIT ≤ 0 (margin-critical directions contract)
(ii)  lambda_max > 0 lives OUTSIDE V_crit (but may be inside or outside V_R)
(iii) T_99 = O(1/|lambda_crit|), independent of rho

The margin-critical subspace V_crit = span{W_R^T(e_{y*} - e_{j*})} per
position has dim ~L/2 = 4 for L=8, covering only 0.4% of state space.
The full readout row space V_R (50% of state) can contain BOTH contracting
and expanding directions — only V_crit must contract.

**D29 empirical support:** lambda_R(max) > 0 but lambda_R(mean) = -0.008
(negative). The bulk contracts, only the extreme tail expands. This is
CONSISTENT with refined Prop 33: V_crit (in the contracting bulk) contracts,
while some V_R\V_crit directions expand freely.

**Corollary 33.1 (Margin Dimension):** dim(V_crit)/n = 1/(2d) = 0.4%.
Only 4 out of 1024 dimensions must contract. The remaining 99.6% can
expand or contract freely without affecting readout correctness.

**Corollary 33.2 (Banach Paradox Resolution):** k_frob ≈ 0.988 reflects
the average over ~1020 mildly contracting + ~4 strongly contracting
directions. The bulk k ≈ 0.99, while k_crit << 1 (consistent with T_99=3).
The original isotropic k ≈ 0.4 prediction is wrong because contraction
is ANISOTROPIC by necessity.

**Corollary 33.3 (Sample-Dependent Criticality):** D29 shows rho(power
iteration, batch avg) = 0.9997 while rho(FTLE, per-sample) = 1.03-1.07.
The dynamics are input-dependent: harder samples are more supercritical.
Extends Prop 30 to within-task variation, not just across-task scaling.

**Theory impact:** Refined Prop 33 provides the mathematical foundation
for Prop 28-31-32 chain, but with V_crit not V_R as the contracting
subspace. D29b (running) will provide the first direct measurement of
lambda_crit via margin-critical decomposition.

[Proof: bottleneck_depth_scaling.md, Proposition 33]
[Calibration: HIGH — 8/10 (downgraded from 9/10 after D29 refinement.
 The theorem is mathematically sound but the original V_R formulation was
 incorrect — corrected to V_crit. The refinement is tighter and empirically
 supported. D29b will provide quantitative test.)]

---

## 3. Diagnostic-to-Theory Mapping

| Diagnostic | Measures | Theorems | Threshold |
|-----------|----------|----------|-----------|
| D1: Token accuracy | Does readout work? | — | >= 90% |
| D2: Normalized residual | How close to fixed point? | 2.4 (finite T) | < 0.01 |
| D3: Decoder margin | How confident is readout? | 2.1 (margin preservation) | > 0 |
| D4: Wrong-attractor rate | Convergence-correctness coupling? | 2.2, 2.3 | < 5% |
| D5: Basin perturbation | Basin size and non-normal stability? | 2.1, 2.4 (Thm 4), 2.7 | >= 90% |
| D6: Random-matrix null model | Jacobian rotation: learned or statistical? | 2.6, 2.7, 2.21 (Prop 27) | alpha = actual/nullB near 1 (statistical) or >1 (directed) |
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
6. Variable-T training tightens sigma_max bound (Theorem 9)
7. CE-only training cannot produce positive recovery (Theorem 13)
8. T_min = max(O(log V), O(D)) for V tokens and depth D (Theorem 17)

### NOT Guaranteed

1. Global existence of correct fixed points for arbitrary contexts
2. Absence of wrong attractors (only empirically testable via D4)
3. Generalization beyond training distribution (only Lipschitz bounds)
4. Convergence in T = 10 steps when sigma_max >= 1 (even if rho < 1)
5. Non-normal transient growth bounded analytically (Theorem 4 gives
   sigma_max^T bound; D5 empirically validates; D7 quantifies severity)
6. Spectral radius staying < 1 during training (only at convergence)
7. Positive recovery without explicit perturbation loss (Theorem 14)
8. C_step value (serial vs parallel carry) for specific architectures

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
- CE gradient contains no basin-shape information (Theorem 11, SOUND)
- Perturbation loss provides needed Jacobian gradient (Theorem 14 core)

**MODERATE claims (conditionally valid):**
- Basin size scales as (1-rho)/M (local linearization, holds near s*)
- Finite-T error bounded by sigma_max^T * ||s_0 - s*|| (Theorem 4,
  rigorous one-step bound, induction over T steps)
- Generalization radius depends on Lipschitz constants (IFT)
- Wrong-attractor rate under shift bounded by Wasserstein distance
  (Theorem 7, requires Lipschitz constants of fixed-point map)
- Variable-T sigma_max bound: conditional on per-horizon CE bound
  and local regime (Theorem 9, needs A9.1-A9.5)
- Recovery non-positive under clean basin geometry (Theorem 13,
  conditional on A13.1-A13.3; weak positive possible with overlap)
- T_readout = O(log V / log(1/sigma_max)) conditional on m_V scaling
  (Theorem 15, needs A15.3 empirical assumption)
- T_depth ~ D for serial carry propagation (Theorem 16, testable by D23)
- Phase transition at L ~ 16-20 in carry-depth (Theorem 17 prediction)

**MODERATE claims (partially confirmed, new):**
- T_min Saturation at T=5 via parallel attention (Proposition 24,
  6 data points D=2-12, single architecture/task/seed)
- Canalization via variable-T training (Proposition 26): D22
  compute window elimination + D25 +27.5% implicit recovery +
  D22 denoising failure all consistent. Coherence gap untested.

**REJECTED claims (empirically falsified):**
- Banach contraction k ≈ 0.4 (Proposition 25 — D28 k=0.988 FLAT)
- U-shaped k_t profile (Corollary 25.1 — D28 FLAT at L=4)

**WEAK-to-MODERATE claims (directional with supporting data):**
- Signal amplification via channel-coding dynamics (Proposition 22):
  dynamics amplify weak encoder signals by exploiting inter-position
  constraints (carry chain as parity checks). L=24 achieves 99.22%
  from E_enc=0%. Channel-coding analogy is structural, not
  derivational — dynamics are not literally belief propagation.
  D27 CONFIRMS: cross-attention re-reading adds +28.4% accuracy;
  dynamics without re-reading still achieve 71.5% via self-attention.
  Encoder fragile (σ=0.05 → 6% drop, σ=0.1 → catastrophic).
- Criticality-optimal recovery at rho ≈ tanh(1/2) (Proposition 21):
  supported by D6/D15/D23 measurements and cross-domain evidence,
  but Nishimori connection is analogical. D25 spectral measurements
  will upgrade or falsify. Predicts recovery training drives spectral
  radius toward 0.462 and explicit criticality targeting outperforms
  implicit basin shaping.

**WEAK claims (directional, new):**
- Anti-oscillation via multi-horizon consistency (Proposition 10,
  depends on diagonalizability and directional sensitivity)
- T_min = max(T_readout, T_depth) decomposition (Theorem 17,
  interaction term I(V,D) not bounded)
- Non-monotonic T_min vs D due to difficulty-dependent contraction
  (Proposition 19, MODERATE per Codex: 6 data points including L=24
  but narrow task/architecture scope, 1 seed per L, need multi-seed)
- Cross-domain connections to Nishimori rho = tanh(1/2) and IB
  criticality (suggestive, not derivational for UESD)

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

---

## 6. Cross-Domain Connections (from research mining, 2026-05-24)

### 6.1 Edge-of-Stability Self-Organization (ρ=1.028)

The observed spectral radius ρ=1.028±0.015 maps onto the Edge-of-Stability
phenomenon (Cohen et al. 2021): during gradient descent, the largest Hessian
eigenvalue self-tunes to 2/η (the stability threshold). The Lyapunov exponent
evolves toward zero — the edge of chaos.

Neural analogy: Priesemann et al. (2014) measured the neural branching ratio
at σ=0.98 (slightly subcritical). True criticality σ=1.0 risks runaway
dynamics (epilepsy). UESD's ρ=1.028 being slightly supercritical (not
subcritical like brains) may reflect that generation benefits from mild
signal amplification, while perception benefits from stability.

Implication: ρ near 1 is not a defect — the training dynamics self-organized
F_theta to this operating point. At the edge of chaos, the system achieves
both sensitivity to new inputs and preservation of past state.

### 6.2 Banach Contraction and Universal T_99=5

The Banach Contraction Mapping Theorem: any contraction mapping on a complete
metric space has exactly one fixed point, found by iteration with geometric
convergence: ||s_t - s*|| ≤ k^t ||s_0 - s*||.

If T_99=5 universally, this implies k^5 < margin_threshold, giving
k < 0.01^(1/5) ≈ 0.398. The universality across carry depths D=2-12
would mean k is problem-independent: the dynamics F_theta are an
equally-fast contractor regardless of input difficulty.

Testable (D28): measure the actual per-step contraction ratio
k_t = ||s_{t+1} - s*|| / ||s_t - s*||. If k ≈ 0.4 constantly across
D and t, the Banach theorem fully explains T_99=5.

Tension with ρ=1.028: the linearization at s* is slightly expansive,
but the nonlinear dynamics may be globally contractive within the
relevant basin. This is consistent with the transformer's nonlinearities
(ReLU/GELU) compressing large states.

### 6.3 Waddington Canalization via Variable-T Training

In developmental biology, canalization (Waddington 1942) channels
developmental trajectories into "creodes" — deep valleys resistant
to perturbation. Rozum et al. (2025) formalized this: canalization
carves deep protective valleys but creates a "coherence gap" — the
system is robust to large perturbations but sensitive to small
structural ones.

Variable-T training IS canalization engineering: the randomized stopping
time prevents over-commitment to a single convergence trajectory,
deepening attractor valleys without explicit robustness training.
The +27.5% recovery at σ=0.2 is the signature.

Falsifiable prediction: variable-T models should exhibit the coherence
gap — robust to noise (stay in correct basin) but sensitive to
fine-grained structural perturbations that distinguish close solutions.

### 6.4 Noise-Induced Order and Recurrence Resonance

Noise-induced order: chaotic systems can become MORE ordered when noise
is added, because noise preferentially kicks the system toward unstable
periodic orbits, stabilizing them. Recurrence resonance (2024): noise
in RNN dynamics pushes the network out of low-entropy attractors,
enabling state-space exploration.

Variable-T training is recurrence resonance engineering: the training
noise (variable stopping time) prevents the dynamics from over-committing
and forces the network to learn dynamics robust across multiple trajectory
lengths. This explains why variable-T recovery (+27.5%) exceeds any
explicit denoising training approach (which completely failed in D22).

### 6.5 Self-Consistency Energy as Morphogenetic Blueprint

Levin's bioelectric pattern memory: cells store a target morphology as
voltage patterns independent of individual cells. When cut, cells detect
the discrepancy between current and target state, then act to minimize it.

E(s) = ||F_theta(s,c)||² IS the discrepancy measure. F_theta IS the
morphogenetic program. The embedding space IS the morphospace. The fixed
point s* IS the target morphology.

Key insight: the target pattern can be reprogrammed — two-headed planaria
maintain their altered morphology indefinitely. This suggests UESD could
support multi-task generation by modifying c to reprogram which fixed
point the dynamics converge to, with the same F_theta serving as a
universal morphogenetic engine. (Validated by D24: same dynamics,
different tasks via conditioning.)

### 6.6 Recurrent Depth and FTLE Signatures (D6 connection)

Recurrent Depth (arXiv:2502.05171) iterates a SINGLE recurrent block at
test time — the same architecture as UESD's weight-tied decoder. They
predict Finite-Time Lyapunov Exponent (FTLE) signatures during inference:
"when FTLE drops, the system has found the solution path."

D6's bimodal SV alignment IS the empirical FTLE signature they predicted
but never measured. The transition from low alignment (high FTLE, exploration)
to high alignment (low FTLE, contraction) at step ~5 is exactly the
FTLE drop marking solution discovery. This connects UESD directly to the
Recurrent Depth framework and provides the first empirical FTLE measurement
in a trained iterative network.

### 6.7 Non-Normal Transient Growth (STRONGEST NOVELTY CLAIM)

D6's finding that actual product-Jacobian sigma is 2.2x LARGER than
matched-spectrum random rotation is a textbook non-normal transient
growth phenomenon (Trefethen & Embree, Spectra and Pseudospectra, 2005).
Non-normal operators can amplify perturbations transiently even when all
eigenvalues are inside the unit disk.

Nobody has identified structured non-normality in trained iterative
networks. Existing work measures spectral radius or largest singular
value per step, but not the inter-step SV alignment structure. D6 shows
that trained dynamics learn STRUCTURED non-normality: early-step rotation
(diverse transformation) and late-step alignment (coherent contraction).

This is the strongest novelty claim of the UESD framework: trained
iterative dynamics don't just tune spectral radius — they organize the
ORIENTATION of their Jacobians to enable a two-phase computation strategy.

### 6.8 BP Convergence Two-Phase Pattern

Turbo codes and LDPC codes achieve channel capacity via iterative belief
propagation (BP). BP convergence exhibits a two-phase pattern: early
iterations diversify beliefs (extrinsic information exchange between
component decoders), late iterations converge (beliefs stabilize).

D6's bimodal SV alignment is the spectral fingerprint of this BP-like
convergence in UESD dynamics. The early orthogonal rotation = extrinsic
information exchange (each dynamics step transforms state in a different
direction). The late coherent alignment = belief convergence (each step
reinforces the same conclusion).

This strengthens Prop 22 (channel-coding dynamics): the dynamics don't
just act AS an iterative decoder — they exhibit the same spectral
convergence signature as actual BP. The number of BP iterations ≈ depth
of the network, and the iteration budget splits between exploration and
convergence just as in D6.

### 6.9 Readout-Projected Contraction and Nishimori Criticality

D28 preliminary data reveals k_frob = 0.9882 (barely contractive) yet
T_99(readout) = 4 — a 97x discrepancy (Proposition 28). The dynamics
don't converge to a fixed point in full state space; they converge in
the readout-relevant subspace.

This connects to the Nishimori criticality framework: systems self-
organize to the edge of chaos where layer-to-layer correlation converges
to rho = tanh(1/2) as a unique fixed point. The readout-projected
contraction is how this manifests operationally — readout-relevant
subspaces achieve criticality while orthogonal dimensions remain weakly
contractive.

Cross-domain evidence from reservoir computing (Bertschinger & Natschlager
2004): reservoirs compute optimally when tuned to the edge of chaos
because they preserve sensitivity while maintaining memory. Our k_frob
≈ 0.99 IS the edge-of-chaos signature: global dynamics are nearly neutral
(maximal dynamic range), but readout directions contract strongly (reliable
computation).

Non-normal transient amplification (Section 6.7) provides the mechanism:
even with spectral radius near 1, the structured Jacobian rotation
(Prop 27) creates exponential transient amplification in readout-relevant
directions before settling. The 97x gap between Frobenius T_99 and readout
T_99 is a direct measurement of non-normal transient growth.

Strongest implication: the Banach contraction interpretation (k≈0.4, Prop 25)
is DEFINITIVELY WRONG in full state space — D28 measured k=0.988 FLAT, rho>1
across 6 independent models. The dynamics work via readout-stable manifold
reachability (Prop 28), not contraction to a fixed point. The non-normal
transient behavior is what makes readout converge in 4 steps while the full
state never converges.

### 6.10 Depth-Dependent Re-Read Necessity and QEC Scaling (REVISED)

D27 multi-seed results (2026-05-24): cross-attention dependency
INCREASES with depth on average (L=8 mean delta 15.6%, L=12 mean 48.5%,
ratio 3.1x) but is HIGHLY SEED-DEPENDENT at L=12 (range 11-86%, CV=109%).
Corollary 22.1 (REVISED) reinterprets this as stochastic: the probability
of the optimizer finding a re-read-dependent strategy increases with D,
but the task CAN be solved without heavy re-reading even at D=6.

Original sigmoid fit (D*=4.95): DEFUNCT. I_single is not a fixed
architectural constant but a LEARNED quantity varying across seeds.

Three cross-domain connections (weakened by multi-seed data but still relevant):

1. **QEC code distance scaling** (ref: _meta error correction research):
   Google's Willow result shows increasing surface code distance requires
   more syndrome rounds. Analogy WEAKENED: if some seeds solve D=6 without
   iterative re-reading, the "code distance requires iterations" story is
   not absolute — it depends on the decoder (dynamics) structure.

2. **BP on loopy graphs** (ref: _meta computation in physics): Cross-
   attention in UESD performs BP-like message passing. The analogy now
   applies specifically to strategy-(A) seeds that learn re-read-dependent
   dynamics. Strategy-(B) seeds suggest alternative non-iterative
   decoding paths exist (analogous to MAP decoding vs iterative BP).

3. **Lyapunov exponent decomposition** (ref: _meta edge of chaos):
   k_frob ~ 0.99 reflects the slowest contracting direction. The
   readout subspace has the fastest (most negative) Lyapunov exponents.
   This connection is INDEPENDENT of the re-read question and remains
   strong — it explains D28's readout-stable manifold behavior.

Gap: L=12 seed=2024 (training now) will break the 1:1 tie. Need n >= 5
seeds at L=12 before making quantitative scaling claims.

### 6.11 Finite-Time Lyapunov and Readout-Stable Manifold Novelty

The readout-stable manifold (Prop 28) — dynamics with rho > 1 that still
produce correct readout in 4 steps — is GENUINELY NOVEL in the Nishimori/
SOC framework. Prior work (Storm et al. PRL 2024) characterized Finite-
Time Lyapunov Exponents (FTLE) in neural network dynamics. The _meta
THESIS distinguishes "static geometry" (the manifold of correct outputs)
from "dynamic geometry" (what's reachable by dynamics). Our finding
instantiates this distinction concretely:

- Static geometry: readout-stable manifold M = {s : argmax(Ws) = y*}
- Dynamic geometry: trajectory converges transversally onto M in 4 steps
  while expanding longitudinally (rho > 1)

The FTLE decomposition explains this: readout-relevant directions have
NEGATIVE finite-time Lyapunov exponents (fast transverse contraction)
while orthogonal directions have POSITIVE exponents (expansion along M).
The infinite-time spectral radius rho > 1 captures the expanding
directions; the 4-step readout convergence captures the contracting ones.

This resolves the "how does rho > 1 coexist with correct readout?"
paradox: the Jacobian is non-normal, so the spectral radius (eigenvalue-
based) does not capture the anisotropic contraction structure. The
FTLE (which captures finite-time, direction-dependent dynamics) reveals
the true mechanism.

Cross-domain validation: reservoir computing operates at the edge of
chaos precisely because readout-relevant information is amplified while
noise is suppressed — the same mechanism as our readout-stable manifold.
[Ref: _meta/inquiry/THESIS.md lines 69-84, 114-126]
[Ref: _meta/research/nishimori-cross-domain.md]
