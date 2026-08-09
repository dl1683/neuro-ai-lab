### Theorem-by-theorem ratings (9–18)

| Theorem | Status | Why |
|---|---|---|
| 9 (Variable-T contraction bound) | **NEEDS WORK** | The core inequality is mathematically consistent with finite-step bounds, but the key premise is underspecified: `L = E_T CE_T` does not imply low CE for every sampled horizon, yet the proof needs that to derive the `T_1` bound. Also requires explicit assumptions `m/(K d0)<1`, local linear regime, and non-normality control (`σ_max<1`, bounded Hessian term). |
| 10 (Anti-oscillation via multi-horizon consistency) | **NEEDS WORK** | The “sin(Tφ)” argument is directional and local, and assumes a stable decomposition into eigenplanes and meaningful `K_dir`. It needs explicit conditions (diagonalizability/normality vs non-normal coupling) and a normed bound on directional sensitivity. Otherwise it is a strong heuristic, not a theorem. |
| 11 (CE gradient sees only nominal trajectory) | **SOUND** | Chain-rule statement is correct and precise for `L=CE(readout(s_T))`. It correctly distinguishes direct CE gradients from perturbed-state information. Minor nuance: indirect effects via shared parameters are not ruled out, but “no direct basin-shape term at `s_T+ε`” is valid. |
| 12 (Basin properties are side effects) | **NEEDS WORK** | Part (a) is solid via contraction/Lipschitz assumptions. Parts (b)(c) overstate “not at all” language; CE can indirectly affect Jacobian geometry through parameter coupling, so this is directional rather than absolute. |
| 13 (Recovery trichotomy) | **NEEDS WORK** | The sharp trichotomy is too strong. It assumes a clean boundary and effectively a single-correct-attractor scenario. In realistic dynamics with overlapping/anisotropic basins and non-normal transients, recovery can be mixed and not strictly ≤0 by the given argument. |
| 14 (Perturbation loss needed for recovery) | **NEEDS WORK** (toward **SOUND**) | The core claim that perturbation-state training injects additional gradient geometry is correct, but “minimal sufficient loss” is asserted without a minimality proof. Needs explicit optimization scope and finite-K/σ bias terms. |
| 15 (T_readout ~ log V) | **NEEDS WORK** | Formula is algebraically coherent from `σ_max^T` and `m_V`, but the `m_V ~ C/√V` scaling is empirical, not derived here. Needs assumption block: packing regime, embedding geometry, and task-dependent margin constant. |
| 16 (T_depth ≥ D/C_step) | **NEEDS WORK** | Useful decomposition, but unproven lower-bound claim. `C_step` is undefined beyond heuristic regimes (serial vs parallel). Requires a concrete computational-graph or information-propagation proof. |
| 17 (T_min = max(T_readout, T_depth)) | **NEEDS WORK** | This is the weakest point. Exact decomposition is unlikely without an interaction term. Convergence and computation can interfere (especially with non-linear readout/state drift), so the equality should be `max(...)` upper/lower envelope with interaction corrections. |
| 18 (Compute-window laws) | **NEEDS WORK** | Practical window expressions are sensible but rely on the same decomposition assumptions and gloss over nonlinear breakdown/finite-precision effects. Need bounds for `T_degrade`, and explicit conditions under which `W_readout` and `W_margin` are valid. |

## 1) Logical soundness
- Precision generally improves from 9 onward, but the new documents contain multiple hidden assumptions left implicit:
  - existence and uniqueness of relevant fixed point(s),
  - validity region where first-order bounds hold,
  - smallness of remainder terms (`O(d0)`, `M` terms),
  - direct CE-per-horizon constraints vs expected CE over horizons,
  - well-defined basin geometry and margin regularity.
- “Hand-wavy zones” are concentrated in Theorems 10, 13, 16–18.

## 2) Consistency with existing framework
- Notation is mostly consistent (`J, rho, sigma_max, m, K, d0, T`).
- Cross-references mostly point to the right foundation blocks.
- No direct logical contradiction with Theorems 1–8, but some claims are stronger than the base assumptions (especially when they become “exact”/“strictly positive/negative” statements).

## 3) Claim calibration
- Suggested downgrades:
  - Theorem 9: STRONG → **MODERATE** (needs assumptions explicitly listed).
  - Theorem 10: STRONG → **WEAK** (directional, model-dependent).
  - Theorem 13: MODERATE → **WEAK/NEEDS WORK**.
  - Theorem 17–18: MODERATE/STRONG claims → **WEAK**.
- Suggestion upgrades:
  - Theorem 11 and the core of Theorem 14 are solidly rigorous enough for STRONG (with phrasing “direct gradient pathway”/“sufficient perturbation term” added).

## 4) Empirical testability (per theorem, with falsifiers)

- **9**: Vary training horizon set `{T}` vs `{4,6,8,10,12,14,16}` on same seeds; measure `σ_max` and per-horizon CE.  
  **Falsify if:** variable-T does not tighten `σ_max` beyond fixed-10 bound while still widening compute window.
- **10**: Keep architectures fixed, swap horizon sets (single `T=10` vs co-prime set), track Jacobian angles/imag parts and readout dip events.  
  **Falsify if:** multi-horizon has no effect on oscillatory modes once margins match.
- **11**: Compare CE-only gradient components via finite-diff probes around `s_T` and `s_T+ε`; verify zero dependence on perturbed states in direct term.  
  **Falsify if:** CE-only directly contributes non-zero terms from perturbed states under same training sample.
- **12**: Track `rho, M` and basin anisotropy under CE-only only.  
  **Falsify if:** CE-only enforces clear basin isotropy objective-like signal in gradients with no recovery term.
- **13**: Grid over `σ, K` and measure recovery for CE-only across many seeds.  
  **Falsify if:** consistent positive recovery appears away from accidental edge cases.
- **14**: Ablate `recovery_*` loss, hold everything else fixed.  
  **Falsify if:** perturbation loss yields no reliable positive recovery despite CE-only baseline being non-recovering.
- **15**: Fix `σ_max` (or match via targeted regularization), sweep `V` and measure minimal successful `T`.  
  **Falsify if:** `T_min` does not grow like `O(log V)` under controlled `σ_max`.
- **16**: For addition at varying lengths, measure minimal observed carry-chain completion steps.  
  **Falsify if:** no lower bound relation with task depth (including partial-credit serial depth).
- **17**: Fit `T_min(V,D)` surface and test additive separability residual:
  - model `T_min ≈ f(V)+g(D)`,
  - check for interaction term with varying `K_step`.  
  **Falsify if:** interaction term is statistically significant.
- **18**: Sweep `L` in D23 and measure actual width `W(D,L)`; compare shrink law under fixed max T range.  
  **Falsify if:** window width is not monotone in D as predicted.

## 5) Mathematical gaps
- Frequent “approximate” steps without bounds:
  - Theorems 9,10,15,16,17,18: no explicit constants or failure probabilities.
  - O-terms (`O(d0)`, `O(·)` constants) and neighborhood radii are not quantified.
- Weak claims that can be strengthened:
  - replace heuristic margins (`m_V ~ C/√V`) with empirical-fit theorem/lemma + confidence intervals,
  - replace “strictly smaller/better” with explicit conditional inequalities on training horizon distribution and per-horizon CE bounds,
  - add interaction bound for `T_min`: `T_min ≤ f(V)+g(D)+I(V,D)` with measured `I`.

## 6) Alternative explanations
- **Thm 9:** D22 gains could also come from horizon-robustness regularization (state trajectory mismatch training) rather than strictly spectral regularization.
- **Thm 13:** Recovery may happen through continuous basin overlap or secondary attractors without violating “sharp boundary” assumptions.
- **Thm 17:** Readout and depth are likely coupled; e.g., additional steps may both propagate carry and distort state margin, so `max` decomposition is not exact.

## 7) Cross-domain connections
- Nishimori (`rho=tanh(1/2)`) and IB at criticality are **suggestive**, not yet derivational for UESD.
- To formalize: define explicit mapping from UESD Jacobian statistics to a Gibbs/mean-field order parameter and prove fixed-point of update map implies `rho` near 0.4621 under stated scaling limits.
- Fisher-Rao language is currently conceptual; it needs a concrete theorem tying CE perturbation trajectory terms to a curvature term in parameter manifold for UESD.

## 8) Priority directive (single most important improvement)
Make hidden assumptions explicit at theorem level and convert every “always/strictly/exact” claim into a hypothesis+guarantee format with measurable constants (`ε`, neighborhood radius, horizon-wise CE bounds, `O(·)` constants, basin overlap/no-overlap assumptions). This single change would resolve most of the SOUND vs NEEDS WORK gap across 9–18.

**Overall assessment:**  
These are strong directions with a coherent architecture extension, but as written they are partially under-specified. Net rating: **mostly NEEDS WORK**, with **Theorem 11 (and the direct gradient argument in 14)** closest to fully sound; the remaining major claims need explicit conditions and measurable quantifiers before they are publication-grade.