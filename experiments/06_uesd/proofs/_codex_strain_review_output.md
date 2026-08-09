No `CLAUDE.md` was found in this checkout (`neuro-ai-lab`), so I’m reviewing against the three requested files only.

## Findings (ordered by severity)

1. High: the new strain decomposition is under-identified, not falsified-by-construction.  
   - [bottleneck_depth_scaling.md:2477](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md:2477) defines  
     `ln(rho(D,T)) = f_complex(D) + g_strain(D/T)`, but with only 9 points there is no uniqueness constraint on the split into two unknown functions.  
   - In practice, `f_complex` is still tuned against the same data it is meant to explain.

2. High: internal consistency of `g_strain` is weak in this section.  
   - The section moves from a divergence form `g_strain(eta)=beta/(1-eta)^gamma` (with eta→1 blow-up) to a **peaked** `g_strain` with eta>1 attenuation without a new formal definition.  
   - See [bottleneck_depth_scaling.md:2525](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md:2525), [2611-2616](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md:2611).  
   - This looks like post hoc patching unless justified by a separate derivation.

3. Medium: the physical mechanism is only sketched, not testable as written.  
   - “Gradient coherence vs noise averaging” is stated qualitatively in [2539-2544](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md:2539) and [2605-2609](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md:2605), but no measurable criterion is defined (e.g., step-wise gradient alignment, signed Jacobian derivative, or per-sample FTLE sign structure).

4. Medium: D30 control interpretation is preliminary, not complete.  
   - The available D30 file has three completed configs, and shows `rho` rising with `T_min` at D=4: 0.9992, 1.0001, 1.0006.  
   - [exp_d30_tmin_control.json:95-99](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\results\exp_d30_tmin_control.json:95)  
     [176-178](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\results\exp_d30_tmin_control.json:176)  
     [256-258](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\results\exp_d30_tmin_control.json:256).  
   - `Config D/E` behavior is still pending in your listed experiment notes, so “confirms monotonicity” should be phrased as partial confirmation.

---

## 1) Mathematical rigor check

- The decomposition is **well-defined syntactically**, but not **identified** statistically from this dataset.
- `f_complex(D)` is borrowed from the old quadratic narrative and then hand-tuned to match D=2 and D=4 exactly.
- `g_strain(eta)` is currently heuristic: first divergent form, then peaked/attenuating form in a contradiction-like switch [2525-2527, 2611-2616](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md:2525).
- There is no theorem proving additivity of complexity + horizon effects; as written, this is phenomenological, not mechanistic.

## 2) Consistency against 9 D28 points + D30

- Existing D28 data match the new model qualitatively (described as explaining all 9 points), not strictly uniquely.
  - FT rho: 1.0018, 1.0026, 1.0024, 1.0016, 1.0042.  
    [exp_d28_contraction_ratio.json:554](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\results\exp_d28_contraction_ratio.json:554)
  - VT rho: 0.9994, 1.0001, 0.9996, 1.0030, plus D28 L=12/24 not yet complete in this run set [table lines 2685-2695](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\proofs\bottleneck_depth_scaling.md:2685).  
- It does not yet fully explain D30 because only 3 configs are complete in the JSON evidence [exp_d30_tmin_control].
- `theory_summary` already reclassifies Proposition 30 as revised/hypothesis and notes the D28 anomaly driving this replacement [theory_summary:589-633](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\proofs\theory_summary.md:589).

## 3) Prediction quality / discriminating power

- The 5 listed predictions are **potentially falsifiable** but broad:
  - especially the L=24 and config D/E ranges are wide enough to be explainable by many smooth alternatives.
- Alternative models (below) can likely pass all current predictions, so current predictions are not yet discriminating enough.

## 4) Alternative models that fit current data

- Alternative A (piecewise linear in eta):
  - `ln(rho)=a + b*D + c*max(0, D/T - c0)` plus a VT offset `-d/T_min` (or additive dataset-specific regularization term).
  - Fits the observed shape (rise at D=4, dip at D=8, jump at D=10; VT reversal around D=8) with no peaked singularity claim.
- Alternative B (phase-boundary model):
  - `rho = rho_complex(D) + s * I(D/T > 1) + epsilon`, where `I` is a smooth logistic transition.
  - Pure threshold interpretation (no peak) can already match the 9 points reasonably.
- To discriminate, you need data where:
  1) eta fixed, D varied (and vice versa),
  2) same training budget and seed, and
  3) unsolvable-at-T_min settings held separately from solvable ones.

## 5) Testing peaked strain (most novel claim)

To prove or disprove the peak:

1. Log per-step gradient contributions to spectral behavior:
   - decompose `||∂L/∂θ||` by backprop time index `t` and correlate with instantaneous contraction diagnostics.
2. Measure Jacobian trajectory products at fixed and random truncation points:
   - compare singular spectrum/FTLE signs at `T = D`, `T = T_min-1`, `T_train`, and `T_min+margin`.
3. Evaluate gradient alignment statistics across seeds/input minibatches at eta>1:
   - if “noise averaging” holds, expect low cosine alignment and high variance with no coherent increase in the relevant expanding subspace.
4. Add held-out eta-control (e.g., same eta with different D,T combinations) to reject confounding by task complexity alone.

If eta>1 does **not** show reduced coherence and does not reduce expansion directionality, the peak hypothesis fails.

## 6) Cross-domain analogues

- **Control/optimization**
  - Finite-horizon training/control tradeoffs resemble receding-horizon effects (early-step dominance changes with horizon mismatch).
  - Truncated backprop/BPTT bias–variance behavior naturally creates regime shifts near horizon saturation.
- **Physics / dynamical systems**
  - Criticality + horizon-limited forcing can produce peak sensitivity near matching scales.
  - Transient growth/non-normality can create large short-horizon anisotropic effects even when long-horizon asymptotics differ.

## 7) Single highest-priority validation/revision

Run an out-of-sample 2D sweep with controlled compute budget and at least 5 seeds to identify model class, not just parameters:

- Grid over `D in {6,8,10,12}` and `T_train in {8,10,12}` (FT) plus `T_min in {2,4,6,8}` (VT),
- Keep architecture, seed count, optimizer budget fixed,
- Collect:
  - rho, per-sample FTLE/step-time spectra, gradient alignment-by-time, and T99,
  - and replicate on an independent task family (not addition-only).

That one test is the fastest route to deciding whether peaked `g_strain(eta)` is real or an overfit to 9 points.