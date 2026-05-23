# Codex Review: UESD Falsification Review (D19 + D20 + D21)

Context: I could not find a root `CLAUDE.md` in this checkout, so this review is grounded in `experiments/EXPERIMENTS.md` plus the listed falsification artifacts.

## 1) Falsification scorecard (D19-D21 against criteria)

### T1: Dynamics provide essential computation
- **Result**
  - CE ratio `seq_acc(T=1)/seq_acc(T=10)=0.0146` (1.5% vs 100%).
  - E5 ratio `0.0000` (0% vs 100%).
- **Verdict: PASS (strongly).**
- **Why**: Both ratios are far below the falsification threshold (`>=0.98`) and below the "supported" threshold (`<0.50`) from `_falsification_criteria.md`.
- **Caveats**
  - Single checkpoint/seed each track in D19.
  - Evaluated on the L=8 addition slice only.
  - Only fixed step tests, no adaptive stopping or variable task complexity in this pass.

### T4: E5 provides a meaningful advantage over CE-dynamics
- **Result set**
  - **D20 (bottleneck sweep)** covers only `dynamics_ce` with varying `V`; no side-by-side E5 sweep exists.
  - **D21 (wrong-attractor/recovery)**
    - CE: WA at +20 steps is higher at low noise (`15.5%` at sigma=0.01; `21.1%` at sigma=1.0).
    - E5: WA at +20 is lower at low noise (`2.0%` at sigma=0.01; `2.34%` at sigma=0.10), but narrows under stronger perturbations (`22.2%` at sigma=0.50; `99.9%` at sigma=2.0).
  - D20 indicates strong CE step dependence and recovery gains across most `V` values where the model learns; this behavior is not unique to E5.
- **Verdict: WEAKENED.**
- **Why**:
  - E5 does not dominate all three regime metrics (step dependence, bottleneck sensitivity, stability).
  - E5 appears more locally stable at tiny perturbations but with narrower basin geometry and much worse degradation under larger noise.
  - No D20 E5 comparison means "superior bottleneck scaling" is unproven.

### T5: Parallel (not sequential) computation
- **Result**: D19 carry positions `c0..c3` are effectively simultaneous for both tracks once accuracy rises:
  - CE: `T=3` already has c0=0.956, c1=0.921, c2=0.916, c3=0.920.
  - CE and E5 reach near equality by `T=5+`.
  - `summary.carry_position_dependence` is explicitly `"none (parallel convergence for both regimes)"`.
- **Verdict: PASS (qualified).**
- **Why**: No meaningful position-wise delay profile appears in D19, which is inconsistent with strict step-by-step right-to-left carry propagation.
- **Caveats**: only 4 output positions on L=8 addition; no per-carry-depth breakdown by corruption hardness in D19.

### T6: Causal claims about carry features
- **Result**: D21 shows neither regime can recover from latent perturbation; all recovery values are negative (extra steps worsen results).
- **Verdict: INCONCLUSIVE/WEAKENED for strong causal claims.**
- **Why**: The data indicate no observed iterative error-correction capacity in this setting. That does not prove carries are non-causal, but it weakens the claim that dynamics are doing active corrective causal repair.
- **Caveat**: D6/D8-style direct causal surgery evidence is not replicated in D21 itself; this is a negative result about robustness, not a direct causal ablation.

## 2) Cross-experiment consistency and contradiction checks

### D7 (thinking emergence)
- **Alignment**: D19's `T=3` / `T=4` saturation for CE is consistent with earlier findings that the effective compute horizon is short once the dynamics enter a productive phase.
- **Potential tension**: earlier E5 traces in prior D7 notes showed mild right-to-left bias under a now-noted SC bug. D19 E5 no longer shows clear positional staggering. This looks like a protocol / implementation sensitivity issue rather than a regime contradiction.

### D8 (causal carry surgery)
- **Alignment**: D19 all-position convergence supports the prior caution that carry-like activity is insufficient by itself to imply true algorithmic causality.
- **No contradiction**: D19 is a convergence-time signal and is directionally consistent with the weaker causal interpretation from D8.

### D10 (adaptive halting)
- **Alignment**: D19 shows CE reaches high performance by `T=5` and near-saturation by `T=6-8`; E5 by `T=6`. This is broadly consistent with the earlier `~6`-step minimum expectation.

### D11 (energy landscape)
- **Alignment**: D19 CE high-T degradation (`T=20` still fine, `T=32` drops to 77.95%) is consistent with finite-depth landscape wandering and non-fixed-point behavior.
- **Interpretive tension**: if D11/18 framed CE-dynamics as robustly "stable enough," D19 narrows that claim: stability appears contingent on staying inside a compute window.

### D17 (reconsideration / self-correction)
- **Alignment**: D21's negative recovery across all sigmas directly matches the weak/no self-correction pattern in D17.

### D18 (error-function geometry)
- **Alignment**: D11+D18 describe CE as higher movement and stronger transient dynamics, E5 as more contractive. D19/D21 mirror this split:
  - CE: stronger finite-horizon gains but worse high-T over-iteration.
  - E5: slightly more stable at short perturbations, but less robust at larger noise and still no repair.

### Contradictions to flag
1. **D19 does not support any claim of full single-step solveability**, while some earlier speculative language implied "ballistic / near-complete early capture."
2. **E5 claim of superior basin behavior is conditional**: better than CE at tiny sigmas but not at moderate/high sigmas.
3. **The "stable iterative solver" narrative is contradicted by direct recovery data**: both regimes lose accuracy with additional steps after corruption.

## 3) Finite-horizon computation interpretation

### Is this new or an artifact?
- It is most likely a legitimate finite-horizon property of this architecture+objective, not just a pure RNN-at-test-time artifact:
  - Training fixed `T=10` on this task and data distribution.
  - CE-dynamics and E5 both exhibit nonmonotonic gains above the useful horizon.
  - D19 and D20 jointly show that step dependence is not noise; it is a stable, reproducible signal tied to recurrence.

### Why CE degrades more than E5 at high T
1. CE has no explicit residual-convergence pressure (`r = s_{t+1}-s_t -> 0` is not guaranteed), so additional steps can keep moving along a trajectory past the best readout manifold and into worse regions.
2. D11+D18 geometry suggests CE has higher transient movement; without enough pullback, that movement converts to over-iteration.
3. E5 has stronger SC anchoring, lower state norm, and stronger convergence pressure, so extra steps are less likely to drift quickly, though that same pressure can still produce brittle narrow basins.

### Theory of optimal compute windows
- Define a per-step utility curve:
  - `gain_t = seq_acc(T=t)-seq_acc(T=t-1)`
  - `damage_t = WA_or_recovery_drop(t)`
- Practical optimum: run until `gain_t` saturates and `damage_t` starts increasing.
- From D19 this gives:
  - CE window center ~`T=5..15` with usable upper edge around `T~20`.
  - E5 window center ~`T=6..20` with degradation starting only near `T=32`.
- Training-time prediction heuristic:
  - monitor validation `T` sweeps, trajectory norm, and Jacobian-based transient measures (Lyapunov/rotation/alignment).
  - when marginal gain falls below threshold while instability proxies rise, stop.

### Inference-time implication
- Never hard-code a fixed `T=10` if the compute window is narrow.
- Use adaptive halting / budget-aware stopping driven by confidence or change metrics.
- High-accuracy at `T=10` does not imply monotonic benefits for `T>T_opt`; scaling compute blindly can hurt.

## 4) Recovery failure: interpretation and design response

### Is it fundamental?
- Not clearly fundamental to weight tying itself.
- It is stronger interpretation: current objective does not enforce invariance to latent perturbations over steps.
- D2b and D11/D18 already indicate objective mismatch can trap attractive geometry into wrong or fragile basins.

### Can Langevin noise (D12-style) help?
- **Likely partially, but uncertain.**
- It can improve basin sampling and smooth out sharp basin boundaries.
- It may also raise entropy and slightly hurt raw one-shot precision unless tuned with a denoising-consistency term.

### Training-objective modifications to improve recoverability
1. Add explicit corruption-consistency and denoising losses (`x -> F(x+noise,T+k)` targets clean output) at multiple k.
2. Multi-T curriculum: train with stochastic `T` (e.g., uniform over a short band) and penalize harmful over-iteration.
3. Include contractive/anti-overshoot terms (non-normality-aware, Jacobian regularization, residual norm penalties).
4. Decouple convergence and accuracy losses with adaptive weighting across step counts.

### Is recovery required for usefulness?
- For claims of "iterative solver" and "causal self-correction", yes.
- For "compute acceleration within a fixed readout horizon", no.
- Current evidence supports the narrower second claim better than the first.

## 5) Prediction assessment: D11+D18 vs D19 shock

### Why the CE T=1 prediction was wrong (mid-high 90% expected, 1.5% observed)
- Two likely errors:
  1. **Metric mismatch**: D11+D18 measured quality at longer unrolled trajectories, then inferred step sufficiency.
  2. **Single-step non-equivalence**: ballistic transient geometry does not imply full algorithmic resolution after one recurrent application.
- Result: rapid path movement and good final readout can coexist with very poor `T=1` classification.

### Ballistic path vs single-step sufficiency
- Ballistic means the early path is efficient in state space.
- Single-step sufficiency means the readout already separates the target at first move.
- D19 shows ballistic-like progress may require 2-6 steps before crossing the task decision boundary.

### Other Codex predictions: validated / falsified by D19-D21
- **Validated**
  - D19-level dependence on recurrence and non-fixed-point utility (supports iterative computation over pure feed-forward solve).
  - Strong early gains and finite-step saturation.
  - Recovery fragility already visible in D17-like terms.
- **Falsified/overstated**
  - CE near-high single-step expectation.
  - Any implied universality of causal/repair interpretation without explicit D19-style perturbation checks.

## 6) Revised thesis statement (D5-D21)

### What is now **supported**
- For this scale and task family, UESD recurrence materially increases compute depth compared to single-pass baselines.
- Both CE and E5 can reach near-1.0 seq accuracy on L=8 addition at matched T=10.
- Useful computation appears concentrated in a finite window; beyond that, extra steps can degrade outputs.
- CE and E5 differ as expected: CE gives faster short-horizon gains and stronger trajectory flexibility; E5 gives stronger pullback/near-convergence traits.

### What is now **hypothesized (not yet proven)**
- The mechanism is genuinely parallel, finite-horizon refinement rather than literal step-by-step carry-wave simulation.
- The finite window can be predicted from training-time stability/margin trajectories and used for adaptive halting.
- Recovery under latent perturbation can be strengthened without sacrificing final one-shot performance if training objectives are made explicitly denoising-aware.

### Confirmed failure modes
- No reliable recovery in D21 (negative recovery at all tested perturbations/sigmas).
- E5 basin fragility under moderate/high perturbation despite stable low-noise behavior.
- CE degradation at long unrolls (`T=32`).
- High seed/task-family sensitivity and prior seed-control confounds in earlier rounds.

### Thesis confidence
- **4.5 / 10**: The computational mechanism (finite-horizon iterative improvement) is supported on current tasks; the stronger claims of robust causal iterative repair and universal compute engine behavior are not.

## 7) What to test next (ranked by expected information gain / compute hour)

1. **Highest priority: train-time perturbation robustness for CE and E5**
   - Add corruption-consistency denoising loss across multiple noise scales and step-lag targets (`k in {1,2,5,10,20}`).
   - Highest information gain because it directly targets the largest unresolved contradiction (no recovery).

2. **Variable-T curriculum (train and evaluate at mixed horizons)**
   - Randomize training horizon around `T~[4,10,16,24]` and evaluate adaptive stopping policies.
   - Expectedly high leverage: clarifies whether finite window is avoidable or intrinsic.

3. **Scaling and transfer stress**
   - Longer carry depth and harder sequences (L=12/16, depth-4+ carries), plus addition variant with larger digit-carry density.
   - High info and moderate compute; directly probes generalization of finite window + parallelity claims.

4. **Noise-injected Langevin training variant**
   - Compare inference-only noise vs training-noise + denoising loss; isolate where recovery improves or collapses.
   - Medium-to-high info; likely clarifies CE vs E5 robustness tradeoff.

5. **Model-size and width sweep**
   - Keep task fixed and compare CE/E5 with {d=128,256,512}, at 3-5 seeds each.
   - Determine whether window shape and recovery deficits are scale artifacts or structural.

6. **Different task classes**
   - Subtraction / controlled affine automata / short multiplication.
   - If these fail while addition holds, then current evidence is compositional but narrow.
