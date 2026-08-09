I read the relevant repo files. `CLAUDE.md` is not present in `neuro-ai-lab`; I only found sibling-project copies, so I did not treat them as binding.

**Verdict**

The evidence now supports a narrower, stronger claim: VT training regularizes finite-time trajectory geometry, primarily through `k` suppression, while `rho` is a secondary ceiling/stability observable. The old rho-first story should be demoted.

High confidence:

- D31 + D33 confirm `k(VT) < k(FT)` on learned addition dynamics: D8 `dk=-0.0023`, `p=0.000017`, 8/8 seeds; D6 `dk=-0.0032`, `p=0.012`, 4/4 seeds.
- VT accelerates convergence: D33 D6 VT `T_99=3` for all seeds vs FT `[4,6]`; D31 D8 gives ~47% faster convergence excluding the undertrained seed.
- `rho` suppression is real in some settings but not the primary mechanism: D6 `p=0.173`, D8 `p=0.083`; D34 shows a strong VT rho ceiling over time, but with identical bad multitask accuracy.
- Learning signal matters. D35 unlearned prefix sum mostly shows no `k` suppression; seed 256 shows `rho=1.0029` at 0% exact accuracy but lower loss, meaning `rho` tracks distributional learning / loss reduction more than exact-match success.

Uncertain:

- Whether `k` suppression generalizes beyond addition.
- Whether `k` suppression weakens monotonically with depth. D6/D8/D10 suggest yes, but D28 D12 still has lower VT `k` in the single-seed table. What vanished at D12 is clearly `rho` suppression, not yet proven `k` suppression.
- Whether the mechanism is readout-critical FTLE contraction specifically, versus a broader training-schedule/regularization effect that happens to lower measured `k`.

**Theory Update**

Propositions 30-35 need consolidation.

- Prop 30 should stop being a rho law. Keep it as “complexity affects criticality,” but `rho(D)` is too volatile and underidentified.
- Prop 31/33 remain structurally important: anisotropic readout convergence explains why `rho > 1` can coexist with correct readout.
- Prop 32 should be rewritten around finite-time convergence pressure and `k`, not `rho`.
- Prop 34’s gradient-coherence model should be reformulated as `Δk ≈ f(q, T_min, learning_progress)`. The old `Δrho ≈ -A*q` version is too weak.
- Prop 35 should become the main VT proposition: VT reduces average/trajectory contraction rate on learned finite-time dynamics; `rho` is only the max-direction witness.

Predictions for D7-D10:

| Depth | Predicted `Δk` | Prediction |
|---|---:|---|
| D7 | about `-0.0027` to `-0.0030` | strong, significant if 4 seeds clean |
| D8 | about `-0.0023` | already confirmed |
| D9 | about `-0.0018` to `-0.0021` | should persist but weaker |
| D10 | about `-0.0014` to `-0.0017` | matches D31 control direction |

I would not predict `rho` significance at D7-D10. I would predict VT `T_99` remains around 2-4 and FT remains around 4-6 unless training fails.

**D35**

Yes, run D35b with learnable prefix sum, probably `V=8`, and make it explicitly a learning-matched generalization test.

The current `V=64` prefix sum is mostly not testing “does VT generalize?” It is testing “what happens when the task does not learn?” Pair 3 is important: `rho=1.0029`, loss `2.09`, exact acc `0%`. That says exact accuracy is too coarse. The model can enter supercritical/tighter dynamics through partial distributional learning before sequence accuracy moves.

D35b should report loss, per-position accuracy, token accuracy, exact sequence accuracy, `k`, `rho`, and `T_99`. The key comparison is not raw VT vs FT; it is VT vs FT at matched learning progress.

**Confidence**

Overall theory confidence: **7/10 is a little high if it refers to UESD broadly; 6.5/10 is cleaner.** The VT `k` mechanism itself is around **8/10 inside addition-like learned dynamics**.

- T1, weight-tied dynamics converge to useful fixed/readout-stable points: **8/10**. Strong evidence across addition and variable-T convergence, but “fixed points” is too strong because many states are readout-stable manifolds, not true fixed points.
- T4, error functions guide dynamics to task-relevant attractors: **4/10**. Better than 3.5, but still weak. The dynamics work; the specific “error function creates attractor geometry” claim is not isolated from architecture, CE readout, and training schedule.
- T5, VT training creates contractive dynamics: **8.5/10** if stated as `k`/finite-time contraction, **6/10** if stated as spectral-radius contraction. D31/D33 are strong; D35 warns that learning dependence matters.
- T6, thinking-generating continuum exists: **2.5/10**. The theory file’s continuum claim remains mostly architectural/philosophical. Current experiments show iterative convergence and readout-stable trajectories, not a demonstrated continuum between reasoning and generation.

**Next Steps Ranked**

1. **Finish D33 D7-D10.** Highest value now. It tests the depth law directly and should determine whether `Δk` decays smoothly.
2. **Launch D35b with `V=8`.** Needed to separate “addition artifact” from “learned finite-time dynamics mechanism.”
3. **Add a falsification probe: matched-loss checkpoints.** Compare FT/VT at equal loss/per-position accuracy. If `Δk` disappears when learning progress is matched, the mechanism is partly just faster/slower optimization, not VT geometry.
4. Larger replication later. D31+D33 already make addition replication less urgent than cross-task and matched-learning tests.

**Strongest Argument Against k-Mechanism**

`k` may be an outcome metric, not the mechanism. VT may simply train models that reach good readout earlier because the loss directly supervises shorter horizons; lower measured `k` could be a downstream signature of shorter-horizon optimization, not a causal dynamical principle.

Falsifier: train FT and VT to matched loss, matched per-position accuracy, matched `T_min` performance, then measure `k`. If `Δk` vanishes under matched learning progress, Prop 35 becomes an optimization-progress artifact. If VT still has lower `k`, the geometry claim survives.

What is missing: causal intervention. Perturb `k`-relevant directions or regularize FT to match VT `k`; then test whether `T_99` and robustness move accordingly.

**Parsimony**

Yes, the framework is overcomplicated. Collapse it into four claims:

1. Readout-stable iterative dynamics can solve compact compositional tasks.
2. Stability is anisotropic: readout-critical directions contract while other directions may remain neutral/expansive.
3. VT regularizes finite-time solver geometry, primarily visible as lower `k`.
4. Generality is unproven until non-addition learned tasks pass.

Drop or appendix-park rho-centric depth laws, Nishimori-heavy framing, and broad thinking-generating continuum claims until they produce discriminative predictions.

