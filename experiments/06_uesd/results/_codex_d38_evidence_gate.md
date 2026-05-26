**Evidence Gate Verdicts**

1. **“Phased CE warm-start eliminates wrong attractors” — CONTESTED**
   
   D38 shows `h_seq_acc=1.0` for all 4 seeds, including seed 512. That validates robust correct **T=10 readout behavior**.

   But `wrong_attractor_rate=0.0` is not strong evidence because `converged_frac=0.0` for every seed. In code, WA is counted only among examples with `norm_r < 0.01`; when there are zero converged examples, WA is reported as `0.0`. So this is partly vacuous: D38 has no measured attractors, wrong or correct.

   Stronger phrasing: **CE warm-start avoids the E5 seed-collapse failure and produces correct non-converged endpoints. It does not yet prove wrong attractors were eliminated.**

2. **Comparison to 20% WA baseline — QUALIFIED**

   The D2b baseline is relevant: same addition family, same seed 512 appears as the E5 failure case, and D38 succeeds on seed 512. But the comparison is not apples-to-apples:
   
   - D2b E5 had SC-driven convergence to wrong fixed points.
   - D38 has 0% convergence, so it may avoid wrong attractors by not forming accepted fixed points.
   - D38 has larger architecture due to the flow head, longer/phased training, VT range `[4,16]`, margin/recovery objectives, and result-only CE fixes.

   Fair claim: **D38 avoids the observed D2b E5 failure mode on these seeds.**  
   Unfair claim: **D38 proves the wrong-attractor problem is solved.**

3. **Flow distribution mismatch diagnosis — VALIDATED, with caveat**

   The code confirms the mismatch. Training uses:

   `z_t = (1 - t) * y_embed + t * eps`

   so `t=1` corresponds to Gaussian-like noise. Inference starts with:

   `z = h.clone()`

   and then calls the flow head at `t=1, 0.75, ...`.

   So the first inference state is `h` while the model was trained to expect noise-like `z` at high `t`. That is a real distribution mismatch and the collapse of flow-corrected accuracy after phases C/D is consistent with it.

   Caveat: the results do not prove mismatch is the only cause. Other possibilities include flow overcorrection, velocity scale/sign sensitivity, conditioning mismatch, or the flow head learning a target-embedding projector that destroys already-correct `h`.

4. **“SC lambda=0.02 is too weak for convergence” — QUALIFIED**

   Metrics support “insufficient under this setup”: residual drops from roughly `0.28-0.37` in Phase A to `0.11-0.14` in Phase D, but `converged_frac` remains `0.0` against threshold `0.01`.

   But “lambda too weak” is not uniquely identified. Confounders include competition with CE/flow/recovery, flow loss dominating total loss, threshold strictness, insufficient SC duration, and SC being applied to transient endpoints from variable `T`. Stronger lambda is plausible, not proven as the only fix.

5. **CE warm-start confound — CONTESTED against phase attribution**

   CE alone nearly explains the main positive result. Phase A already reaches:
   
   - seed 42: `h_seq_acc=0.9951`
   - seed 137: `0.9995`
   - seed 256: `0.9976`
   - seed 512: `0.9968`

   By Phase B/C/D this becomes 1.0. Without an ablation such as Phase A longer, CE+SC only, CE+margin only, or CE+recovery only, the claim that the whole 4-phase blueprint caused success is over-attributed.

6. **Proposed flow fix: train on `z_t = (1-t)*y_embed + t*h` — QUALIFIED**

   This is theoretically sound for the actual intended use: correcting from UESD output `h` toward `y_embed`. It makes the endpoint distribution match inference better than noise interpolation.

   But implementation detail matters. If `h` is already highly accurate, a correction flow can easily hurt unless trained with identity/no-op behavior near correct states or evaluated with acceptance gating. A better version may be `h + sigma*eps -> y_embed`, or a residual denoiser with a reject/accept rule based on margin and readout stability.

7. **Statistical rigor — QUALIFIED/WEAK**

   Four seeds is useful smoke evidence, not a robustness claim. For 4/4 seed-level successes, the exact 95% lower confidence bound on true success rate is only about `40%` by two-sided Clopper-Pearson, or about `47%` one-sided. Equivalently, observing 0/4 failures still leaves a large upper bound on true failure probability.

   Also, the 4096 eval samples per seed strongly support per-run `h_seq_acc`, but not seed-level reliability.

8. **k/residual consistency with D33/D37 — QUALIFIED**

   D38’s `k≈0.967-0.969` is lower than the D33/D37 VT setpoint around `0.988`, so it is not “consistent” in the narrow numeric sense. It is plausible because D38 uses additional SC pressure and a different objective stack. The residual range `0.11-0.14` is consistent with prior warnings that these systems can be correct at readout while not fixed-point converged.

**Bottom Line**

D38 validates a narrower result: **CE-first phased training reaches perfect transient readout accuracy across 4 seeds and avoids the observed E5 seed-collapse case.**  

It does **not** validate “wrong attractors eliminated” as an attractor-level claim, because no examples converge under the experiment’s own threshold. Flow mismatch is real and likely important, but the proposed fix still needs an ablation.