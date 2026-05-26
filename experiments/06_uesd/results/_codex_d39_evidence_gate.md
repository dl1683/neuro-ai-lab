Verdict: **WARNING: D39 supports a useful narrow finding, but several claimed conclusions are overstated. Evidence quality is moderate for “higher SC lowers residual while preserving readout accuracy,” weak for “wrong attractors eliminated,” weak for “k is architecture-determined,” and not publication-ready without follow-up controls.** `CLAUDE.md` was not present at the requested path.

**Findings**

**CRITICAL** Wrong-attractor claims are vacuous under the D39 evaluator.  
In [exp_d39_convergence_sweep.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d39_convergence_sweep.py:335), an attractor is only counted when `norm_r < 0.01`. All 12 runs have `converged_frac=0.0`, and [line 357](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d39_convergence_sweep.py:357) reports `wrong_attractor_rate=0.0` when no examples converged. So D39 proves “no wrong converged attractors were observed because no attractors were observed,” not “CE warmstart eliminates wrong attractors.”

**WARNING** The central residual finding is supported.  
Final residual decreases monotonically with lambda: `0.094 ± 0.012`, `0.068 ± 0.012`, `0.046 ± 0.011` for `lambda_sc=0.1/0.3/1.0`. That is a real, replicated effect across 4 seeds per lambda. Accuracy remains high: final seq acc ranges `99.93%` to `100%`.

**WARNING** The k effect is small but not zero, and the architecture claim is undercontrolled.  
Mean k moves from `0.9647` to `0.9584`; seed 512 reaches `0.9478`. That supports “SC has much stronger effect on residual than k,” but not “k is architecture-determined.” Alternative explanations remain: limited lambda range, training schedule, dropout/noise, optimizer basin, flow-loss interference, spectral norm parametrization, and the specific k estimator in [measure_k](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d39_convergence_sweep.py:270). Required control: same architecture with `lambda_sc=0`, higher `lambda_sc`, no flow, no recovery, and ideally a different architecture width/depth.

**WARNING** Flow correction is empirically broken, but the root-cause diagnosis is not proven.  
D39 strongly confirms the symptom: final `flow_K4_seq_acc=0.0` and `flow_K8_seq_acc=0.0` in all 12 runs. But “fundamentally broken due to space misalignment + SNR” is still an interpretation. The code trains `flow_loss_matched` from `h` toward token embeddings at [line 156](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d39_convergence_sweep.py:156), and inference applies `flow_correct` from `h` at [shared/model.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:246). Plausible alternatives include velocity sign/scale, target embedding/readout geometry mismatch, or flow overcorrection of already-correct states.

**WARNING** Predicted extended-T convergence is a model-based extrapolation, not a result.  
Using `residual * k^n < 0.01`, D39 predicts additional steps from about `21` to `72`, or total T from about `31` to `82` after T=10. This justifies D40, but should be labeled as extrapolation because k is measured by a separate estimator against a 50-step reference state, not by directly evaluating residual decay at T=`25/50/100`.

**WARNING** Statistical robustness is smoke-level, not publication-grade.  
Four seeds per lambda are enough to detect the monotonic residual trend, but too few for reliability claims. For 4/4 successes at a lambda, the lower confidence bound on true seed-level success remains poor. Also seed 512 is consistently best, suggesting initialization effects are material.

**INFO** No obvious train/eval data leakage.  
The task generator creates fresh random synthetic addition batches in [data.py](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/data.py:22); evaluation uses newly generated samples, not a reused held-out file. This is fine for synthetic generalization, though it does not test out-of-distribution sequence lengths, bases, carry-chain stratification, or adversarial carry cases.

**INFO** Methodology is mostly coherent but attribution is confounded.  
Phase A CE alone already gets `99.51%` to `99.95%` seq accuracy before SC. D39 cannot attribute final accuracy to the four-phase stack. The clean claim is: “Starting from a strong CE warmstart, adding margin-gated SC reduces residual while preserving accuracy.” It does not isolate flow, margin, SC, and recovery contributions.

**Required follow-ups before publication**

1. Direct extended-T evaluation: T=`10/25/50/100/200`, report residual, accuracy, and wrong converged attractors.
2. Ablations: no-flow, `lambda_sc=0`, CE+SC only, CE+margin only, CE+recovery only.
3. More seeds: at least 10 per setting for reliability claims.
4. Architecture controls: vary width/depth or dynamics block to test the “architecture-determined k” claim.
5. Flow diagnosis: compare flow target spaces and add acceptance-gated/no-op flow baseline.
6. Report WA only alongside nonzero convergence; avoid claiming WA solved when `converged_frac=0`.

Overall evidence quality: **Moderate for residual control, weak for attractor/convergence claims, weak-to-moderate for flow failure, insufficient for architecture-level k claims.**