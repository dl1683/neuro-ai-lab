I found no CRITICAL launch blockers, but I would not call this a clean PASS. There are several MODERATE issues worth fixing before a 5.5h GPU run.

**Findings**

MODERATE: The reported “spectral radius” is not a full-state spectral radius estimate. In [exp_d36_architecture_sweep.py](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:213>), `torch.autograd.grad(..., grad_outputs=v)` computes a VJP, effectively `J^T v`, not a JVP. That is not automatically wrong for eigenvalue growth, since `J` and `J^T` share eigenvalues, but the estimator then normalizes with `norm(dim=-1)` and averages per token in [line 218](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:218>) and [line 221](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:221>). That is not the spectral radius of the flattened `(seq_len * d_model)` dynamics operator. It matches D33’s protocol, so D36 is comparable to D33, but the metric label is stronger than what is actually measured.

MODERATE: Architecture robustness verdict uses seed-level VT `k` range, not architecture-level means. In [lines 383-384](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:383>) and [421-439](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:421>), `vt_k_range = max(all_vt_k) - min(all_vt_k)` mixes architecture effects with seed noise. For an architecture sweep, the primary verdict should use `mean_vt_k` per architecture, with seed-level spread reported separately.

MODERATE: `T_99=None` is conflated with `T_99=30` in analysis. [Lines 379-380](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:379>) replace missing convergence with `30`, but an actual first hit at step 30 has the same value. This biases averages downward for censored runs; use `TRAJECTORY_T + 1`, keep `None`, or report censored counts separately.

MODERATE: Resume is run-granular only and does not validate config compatibility. [Lines 454-459](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:454>) skip any existing run with the same key. If constants/configs changed after an earlier partial run, stale results will be silently reused. It resumes correctly after interruption between runs, but interruption mid-run loses that run entirely.

MINOR: Parameter count comments are slightly low. Actual instantiated counts are:
`small`: 183,168, not ~175K  
`baseline`: 702,208, not ~694K  
`large`: 2,747,904, close to ~2.7M  
`many_heads`: 702,208  
`few_heads`: 702,208  
All five configs are valid: every `d_model` is divisible by `n_heads`.

MINOR: `best_acc`/`phase_transition_step` use the same batch just trained on at checkpoint time, not a held-out batch. See [lines 167-171](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:167>). This is the D33 pattern, but the `accuracy` field should not be treated as an unbiased evaluation metric.

MINOR: `scipy_stats` is imported but unused in D36 at [line 40](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:40>).

**Checklist Status**

RNG isolation around `q`: PASS for normal single-GPU training. It saves/restores Python, NumPy, CPU torch RNG, and current CUDA RNG in [lines 110-129](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:110>). Multi-GPU “all CUDA states” is not covered, but this script uses one `cuda` device.

Training loop: PASS. Variable-T selection, addition half-sequence CE, optimizer step, and grad clipping are correct.

Contraction `k`: PASS relative to D33. It uses `stable_k = k_values[1:10]` exactly as D33 in [line 253](</mnt/c/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:253>).

`T_99` measurement: PASS at run level; analysis handling is the issue.

`compute_architecture_analysis`: Mostly correct pairing by `(arch, seed)`, but the global verdict should use architecture means and it should preserve paired rows / censored T99 status.

Recommendation before launch: fix the analysis range calculation and censored `T_99` handling at minimum. If the rho number will be interpreted literally, rename it or replace the estimator with a flattened-vector power iteration.