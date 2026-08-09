**Findings**

Critical, blocks launch: intermediate dynamics checkpoints perturb the training RNG. [measure_contraction_summary](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:165) and [measure_spectral_radius](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:126) call `full_seed(9999)` without saving/restoring RNG state. Since these run inside training at steps 5000/10000/15000/20000, subsequent training data and `variable_t` choices are reset to the same stream. `measure_q_at_t_min` does isolate RNG state, but it restores to the already-corrupted state after those two helpers. Fix before launch.

Moderate, blocks matched-loss claims: matched-loss analysis uses stochastic current minibatch training loss, not a standardized evaluation loss. At [lines 262-278](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:262), checkpoint `"loss"` is either the logged training loss or current batch loss; for VT it is also loss at the sampled random `t_steps`, while FT is always `T=10`. Then [lines 433-438](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:433) nearest-matches those noisy, non-common losses. Use a fixed eval batch and common `T` for both variants.

Moderate: reported `"accuracy"` is `best_acc` on training minibatches sampled only at log checkpoints, not final held-out accuracy. See [lines 244-249](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:244) and storage at [line 329](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:329). This can overstate learnability. Add final held-out seq/token/per-position accuracy at `TRAIN_T`, and label T-min metrics as q metrics.

Minor: `token_accuracy_final` and `per_position_accuracy` are measured at `T_MIN=4`, not final/train `T=10` accuracy. The computation itself is correct for q-at-T-min, but the names are misleading. See [lines 315](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:315) and [330-331](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:330).

Minor: resume logic skips runs by key only and does not validate config compatibility. A stale `exp_d35b_learnable_prefix_sum.json` with changed constants would be silently reused. See [lines 448-452](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d35b_learnable_prefix_sum.py:448).

**Checklist**

1. `generate_batch("prefix_sum", ..., VOCAB_SIZE=8)` is supported and vocab-parametric. No hardcoded 64 in the generator. See [data.py lines 92-123](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/data.py:92).
2. `measure_q_at_t_min` mostly isolates RNG correctly for CPU, Python, NumPy, and current CUDA device. It is not `try/finally`, and multi-GPU CUDA state is incomplete, but single-GPU launch is probably fine.
3. `measure_spectral_radius` uses `no_grad` for fixed-point iteration and `enable_grad` for the autograd call correctly. Note: the method computes a VJP-style power iteration (`J^T v`), inherited from D33/D35, so call it comparable to prior experiments but not a rigorous spectral radius estimator.
4. No. Dynamics checkpoints perturb training RNG. Launch blocker.
5. No. Matched-loss analysis is not valid enough for claims. Launch blocker for that analysis.
6. Core training loss over all prefix positions and gradient clipping are correct.
7. Likely CUDA-safe for single GPU; I see no obvious device mismatch.
8. No material off-by-one issue. q checkpoints happen after the optimizer step at step 1/2000/etc.; dynamics checkpoints after steps 5000/etc.; T99 loop labels one dynamics step as `t=1`, which is consistent.
9. Resume works mechanically, but lacks config validation.
10. Per-position accuracy computation is correct for the fixed q-at-`T_MIN` eval batch, but not for final `TRAIN_T` accuracy.

Bottom line: do not launch as-is. Fix RNG isolation around all measurement helpers first, then replace matched-loss inputs with common fixed-eval losses. 

