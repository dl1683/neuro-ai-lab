**CRITICAL**

1. [exp_d37_alt_k_estimator.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d37_alt_k_estimator.py:478>) does not seed before model construction. D33 seeds before `make_model` at [exp_d33_crossover_probe.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:285>), then seeds again inside training. D37 only calls `full_seed(cfg["seed"])` inside `train_model` after weights already exist. So the data/T schedule is seeded, but initial weights are not D33-matched.

Fix: call `full_seed(cfg["seed"])` immediately before `make_model(cfg, device)`, or move seeding into `make_model`.

2. [exp_d37_alt_k_estimator.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d37_alt_k_estimator.py:314>) is not a full Jacobian estimator. It computes only the per-token diagonal block `d s_out[tok] / d s_in[tok]`, because it appends `grad[0, tok_idx]` and ignores cross-token derivatives. But `TransformerDecoderLayer` self-attention couples token positions, so the true state Jacobian is `(L*d) x (L*d)`, not `d x d` per token.

Autograd orientation is fine: `grad(s_out[..., out_dim], s_in)` gives a row of the Jacobian for that scalar output; stacking rows gives `J`, not `J^T`. `eigvals` is correct for that block. The issue is that the block is not the full system Jacobian.

Fix: for d=64/L=16, compute the full `1024 x 1024` Jacobian for one or a few samples, or rename this estimator to “per-token diagonal-block Jacobian” and do not call it ground truth.

**MODERATE**

1. The standard estimator matches D33’s measurement protocol. D37 uses seed `9999`, `EVAL_SAMPLES=4096`, `FP_T=100`, `TRAJECTORY_T=30`, and `stable_k = k_values[1:10]` at [exp_d37_alt_k_estimator.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d37_alt_k_estimator.py:167>), matching D33 lines 232-252. Only output rounding differs: D37 returns 6 decimals, D33 returns 4.

2. Random-direction estimator preserves `s_star_orig` correctly at [exp_d37_alt_k_estimator.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d37_alt_k_estimator.py:210>), and each direction starts from a fresh clone. `PERTURBATION_SCALE=0.01` is defensible: it is small relative to typical embedding/state scale and not near fp32 noise. But because LayerNorm and attention can make local scale behavior nontrivial, I would add a scale sweep, e.g. `1e-3`, `1e-2`, `3e-2`, and require stable estimates across scale.

3. Pairwise estimator is executable but not cleanly measuring the same local contraction rate. `UESDModel.dynamics_step` accepts arbitrary real tensors of shape `[B,L,d]` at [model.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:41>), so Gaussian states will run. But training always starts from learned positional `init_state` at [model.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:37>), and `randn * 0.1` may be off-manifold. This makes C more like a basin/global convergence stress test than an independent local k estimator.

Fix: add burn-in before measuring, or initialize `s1/s2` as `init_state + noise` with multiple noise scales. Report distance trajectories to confirm they enter the same basin.

4. D37 claims small models are from D36, but D36 used `SEQ_LEN=12`, `CARRY_DEPTH=6` at [exp_d36_architecture_sweep.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d36_architecture_sweep.py:56>). D37 small configs use `seq_len=16`, `carry_depth=8` at [exp_d37_alt_k_estimator.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d37_alt_k_estimator.py:74>). That may be intentional for identical D8 conditions, but it is not “D36 small” protocol.

**MINOR**

1. Full Jacobian memory risk is low because it recomputes each scalar output with `retain_graph=False`. Runtime is the bigger problem: d64/L16/batch8 means 8192 decoder forward/backward passes per model.

2. Resume safety is only key-based at [exp_d37_alt_k_estimator.py](</C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d37_alt_k_estimator.py:457>). Atomic save prevents most partial writes, but stale results with changed configs/constants will still be skipped. Add a run signature check before skipping.

3. `.claude/CLAUDE.md` was requested but does not exist in this checkout. The `.claude` directory exists, but contains only `worktrees` and `scheduled_tasks.lock`.

Recommended minimum fixes before running D37: seed before model creation, fix or relabel the Jacobian estimator, add perturbation scale sensitivity for B, and add a burn-in/on-manifold variant for C.