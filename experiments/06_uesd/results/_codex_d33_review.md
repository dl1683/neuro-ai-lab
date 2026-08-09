**BUGS FOUND**

1. **Paired design is analyzed as unpaired.**  
   D33 says seeds are matched across FT/VT, but [compute_crossover_analysis](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:341) uses `scipy_stats.ttest_ind(vt_rhos, ft_rhos)`. This should be `ttest_rel` on per-seed matched pairs: `rho_vt(seed) - rho_ft(seed)`. Current p-values are wrong for the stated design.

2. **The q/rho correlation does not test the claimed across-seed relationship.**  
   [Lines 360-365](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:360) correlate 5 depth-level mean deltas with 5 depth-level mean VT q values. The experiment claim says q should correlate with delta_rho “across seeds AND depths.” That requires per-depth, per-seed paired deltas, not depth means. Also the correlation result is printed but not saved.

3. **`measure_q_at_t_min()` measures eval-time T_MIN accuracy, not the theory’s “training-time coherent-gradient q.”**  
   The function itself is mechanically correct for heldout sequence accuracy at `T_MIN=4` ([lines 104-115](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:104)). But Prop 34 defines q as the cumulative fraction of T_min batches with coherent/solvable gradients during training; the proof explicitly says post-training eval q is insufficient and q must be training-time/cumulative ([proof lines 3200-3209](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:3200)). D33’s q is a proxy, not the claimed quantity.

4. **q measurement perturbs the training RNG stream.**  
   Every q checkpoint generates an extra 1024-sample batch ([line 109](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:109)), consuming the global torch RNG. This makes D33 not bit-comparable to D28/D32 training trajectories. It is reproducible within D33, but the measurement changes the experiment.

**WARNINGS**

- RNG is mostly controlled: model init is seeded before construction and training is reseeded after construction ([lines 273-277](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:273)). Matched FT/VT seeds should share initialization. However, `set_seed()` only sets torch seeds; deterministic CUDA flags are not enabled in shared training ([training.py:109](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/training.py:109)).
- Results writes are atomic for a single process: temp file then `replace()` ([lines 84-89](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:84)). They are **not safe for parallel multi-process execution** because every process would use the same `.tmp` path and same JSON checkpoint.
- GPU memory should be safe on a 32GB RTX 5090. The largest eval is 4096 x seq_len 20 with no grad, and spectral radius uses only 32 examples with grad. Single-run memory should be well below 32GB.
- Metric code matches D28/D32, but the “spectral radius” estimator is inherited and approximate. Good for comparability, not exact linear algebra.

**D28/D32 CONSISTENCY**

- Architecture matches: `d_model=128`, `heads=4`, `d_ff=512`, `n_enc_layers=2` ([D33 lines 49-53](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d33_crossover_probe.py:49); [D32 lines 52-56](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d32_multitask_mechanism.py:52)).
- Training hyperparameters match D28/D32: 20K steps, batch 256, lr `3e-4`, `TRAIN_T=10`, VT range `[4,6,8,10,12,14,16]`.
- Spectral radius and contraction summary follow D28/D32 patterns.
- Results format is weaker than D32: D33 config omits `batch_size`, `lr`, `fp_steps`, `trajectory_steps`, and per-run `params`/`run_signature`. Add these before launch.

**RECOMMENDATIONS**

- Replace unpaired t-tests with seed-matched paired deltas.
- Save `correlation_analysis` with Pearson/Spearman over per-seed paired rows: `{D, seed, delta_rho, q_vt, q_ft, delta_q}`.
- Use a local `torch.Generator` for q eval batches, or save/restore torch RNG state around `measure_q_at_t_min()` so measurement does not alter training.
- Rename `cumulative_q` to `mean_checkpoint_tmin_seq_acc`, or add a stronger q metric that records per-checkpoint T_MIN batch success without rounding and with sample counts.
- If parallelizing, shard by separate result files and merge after; do not run multiple writers against this JSON.

**VERDICT: CONDITIONAL PASS**

Training itself is likely safe and D28/D32-compatible. Do **not** launch unchanged if the paired p-values, q correlation, or Prop 34 conclusion will be used as evidence. The required fixes are small, but the current analysis layer would produce misleading statistics.

