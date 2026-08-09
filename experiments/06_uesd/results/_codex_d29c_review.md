Findings (severity-ranked)

- [CRITICAL] None identified.

- [HIGH] The “theory-correct” partition is implemented, but the projector `P_R` is still the 4-dim margin-critical subspace (one direction per output position), not the full readout Jacobian subspace `V_R(s*) = row_space(D_R(s*))` from Proposition 31.  
  - `compute_readout_subspace` builds directions from `d_critical` per half-position and stacks `half = SEQ_LEN//2` vectors into a rank-4 projector.  
  - [exp_d29c line 120](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:120), [exp_d29c line 170](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:170), [proof definition](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2577).  
  - Impact: this is a methodological mismatch with the theorem statement if strict interpretation is required.

- [MEDIUM] D29c’s “D29b comparison” branch is not a faithful reconstruction of D29b’s own procedure.  
  - D29c uses `ANALYSIS_T_VALUES = [1,2,3,5,8,10,15,20]` (drops `4`) whereas D29b uses `[1,2,3,4,5,8,10,15,20]`.  
  - D29c compares “D29b” via threshold `0.5` over all 1024 vectors, while D29b’s own thresholding was top-100 with 0.3 (readout-critical) / 0.7 (null) masks.  
  - [exp_d29c line 50](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:50), [exp_d29c line 311](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:311), [D29b line 47](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:47), [D29b line 263](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:263).  
  - Impact: cross-experiment conclusions (“vs D29b”) are not directly commensurate.

- [MEDIUM] Non-convergence handling diverges between branches used in the same experiment.  
  - Theory-correct aggregate supports use only converged samples, but D29b-style projected metric aggregates over all samples, including non-converged fixed-point cases.  
  - [exp_d29c line 416](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:416), [exp_d29c line 435](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:435).

- [LOW] FP convergence criterion is absolute rather than scale-aware on a 1024-dim state vector.  
  - `FP_TOL=1e-4` on `||s_{k+1}-s_k||` can be overly strict when state norm is O(100), and can trigger false non-convergence.  
  - [exp_d29c line 48](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:48), [exp_d29c line 219](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:219).

Direct answers to your 5 checks

1. Mathematical correctness
- `D_norm = (I - h_hat h_hat^T)/||h||` is correctly coded in `[exp_d29c line 155-156](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:155)`.
- Chain rule form `W_R^T @ D_norm^T @ delta_e` is correct for margin wrt state in this model (up to constant `1/tau`, which cancels due per-direction normalization). See `[model readout](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:56)` and `[exp_d29c line 170](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:170)`.
- The alignment partition formula against all right singular vectors using `||P_R v_i||/||v_i||>0.5` is implemented as written. See `[exp_d29c line 286-299](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29c_ftle_theory_correct.py:286)` and the Prop31 spec `[bottleneck_depth_scaling.md:2581-2582](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2581)`.

2. Implementation bugs (dimensions/Jacobian/SVD/loops)
- No obvious tensor-shape or index-off-by-one errors in jacobian/SVD/update loop were found.
- Jacobian multiplication order is consistent for composed transition (`Phi = J_t @ Phi` with `T=1..max_T`).
- `torch.linalg.svd(Phi)` then `Vh[i]` as right singular vectors is consistent with the implemented alignment logic.

3. Fixed-point convergence
- 500 steps is a reasonable cap for screening but not a guarantee; with your reported O(100) state norms, absolute `1e-4` is extremely tight (`~1e-6` relative).  
- Prefer relative stopping: `||Δs|| / (||s|| + ε)` and optional `Δ²`-improvement check; this is more robust than only absolute residual.
- Anderson acceleration could accelerate convergence if this is truly a fixed-point solve, but it changes the solver dynamics and should be reported as a solver variant rather than “same as dynamics-at-tractor” procedure.

4. Comparison integrity
- Not fully intact due non-identical T grid, different subspace/threshold conventions, and mixed converged-vs-all sample aggregation as noted above (MEDIUM issues).

5. Resource estimate
- Memory: should fit on 24GB. Core dense objects: `J`, `Phi`, `P_R`, `P_R_29b` are each ~4 MB at 1024×1024 fp32; peak is dominated by autograd graph in each Jacobian row.
- Compute: 8 samples × 20 steps × 1024 outputs = 163,840 scalar backprop calls for Jacobian rows; plus 64 SVDs of 1024×1024.
- Expected wall time is dominated by Jacobian loop; practical range is broad (few minutes to tens of minutes depending on cuDNN kernel/runtime), likely manageable but expensive; not a memory-bound case.

