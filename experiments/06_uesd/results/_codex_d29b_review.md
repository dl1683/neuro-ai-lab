1) Is `W_R.T @ (e_correct - e_second)` correct for margin-critical direction?  
**Mostly no (approximate only).**  
It gives the steepest direction for the **unnormalized** readout map `W_R @ s`, not the actual normalized readout used by the model. In this codebase `readout_logits` normalizes `h = readout_proj(s)` before dotting with token embeddings, so the true first-order margin direction should include the normalization Jacobian factor:
`J_margin ∝ W_R^T @ D_norm(h_l) @ (e_correct - e_second)` (and sign optional), with `D_norm(h)= (I - \hat h \hat h^T)/||h||`.  
You currently use the simpler form at [exp_d29b_ftle_corrected.py:150](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:150), so it is not exact but a proxy direction.  
See contrast in D29 state-dependent readout Jacobian logic at [exp_d29_ftle_decomposition.py:185](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29_ftle_decomposition.py:185).

2) Is `P_crit @ Phi @ P_crit` correct for restricted FTLE, or should it be `Phi @ P_crit`?  
**Depends on what “restricted” means; currently this is a stricter projection than needed.**  
- For FTLE of directions that start in the critical subspace (standard “restricted subspace growth”), you typically want `Phi @ P_crit` (or `Q^T Phi Q` with basis `Q`).  
- `P_crit @ Phi @ P_crit` enforces projection both at input and output and measures in-subspace component of the evolved vector, which can understate growth and is a different quantity.  
Current usage [exp_d29b_ftle_corrected.py:284](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:284) is therefore **not equivalent** to the usual subspace-restriction interpretation.

3) Is readout null-space computation correct?  
**Partially, but incomplete relative to the model’s actual readout map.**  
You compute null of `E_norm @ W_R` [exp_d29b_ftle_corrected.py:179-184](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:179).  
But because readout normalizes `h`, the exact Jacobian wrt state is `E_norm @ D_norm(h_l) @ W_R / tau` (position-wise), so true zero-readout directions should be based on that position-dependent Jacobian. D29 had this form in `compute_readout_projection_at_state` [exp_d29_ftle_decomposition.py:185-190](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29_ftle_decomposition.py:185).

4) Bugs / numerical issues / edge cases  
- **High:** target token used for critical direction is dataset target, not necessarily current argmax class. If sample is mispredicted at `s*`, the “decision-boundary threat” direction is inconsistent; better derive from current argmax vs 2nd-best on logits [exp_d29b_ftle_corrected.py:141-148](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:141).  
- **Medium:** `P_crit`/`P_null` are block-constant and independent of time, while `J_t` changes over trajectory; if trajectory is not exactly at fixed manifold, decomposition should be state-dependent through time for strict validity.  
- **Medium:** only top-100 singular vectors are screened for alignment [exp_d29b_ftle_corrected.py:257-267](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:257), which can miss aligned directions outside this band.  
- **Medium:** rank threshold uses relative cutoff `S > S[0]*1e-6` [exp_d29b_ftle_corrected.py:182](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:182), which is brittle when spectrum scales are tiny.  
- **Low:** hard-coded `SEQ_LEN` inside helper functions [exp_d29b_ftle_corrected.py:126](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:126), [169](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:173) reduces reuse and can silently break on different sequence settings.

5) Is the methodology sound for testing Proposition 31?  
**Partially.** The conceptual split into “margin-critical” vs “readout-null” is much better than D29’s thresholded row-space partition, but in current form it is not a fully faithful FTLE test of Proposition 31 because it omits the readout normalization Jacobian and does not consistently define the same notion of subspace restriction in both direct-FTLE and alignment metrics.  
If you keep those two fixes (state-dependent null-space/critical-direction derivation + consistent projector placement), the framework becomes much more defensible.

If you want, I can provide a minimal patch that aligns all three of these precisely (critical direction, null space, and direct restricted FTLE) while preserving your existing file structure.