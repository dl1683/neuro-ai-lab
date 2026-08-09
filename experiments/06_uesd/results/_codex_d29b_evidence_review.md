Short verdict: **not yet a clean confirmation**.  
D29b shows a strong and direction-consistent empirical pattern, but several core FTLE-construction steps are still not faithful to the Proposition 31 setup, so the claim is better treated as *provisionally supportive* than confirmed.

1) **Methodology assessment**

- The margin-critical idea is conceptually better than D29’s 512/1024 readout-row partition, but implementation is incomplete.  
- [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:129) now builds a low-rank margin-critical subspace (`dim_crit=4`), matching the intended dimension reduction.  
- It does **not** include the readout-normalization Jacobian in the critical direction; this is exactly the prior concern. In `readout_logits`, `h` is normalized (`F.normalize`) before dot with token embeddings, so the exact margin derivative should include `D_norm(h)` [shared/model.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:56).  
- The current restricted-FTLE computation uses `P_crit @ Phi @ P_crit` and takes the top singular of that product [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:284). This enforces in-subspace projection at both input and output and is stricter than the usual “restricted to subspace” quantity (`Phi @ P` or `Qᵀ Φ Q`).  
- Prop. 31 definition in the proof uses readout-relevant subspace at manifold point and thresholded alignment on right singular vectors (`a_i>0.5`) [bottleneck_depth_scaling.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2578). D29b’s proxy is closer to that intent but not equivalent in this exact form.

2) **Statistical validity**

- Provided aggregates: at T=5, `lambda_R = -0.004306 ± 0.002908`, `lambda_null = 0.04031 ± 0.016006`, `n=8` [exp_d29b_ftle_corrected.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d29b_ftle_corrected.json:4312).  
- All 72 values (8 samples × 9 Ts) are `lambda_R<0`; all 72 `lambda_null>0` in the JSON; this gives a sign-based one-sided p-value of about `2^-72 ≈ 2.4e-22` under i.i.d. sign null (very conservative, strong evidence).  
- Across each T (8/8 negatives), one-sided sign p ≈ `0.5^8 = 0.0039`.  
- One-sample t-stat for T=5 is about `t ≈ -4.19` (df=7), so `p < 0.003` one-sided for `mean(lambda_R)<0` (approx).  
- Caveat: repeated T values within the same sample are correlated, so this inflates naive effective n; still the sign separation is strong.

3) **Confounders**

- **Margin-critical definition / projection mechanics**: omission of `D_norm(h)` for margin directions means the subspace may be systematically misoriented [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:150), [shared/model.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:59).  
- **Projection methodology**: `P_crit @ Phi @ P_crit` is not the standard restricted FTLE operator used in the theorem statement and can understate growth from vectors starting in the subspace [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:284).  
- **n_crit = 0 prevalence**: thresholding only top-100 vectors with `crit_threshold=0.3` leads many zero counts at several T, while the same file still reports `lambda_R_direct` from the bilinear projected product [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:257) (methodological inconsistency).  
- **Fixed-point approximation**: `fp_steps` is capped at 100 but there is no convergence gate here; `fp_residual` is large (~13.7–19.3), so the nominal manifold point is not truly at a fixed point [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:205), [exp_d29b_ftle_corrected.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d29b_ftle_corrected.json:537).

4) **Alternative explanations (besides Proposition 31)**
- Non-normal/transient finite-time effects: positive/negative exponents can coexist and vary with sampling location without representing asymptotic invariant subspace dynamics.  
- Subspace mis-specification can produce a negative projected value even if true readout-relevant FTLE geometry differs.  
- Input-dependent Jacobians and using target-token rather than current-argmax criticality can bias the “critical direction” basis for some samples [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:141).  
- Numerical conditioning and finite-T truncation effects can compress signal, especially with repeated Jacobian products in one sample.

5) **Generalizability**
- This is a **single training condition**: `variant="variable_t", L=8, seed=42, d=128` [exp_d29b_ftle_corrected.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d29b_ftle_corrected.json:454).  
- No fixed-T replicate, no L sweep, no seed sweep in this run.  
- Therefore cross-configuration confidence is low until those ablations are done.

6) **Quantitative consistency**
- If `lambda_R = -0.004` were the local readout contraction rate from manifold, it implies very slow contraction; to get ~100× readout-error reduction via `exp(lambda_R*T)` would require hundreds of steps, not T=3. The proof’s local-to-global conversion caveat explicitly warns about this mismatch when measurements are local vs early-trajectory behavior [bottleneck_depth_scaling.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2753).  
- `lambda_null ≈ +0.04` (factor 1.04/step) can *plausibly* support weak orthogonal expansion and is directionally consistent with late U-shaped updates, but this run does not directly link the exponent to per-step norm trajectories [bottleneck_depth_scaling.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2816).

7) **Confidence scores (1–10)**

- “**Prop 31 is confirmed**”: **6/10** (strong directional support, but methodological gaps prevent full confirmation)  
- “**Anisotropic contraction mechanism is THE explanation for readout convergence**”: **5/10** (likely part of mechanism, but not isolated and maybe only partially measured)  
- “**This result will replicate on different configs**”: **3/10** (no config/seed ablations in this run)

**Severity-ranked issue summary**

- [High] Omitted readout normalization Jacobian in critical/null subspace construction [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:150), [shared/model.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/shared/model.py:59)  
- [High] `P_crit @ Phi @ P_crit` is not the standard restricted-FTLE object used in the stated theory [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:284), [bottleneck_depth_scaling.md](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2578)  
- [High] Fixed-point approximation is not verified; residual is large [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:205), [exp_d29b_ftle_corrected.json](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d29b_ftle_corrected.json:537)  
- [Medium] Top-100 alignment truncation + threshold choices can miss aligned modes and makes `n_crit` unstable [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:257), [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:263)  
- [Medium] Critical direction uses dataset target token (`y_correct`) rather than current argmax/decision boundary for inconsistent points [exp_d29b_ftle_corrected.py](C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/exp_d29b_ftle_corrected.py:141)  

