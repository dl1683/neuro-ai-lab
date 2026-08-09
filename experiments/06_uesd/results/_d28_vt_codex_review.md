Your interpretation is directionally right, but you should treat it as a **phase-conditional mechanism**, not a theorem-level replacement yet.

1. Validate/challenge the D=8 anomaly
- The anomaly is real and internally coherent with the logged metrics: `rho_VT` flips above `rho_FT` at D=8 while `k_frob` and `T_99` still improve.
- But this is a **single-condition refutation** of the old universal claim, not yet a full falsification of the broader framework.
- Strong point: the D28 table already encodes the falsification as a clean sign flip (`Δρ=+0.0014`) with unchanged `T_min=4`.
- Weak point: this is still one operating point (and effectively one heavy confound regime: long-depth carry, low `T_min`) so inference about mechanism is provisional.

2. Is “T_min insufficiency” sufficient, or are there alternatives?
- Yes, as a mechanism, it is plausible and matches your own D30 control in the dual direction (`Δρ` shifts with `T_min`; H1/T99 evidence supports training-pressure logic).
- But alternative explanations are at least equally serious:
  - `Δρ` could be driven by **non-normal transient structure** rather than average spectral penalty; short-horizon batches inject directionally misaligned gradients when trajectories are far from solved at that horizon.
  - D8 may have a heavier load of **hard/unsolved samples at T=4**, so VT gradients are partly optimizing a “do-something-different” objective instead of clean spectral suppression.
  - Residual optimization mismatch (not-yet-converged/low-accuracy VT state) can bias `rho` measured at `s*`; the same seed dependence and wrong-attractor dynamics elsewhere in this series suggest this is real risk.
  - Your boundary cannot stay as `T_min ≷ D`; internal evidence already says borderline at D=6 and then flips at D=8, implying a **D_intrinsic or solvability-at-T metric** is the right latent variable.

3. Does this strengthen or weaken the UESD thesis?
- It **weakens the strict “VT always lowers rho by constant Δ” thesis**.
- It **strengthens the anisotropic, readout-centric thesis**: VT can improve readout convergence (`T_99`) while hurting null-space amplification (`rho`).
- Net: global thesis survives, but the “constant Δρ” corollary is demoted from law-like claim to conditional regime claim.

4. Is the three-way decoupling genuine or artifact?
- Genuinely plausible mathematically from the FTLE decomposition: `k_frob` (average), `rho` (max), `T_99` (readout direction) should decouple under anisotropy.
- Empirically, the D28 D=8 row is exactly the expected pattern.
- But artifact risk is nontrivial:
  - `k_frob` and spectral diagnostics are aggregate approximations; direction-specific definitions are sensitive to subspace estimation choices.
  - Existing caveats in your own docs flag non-normal/estimation issues and limited replication for some FTLE-style claims.
- Verdict: insight is real, but current strength is **“hypothesis-grade to moderate”**, not final.

5. Prediction for D=10 (L=20) and D=12 (L=24)
- Under the revised boundary view, expect VT likely continue to fail at suppressing rho at these depths when `T_min=4`, potentially with `rho_VT` ≥ `rho_FT`, and likely >1.003-ish in absolute drift terms depending on saturation effects.
- `T_99` should track the same boundary logic: continue near 4 while intrinsic-depth-limited regime stays active, with possible shift/plateau behavior around the “exception” style observed at higher depths (your own notes already report a D=12-like exception pattern).
- `k_frob` should remain architecture-like (~0.988) unless architecture changes.
- `VT accuracy` likely continues to degrade at depth, potentially below 95% at D=10 and worse variance at D=12.

6. Confidence in revised Corollary 30.1 (conditional form)
- I’d rate it **5/10 (provisional)**.
- Upgrade path:
  - Replicate D=8 at >5 seeds and fixed seed-matched training/evals.
  - Add explicit D30-style sweep in D while holding `T_min` and matching budgets.
  - Add D=10/12 VT at multiple `T_min` values to estimate the decision boundary in `(D, T_min)` rather than binary rules.

7. What to derive mathematically
- Derive a **phase-boundary surface** instead of a scalar corollary:
  - Let `q(D,T_min)` = fraction of samples solvable by `T_min` steps (or solvability margin proxy from T-accuracy).
  - Then write
    \[
    \Delta \rho(D,T_min)\approx -A(T_min)\,q + B(T_min)\,(1-q) + \epsilon
    \]
    with \(A>0\), \(B\) capturing gradient conflict when gradients are from undercomputed trajectories.
  - Boundary is `Δρ=0` at \(q^\* = B/(A+B)\), not at `T_min=D`.
- Also formalize the 3-metric split:
  - \(\log \rho = \lambda_{\max}\) (readout-orthogonal/extreme FTLE),
  - \(k_{\text{frob}}\approx \exp(\bar\lambda)\),
  - \(T_{99}\approx \ln(0.01)/|\lambda_R|\),
  and make explicit that `VT` can shift \(\lambda_R\) and \(\bar\lambda\) without fixing \(\lambda_{\max}\).

Natural next step: run a 2D grid \(D\in\{8,10,12\}\times T_{\min}\in\{2,4,6,8\}\) at matched seeds to estimate the boundary surface before writing a second revision. If you want, I can draft that experiment card (minimal confound-controlled protocol + expected falsifiers).

