1. RESULT INTERPRETATION

- Constant Δρ at D=10 is **not statistically demonstrated** from this run set. With one seed per config and no replicate variance, you cannot claim significance for a single point. Using the stated “constant” target Δρ≈−0.0026±0.0002, D=10’s Δρ=−0.0025 is a ~0.5σ deviation; statistically it is indistinguishable from the same-at-all-depths value.  
  [exp_d28_contraction_ratio.json](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/results/exp_d28_contraction_ratio.json)

- D=8 anomaly interpretation:
  - (a) Seed-dependent fluctuation is still the leading hypothesis because one-seed, and D=8 is the only violating depth.
  - (b) `V_crit` near-saturation at D=8 is plausible as a **mechanistic amplification** channel (your own note in this section already flags this in-file), but that would need matching D=8 multi-seed evidence.
  - (c) A genuine phase boundary is less supported right now because D=10 restored the “constant” pattern, weakening a monotone boundary explanation.
  - (d) Most likely blend: a regime interaction (near-saturation + seed-specific optimization trajectory) at D=8, i.e., (a)+(b).

- FT non-monotonic pattern {1.0018, 1.0026, 1.0024, 1.0016, 1.0042} is best read as “competing effects with different scaling directions” rather than a single smooth function: early peak (complexity term), mid-depth squeeze, then an apparent horizon-driven surge at eta≈1 for fixed-T at D=10. This matches the revised `Training Horizon Strain` framing for FT only.

2. THEORY STATUS

- The strain model’s original claim about VT behavior at D=8 is clearly contradicted by the measured D=10 VT point (and this is now in-file). So for VT, the model no longer has causal status for sign/magnitude of Δρ.
- The FT explanation for non-monotonicity still has support (minimum at D=8, surge at D=10), but it is no longer sufficient to explain VT shift.
- Two-component decomposition  
  `ln(ρ)=f_complex(D)+g_strain(D/T)` is still defensible as an **FT phenomenology**, but current data say VT shift is better described by an additive offset term tied to training objective pressure.
- Constant Δρ means: at fixed `T_min`, VT appears to impose a near-constant reduction in `ρ` (or equivalently max-FTLE) for most depths; this is architecture/tuning-driven pressure on readout-relevant trajectory stability, largely separable from the FT depth structure. This is exactly what your own Cor 30.1 revision narrative in-file converges to (when excluding D=8).  
  [bottleneck_depth_scaling.md](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:2923)

3. D=8 DIAGNOSIS

- Minimum discriminating experiment: rerun only D=8 at D28 protocol with both fixed_t and variable_t using **the same 8–10 seeds** (same random seed list used elsewhere), plus a small D=6 and D=10 seed-matched control to check whether D=8 is isolated.
- Statistical probability framing:
  - If residuals are Gaussian with σ=0.0002, one-tail 20σ event probability is ~`Φ(-20) ≈ 3.9e-90`; two-sided is ~`7.8e-90`.  
  - Across 5 depths (single-seed family-wise): `P(any |20σ) ≈ 5*2Φ(-20) ≈ 4e-89` (or ~`~10^-88`).  
  - This is vanishingly small under model assumptions; the real inference risk is that model assumptions are violated (single-run noise + dependence + non-Gaussian seed effects), not that random noise with σ=0.0002 produced it.

4. PREDICTIONS FOR PENDING EXPERIMENTS (revised understanding)

- D28 L=24 FT (D=12, T_train=10): use explicit in-file prediction band: **ρ_FT(D=12) ∈ [1.003, 1.006]**.
- D28 L=24 VT (D=12): revised constant-offset expectation: **Δρ_VT ≈ −0.0025**; hence **ρ_VT(D=12) ≈ ρ_FT(D=12) − 0.0025**, i.e. about **[1.0005, 1.0035]** (and likely lower than FT by ~0.0025).
- D30 config D (T_min=8): use revised bound in-file, **ρ ≈ 1.001–1.002**, close to FT at D=4.  
  [bottleneck_depth_scaling.md](/C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/06_uesd/proofs/bottleneck_depth_scaling.md:3066)

5. THREE COMPETING FT MODELS

- Best by fit simplicity on current 5-point FT set: **Model A (piecewise linear)** (single-shape, kinked, no extra coherence assumptions). It captures rise→dip→late rise with minimal assumptions.
- Model B (logistic) is structurally weak because it cannot natively produce the D10 surge with local D8 dip in one smooth monotone transition.
- Model C (peaked strain) can explain FT non-monotonicity but is currently under-identified against 5 points + single-seed variance.
- L=24 FT is the discriminator:
  - Model A: continued rise from trend near D/T=1.
  - Model B: saturating/plateau behavior.
  - Model C: decline after peak.
- Other model class to consider: a **2D solvability-regime model** with depth-dependent `q(D,T_min)` plus VT offset, instead of univariate ρ(D) curves (as the in-file 2D unification suggests). That gives falsifiable boundaries and avoids ad hoc one-dimensional curve fitting.

6. ANTI-OVERCONFIDENCE CHECK

A critic can legitimately say: “this is post hoc curve surgery across the same tiny, single-seed dataset.” The theory has repeatedly been adjusted after seeing each new point (quadratic → strain → conditional → corrected conditional), which is exactly the overfitting danger your own confidence history documents. Risk is not just wrong predictions; it is model churn without pre-registered discriminants.

Simpler explanation for all 10 points (5 FT + 5 VT):
- A robust **additive VT pressure term** on max-FTLE (roughly constant for fixed `T_min`),
- plus a **depth/horizon-dependent FT baseline** (architecture + task compute geometry),
- plus a large seed-level/optimization mode component that is currently unresolved.
This avoids forcing gradient coherence semantics to explain every anomaly; it keeps strain for FT structure and explains VT mostly via a bounded regularization term.

7. PRIORITY DIRECTIVE (single most important next step)

Run one targeted 2-branch, multi-seed replication:

- Experiment: D=8, `d_model=128, h=4, d_ff=512, n_enc=2, vocab_size=64, train_T=10`.
- Branches: fixed_t vs variable_t (same `T_min` policy as D28), with **8 seeds each**.
- Metrics: `ρ` (power-iteration at s*), `k_mean`, `T_99`, `acc`, and per-seed Δρ vs existing FT baseline.
- Specific adjudication:
  - If Δρ remains >0 for most seeds → D=8 conditional phase boundary likely true.
  - If Δρ re-centers near −0.0025 and D=10 behavior repeats → D=8 was one-seed anomaly; simplify model to constant VT shift + FT depth baseline.

