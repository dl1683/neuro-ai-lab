1. **Statistical validity (generalization scope):**  
   The evidence is currently strong for one *easy* regime only: `L=4 (D=2)` and fixed-T dynamics, not for harder depths. It is a single-configuration result (one `seq_len` point, one seed, one run) and `T_fixed=...`/`fp_steps=100`, so it cannot support a general claim about UESD at large depth.  
   At `L=8+ (D=4+)` you should expect qualitatively different behavior because carry depth pressure increases and previous experiments in this repo suggest depth changes can shift optimization pressure and stability geometry (e.g., nontrivial depth-dependent behavior in the same experiment stack). So the manifold picture at `L=4` is plausible but unproven as a universal law.

2. **Alternative explanations for `rho > 1` + fast readout:**  
   - **(a) `T=100` insufficient:** possible, especially if the trajectory is slow and mostly neutral/transverse, but with mean residual `0.099` and continued drift of `~7` units/step after readout is done, this does not alone explain a 387-step Frobenius prediction failure.  
   - **(b) FP exists but is far/remote:** very plausible. A distant weakly attracting/neutral structure can give short readout hit-time but no fixed-point convergence in 100 steps; this is compatible with a manifold-like interpretation.  
   - **(c) Numerical issues in spectral diagnostics:** always possible, but the observed pattern is broad and internally consistent: tight `k` distribution (`std=0.0017`, no bimodality), and coherent geometry decomposition. A pure numerical artifact is less likely than a real geometric effect, though re-checking Jacobian/radius estimation with independent methods (e.g., autograd vs FD, different eps/steps, same sample) is still needed.

3. **Theory revision vs rejection risk:**  
   It is too early to *reject* Corollary 25.1 / Proposition 25 globally from `L=4` only. The current file already states these were derived for harder regimes (`D>=4` in the open questions context), and `L=4` is explicitly easy (`D=2`).  
   A stronger revision is: **falsify-by-regime**, not universal rejection.  
   - Keep Banach-style claims as hypotheses under conditions: sufficiently easy/noise-filtered regimes with an actual convergent FP and contraction in relevant coordinates.  
   - Add explicit caveat: `L=4 fixed_t` shows this regime fails.

4. **Falsification criteria from future `L=8+` D28:**  
   - **Restores Banach framework if you see:**  
     - `rho < 1` with good margin and small variance,  
     - FP residual small (`→0`) by `T=100` (or longer fixed horizon),  
     - `k_frob` and readout convergence times close (no huge `T99` gap),  
     - and no persistent post-readout state drift (or drift that still converges to the same attractor).  
   - **Confirms readout-stable manifold if you see:**  
     - `T99(readout)` remains tiny (`~4–5`) while full-state convergence remains absent/very slow,  
     - `k_read` consistently `<< k_frob` and this remains across seeds,  
     - trajectory decomposition showing bounded normal error to manifold but sustained tangential motion,  
     - and recurrent non-convergence or neutral/expanding tangential modes across long horizon.

5. **Novelty assessment:**  
   The core mathematical idea is **not fundamentally new** in dynamical systems. Readout-correct invariant sets, neutral directions, and center/slow manifolds are standard language.  
   The new part is mostly the **UESD-specific empirical framing**: tying this to transformer readout geometry, `k_frob`/`k_read` discrepancy, and the specific failure mode where trajectory reaches correct logits early and then drifts while staying readout-correct.

6. **Single most important next step:**  
   Run one controlled D28 sweep at `L=8` and `L=12` (fixed_t + variable_t, same architecture, same seed set) and, for each, log: `k_frob(t)`, `k_read(t)` from readout projection, spectral-radius profile, FP residual at `T=100` and `T=300`, plus normal/tangential decomposition. This one batch is the cleanest discriminator between:
   - “Banach-like contraction to FP” vs
   - “readout-stable manifold + manifold drift”  
   in the first nontrivial depth regime.

Sources used:  
- [experiments/06_uesd/proofs/bottleneck_depth_scaling.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md)  
- [experiments/06_uesd/results/exp_d28_contraction_ratio.json](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\results\exp_d28_contraction_ratio.json)  
- [experiments/EXPERIMENTS.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\EXPERIMENTS.md)

