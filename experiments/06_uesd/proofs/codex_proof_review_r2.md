I reviewed all requested files and the Exp D JSON.

### 1) Theorem 5 (CE-to-probability) in [convergence_correctness.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\convergence_correctness.md)
- Current status since R1: **Addressed**
- Mathematical correctness rating: **STRONG**
- Status note:
  - The file now uses \(p(y^*|s_T)\ge \exp(-\varepsilon)\) with \(\varepsilon=\delta/\lambda_2\), not \(1-\varepsilon\).
  - Margin lower bound is now:
    \[
    m \ge \tau\log\frac{p(y^*)}{\max_{v\neq y^*}p(v)} \ge \tau\log\frac{e^{-\varepsilon}}{1-e^{-\varepsilon}}.
    \]
- Empirical support:
  - **Yes for E5:** addition/e5 runs have low CE (\(\ll 0.1\)) and large positive margins.
  - **No for E1:** CE stays near \( \ln 64 /2\approx2.08\), so only weak bound, matching near-zero correctness coupling.

### 2) Spectral condition in [spectral_contraction.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\spectral_contraction.md) and [nonnormal_stability.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\nonnormal_stability.md)
- Current status since R1: **Addressed**
- Mathematical correctness rating: **STRONG** (for statement of condition), **MODERATE** (for sufficiency claims around finite-T use)
- Exact required form:
  \[
  |1+\lambda_i(dF/ds)|<1 \quad \Leftrightarrow \quad \lambda_i(dF/ds)\in\{z:|z+1|<1\}
  \]
  rather than \(\Re(\lambda)\in(-2,0)\).
- Empirical support:
  - **Partially supports:** Exp D addition_e5 lam0.1/1.0 have \(\text{max }\rho\approx1.00\) and high accuracy.
  - **Contradictory signal:** E1 sort/dedup reported \(\rho_{\max}>1\) (1.19865 / 1.709 from your context), which is incompatible with asymptotic contraction assumptions if interpreted literally.

### 3) MI-temperature claim in [information_bottleneck.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\information_bottleneck.md)
- Current status since R1: **Addressed**
- Mathematical correctness rating: **STRONG**
- Exact correction:
  - Temperature \(\to 0\) does **not** imply \(I(h_t;x_t)\to 0\).
  - Deterministic argmax can retain substantial MI; only non-invertible quantization destroys information.
- Empirical support:
  - **No direct empirical contradiction in current tables** (this is a theory framing correction).  
  - E5 success can still be interpreted as “process advantage” (deferred commitment + continuous refinement), consistent with this section.

### 4) Finite-\(T\) treatment in [finite_step_convergence.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments/06_uesd/proofs/finite_step_convergence.md)
- Current status since R1: **Addressed (partially)**
- Mathematical correctness rating: **MODERATE**
- Exact issue/fix:
  - Existing bound needs explicit neighborhood and norm-control assumptions.
  - Recommended hard form:
    \[
    \|s_T-s^*\|\le \|J\|^T\|s_0-s^*\|+\frac{M}{1-\|J\|}\|s_0-s^*\|^2\quad(\|J\|<1).
    \]
    For non-normal \(J\), use \(\|J\|=\sigma_{\max}(J)\) or include a transient-growth constant (Kreiss/logarithmic norm path), not only \(\rho(J)\).
- Empirical support:
  - **Contradicts naive spectral-only expectation:** several models with mean \(\rho\approx0.97\!-\!0.99\) are high-accuracy, but T=10 is borderline if \(\rho\) is near 1 and non-normal growth exists.
  - **Supports** the need for explicit finite-\(T\) diagnostics (D5).

### 5) Nonnormal Jacobian risk in [nonnormal_stability.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments/06_uesd/proofs/nonnormal_stability.md)
- Current status since R1: **Addressed**
- Mathematical correctness rating: **MODERATE**
- Exact required fix (explicitly still open):
  - D6 currently checks \(\rho(J)\), not \(\sigma_{\max}(J)\).  
  - Need to add a theorem/diagnostic for finite-\(T\):
    \[
    \|s_T-s^*\|\le \sigma_{\max}(J)^T\|s_0-s^*\| + \text{nonlinear remainder}.
    \]
- Empirical support:
  - **Supports in part:** E1 failure and high \(\max\rho>1\) are consistent with transient/instability behavior.
  - **Supports practical claim cautiously:** E5 runs show high basin stability despite near-critical \(\rho\), but this is empirical, not a bound on non-normal amplification.

### 6) Fixed-point existence/uniqueness in [fixed_point_existence.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments/06_uesd/proofs/fixed_point_existence.md)
- Current status since R1: **Addressed (partially, with a known gap)**
- Mathematical correctness rating: **WEAK**
- Exact needed correction:
  - Replace claimed  
    \[
    \|s^*-s_T\|\le \frac{\|F(s_T,c)\|}{1-\rho(J)}
    \]
    with a condition-based bound using inverse norm:
    \[
    \|A^{-1}\|\text{ with }A=dF/ds|_{s_T},\quad
    \|s^*-s_T\|\le \frac{2\|A^{-1}\|\|F(s_T,c)\|}{1+\sqrt{1-2L\|A^{-1}\|\|F(s_T,c)\|}}
    \]
    (e.g., Kantorovich/Newton-type form), requiring explicit \(L\) and invertibility bound on \(A\).  
  - Do **not** derive \(\sigma_{\min}(A)\ge 1-\rho(J)\) in non-normal regime.
- Empirical support:
  - **Mixed:** E1 addition has low residual but wrong convergence basin; “approximate fixed point exists” is consistent, but wrong-attractor proximity remains.

### 7) Missing derivation a) CE-to-margin conversion
- Current status since R1: **Addressed**
- Mathematical correctness rating: **STRONG**
- Exact statement currently present:
  \[
  CE<\varepsilon \Rightarrow p(y^*)\ge e^{-\varepsilon},\quad
  m \ge \tau \log\frac{e^{-\varepsilon}}{1-e^{-\varepsilon}}.
  \]
  This is the correct explicit finite-\(\varepsilon\) bound.
- Empirical support:
  - **Yes:** E5 CE collapse gives strong margins and high correctness.

### 8) Missing derivation b) fixed-point perturbation bound
- Current status since R1: **Not fully addressed**
- Mathematical correctness rating: **WEAK**
- Exact fix needed:
  - Add one theorem with explicit assumptions:
    - \(F\in C^2\), \(F(s^*,c)=0\), \(A=dF/ds|_{s^*}\) invertible, \(\|A^{-1}\|\le \beta\), and \(\sup\|d^2F\|\le L\).
    - If \(2L\beta\|F(s_T,c)\|<1\), then unique nearby \(s^*\) and explicit \(\|s^*-s_T\|\) bound as above.
- Empirical support:
  - **No:** current results do not verify this perturbation theorem directly.

### 9) Missing derivation c) wrong-attractor risk under shift
- Current status since R1: **Not addressed**
- Mathematical correctness rating: **WEAK**
- Exact fix needed:
  - State a distribution-shift theorem for wrong-attractor rate. A workable form:
    - Assume \(c\mapsto s^*(c)\) and \(m(s^*(c))\) are Lipschitz with constants \(L_s, L_m\) in a neighborhood.
    - Let margin-risk surrogate \( \phi_\gamma(s)=\mathbf{1}\{m(s)<\gamma\}\) (or hinge-smoothed).
    - Then for shift \(W_1(P,Q)\le \eta\):
      \[
      R_Q(\phi_\gamma)\le R_P(\phi_\gamma)+L_m L_s\,\eta/\gamma
      \]
      and wrong-attractor rate is bounded by choosing \(\gamma\) and a coverage radius in \(c\)-space.
- Empirical support:
  - **Partly suggestive only:** E1 shows 100% wrong-attractor despite low residual, indicating unmodeled shift/multiple-attractor structure.

### 10) Missing derivation d) separating decoder shaping (CE) from dynamics semantics
- Current status since R1: **Not addressed**
- Mathematical correctness rating: **WEAK**
- Exact fix needed:
  - Add explicit decomposition theorem for parameter blocks \( \theta=(\phi,\psi)\):
    \[
    L(\phi,\psi)=\lambda_1\|F_\phi(s_T,c)\|^2+\lambda_2 CE(R_\psi(s_T),y^*),
    \]
    with conditions under which (i) \(F_\phi\)-dynamics determine attractor set and (ii) \(R_\psi\)-loss only selects among reachable attractors.  
  - Formalize gradient coupling terms and identify when they decouple (e.g., block-separable Jacobian / weak cross-Hessian region).
- Empirical support:
  - **Not demonstrated:** Exp D does not yet isolate this mechanism with ablations (e.g., CE frozen, readout-free dynamics-only metrics, or readout-only controls).

---

## Overall verdict on publication readiness
**NOT ready for a theorem-strong publication.** The corrected CE bounds and many core conceptual errors are fixed, but rigorous finite-\(T\) guarantees under non-normality, distribution-shift risk, and fixed-point perturbation claims still need formalization.

## Priority-ordered remaining fixes
1. Replace fixed-point perturbation step from \(1/(1-\rho)\) to explicit inverse-norm/Kantorovich conditions (including non-normal caveat).
2. Add a finite-\(T\) stability theorem using \(\sigma_{\max}(J)\) (or Kreiss/logarithmic norm bound), plus optional D7 \(\kappa=\sigma_{\max}/\rho\).
3. Add formal wrong-attractor risk theorem under distribution shift with explicit coverage/margin-radius term.
4. Add explicit dynamics-decoder separation theorem (parameter/block/gradient decomposition) to justify “semantics via CE, stability via residual” split.
5. Add non-asymptotic proof path for phase-transition/symmetry-breaking behavior during \(\lambda\)-warmup.

## Do the proofs adequately explain Exp D findings?
**Partially.**
- They explain E1 failures via wrong-attractor existence + weak CE coupling well.
- They explain E5 success on addition/dedup with high margins + low wrong-attractor + near-1 but controlled dynamics only partially.
- They do **not** yet fully explain the sharp phase transition at step ~4000 or the reported high \(\rho_{\max}>1\) in E1 sort/dedup as a formal bifurcation/transient-growth event.