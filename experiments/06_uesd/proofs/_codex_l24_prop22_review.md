## 1) Proposition 22 review (rigor vs suggestive analogy)

Short version: the current Prop.22 evidence is **plausible but mostly suggestive**, not yet a formal channel-coding theorem.

- Sources: [bottleneck_depth_scaling.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\bottleneck_depth_scaling.md), [theory_summary.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\theory_summary.md), [recovery_impossibility.md](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\proofs\recovery_impossibility.md)

1. **(a) Mapping rigor**
   - Not rigorous as a formal equivalence to BP.
   - Breaks:
     - No explicit probabilistic channel law \(p(y\mid x)\) tied to encoder outputs.
     - No explicit factor graph specified for all tokens/carry variables with parity-like constraints and local messages.
     - No formal derivation of BP-style extrinsic update decomposition.
     - Weight-tied Transformer block may implement rich non-BP dynamics (attention mixing + nonlinear residual map), so channel-code analogies are interpretive, not necessary.
   - What is useful: the **structural analogy** (repeated noisy input reinterpretation + consistency constraints) is good intuition.

2. **(b) L=24: 0% encoder seq_acc, 99.61% dynamics**
   - This is real in the logged data ([exp_d23_carry_depth_scaling.json](C:\Users\devan\OneDrive\Desktop\Projects\neuro-ai-lab\experiments\06_uesd\results\exp_d23_carry_depth_scaling.json)), but it is not yet uniquely explained by coding theory.
   - A simpler explanation is plausible: strong recurrent transform learns a task-specific attractor/solver in hidden space, partly independent of one-shot encoder quality.
   - Also, there is an internal doc inconsistency between 99.22% and 99.61% for the same L=24 row—this requires reconciliation before over-interpreting.

3. **(c) Per-token MI estimates**
   - If MI was computed directly from token-accuracy under a 64-ary symmetric model, L=24 tok_acc=29.77% would imply more than tiny ~0.17 bits/token (order ~0.9+ bits depending assumptions).  
   - Therefore the reported 0.173-bit figure is likely for a **reduced binary subproblem** (e.g., carry-edge uncertainty) or a different MI estimator, not raw full-token entropy.
   - So yes, MI table is potentially correct but **ambiguous unless metric definition is made explicit** (alphabet, prior, estimator, conditioning variable).

4. **(d) Untested falsifiable predictions from Prop.22**
   - Threshold behavior vs synthetic channel noise around \(\rho=\tanh(1/2)\approx 0.462\) has not been cleanly isolated.
   - If truly BP-like, performance should exhibit a **decoder-complexity vs SNR phase boundary** under controlled MI perturbations at fixed architecture.
   - Predicted dependence on schedule/edge-activity: breaking cross-attention re-reading should collapse gains if channel-code story is right.

5. **(e) Calibration**
   - Proposed level should be **WEAK-to-MODERATE**, not stronger.
   - Reason: strong empirical anomaly (L=24) is real, but mechanism is not uniquely attributable to BP/channel decoding.

---

## 2) L=24 breakthrough analysis

- **Encoder Independence Model, \(E_{\text{enc\_seq}} \approx \text{tok\_acc}^D\)**:
  - The derivation is mathematically sound under *independence* and “all-token-correct implies sequence-correct.”
  - For D=12, \(0.2977^{12}\approx4.9\times10^{-7}\), so seq_acc ≈ 0 at finite-sample scale is exactly what model predicts.
  - Existing L values show close-ish matches for earlier lengths, except some deviation (e.g., L12/L16).

- **C_step from 0.67 → 2.40**
  - Interpretable as stronger per-step coupling/amplification, i.e., recurrence becoming less “stable-passive” and more “solver-like.”
  - That can explain why weak encoder is rescued more at larger L, but also raises risk of brittleness/chaos under larger perturbations (consistent with recovery crossover).

- **Cliff prediction falsification**
  - Anti-cliff (weaker encoder \(\to\) stronger dynamics) is *consistent* with data so far, but currently not uniquely causal.
  - Confounds:
    - training-instance variance and optimization luck,
    - sample size / confidence intervals,
    - T-selection bias (if “best” over T is reported),
    - possible baseline metric mismatch (token vs sequence, fixed-step vs best-step).

- **Alternative explanations**
  - Learned implicit algorithmic map (not BP): recurrence acts like an internal iterative solver over carry structure.
  - Attractor basin widening from more T-steps and better-tuned latent dynamics.
  - Non-robust reporting artifacts (precision, inconsistent row entry, seed effects).

---

## 3) Universal +1 positive recovery

- Pattern: all six L values show small positive recovery at +1, then negative by +5/+10/+20 in many rows.
- This is consistent with **within-basin contraction near optimum** plus eventual basin-boundary escape under larger perturbation.
- Could still be artifact:
  - +1 values are very small (0.02–0.61%); without confidence intervals, significance is unclear.
  - Possible numerical/readout effects from post-hoc projection and noise sampling.
- Interpretation of crossover timing:
  - “Basin of attraction” has a small local core; dynamics correct minor noise but fail for larger perturbation, producing reversion beyond short-step range.

- Calibration impact on Prop.20:
  - Keep at most **WEAK-to-MODERATE**, not MODERATE, unless repeated-seed significance testing confirms the +1 effect is stable.

---

## 4) Cross-domain challenge to Prop.22

1. **Information theory**
   - Real BP claims usually require: explicit random variables, factor graph, message schedule, and a defined channel output LLR stream.
   - Here, no clear code rate, no explicit parity-check structure, and no decoder-input likelihood calibration. This is more **learned latent solver dynamics** than canonical coding.

2. **Dynamical systems**
   - A contraction/attractor explanation is more conservative and likely closer:
     - learned map \(s_{k+1}=F_\theta(s_k,c)\) iteratively denoises toward fixed points minimizing \(E_\theta(s,c)=\|F_\theta(s,c)\|^2\),
     - with local Jacobian spectra controlling immediate recovery vs long-horizon divergence.
   - This can generate the same +1 recovery then crossover pattern.

3. **Coding theory realism**
   - Turbo/LDPC decoders need explicit channel reliabilities and sparse code-constraint structure.
   - UESD currently lacks explicit syndrome constraints and LLR extraction guarantees; asking cross-attention tokens to be “messages” is useful metaphor only, not proof of equivalence.

---

## 5) Experimental gaps (to test Prop.22 rigorously)

- Add **controlled channel-noise ablations** to encoder embeddings:
  - inject AWGN at controlled SNR and vary SNR continuously; test if performance follows a threshold curve tied to \(\rho\).
- Freeze decoder dynamics and only degrade encoder quality; separately freeze encoder and change T to decouple hypotheses.
- Fix \(T\) (no best-over-T selection) when reporting key tables; report CIs over many seeds.
- Explicitly vary attention schedule and cross-attn access to context. If channel-analogy is right, removing re-read paths should hurt sharply.
- Add ablations that destroy carry-factor structure (token permutation, distractor coupling) while matching MI.
- For **D27 prediction**:
  - current design is directionally right, but incomplete unless multiple seeds, same compute budget, and matched randomness seeds.
- FALSIFY Prop.22 specifically:
  - Find a condition with lower raw MI but equal/stronger dynamics-than-expected where no cross-attn re-read is needed.
  - Or show high-quality decoders with no BP-like graph dependencies still produce the same gains.

---

## 6) Calibration audit

- **Prop 19 (currently MODERATE-to-STRONG)** → downgrade to **MODERATE**.
  - 6 points are useful but narrow protocol/task/architecture scope and some non-monotonicities.
- **Prop 20 (currently MODERATE)** → downgrade to **WEAK-to-MODERATE** unless significance/CI checks are added.
- **Prop 21 (WEAK-to-MODERATE)** → keep, or downgrade to **WEAK** if “untested prediction” remains mostly untested.
- **Prop 22 (WEAK-to-MODERATE)** → keep, but label as **hypothesis with strong suggestive evidence**, not structural proof.

---

## 7) Single most important next step

Run a **single falsification-focused benchmark** on one branch:

1. Fix architecture and \(T\).
2. Sweep synthetic embedding noise levels to set encoder MI precisely at each L.
3. Compare: full recurrence vs no-re-read vs no-cross-attn variants.
4. Include >10 seeds and report seq_acc + recovery curves with confidence intervals.

This directly targets the BP claim, not just the general “dynamics helps” claim.

---

## 8) Anti-overconfidence protocol (boring explanations to rule out)

- Resolve the 99.22% vs 99.61% inconsistency for L=24 before causal claims.
- Rule out seed/variance effects with multiple random seeds and CI bars.
- Avoid “best-T cherry-picking” in headline claims; report fixed-step baselines.
- Verify encoder and dynamics metrics use the same tokenization, decoding, and sequence-length convention.
- Check whether tiny +1 recovery is statistically distinguishable from 0.
- Validate no parser/data leakage between training/eval for L=24 and D values.
- Test for hidden short-path memorization by out-of-range or scrambled carry instances.

---

Next, I can turn this into a stricter “claim-strength table” with explicit status (proven/observed/inferred) and a one-page experimental protocol you can run next.