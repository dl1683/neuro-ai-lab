## 1) FALSIFICATION TESTS

These are designed to **falsify** the parallel computation engine claim, not confirm it.

1) **Single-step ablative falsification**
- Test: run the same checkpoints with recurrence disabled at test time (or forced to 1 update step), keeping parameters, prompt, and decoding identical.
- Metric/threshold: if `seq_acc(T=1)` is within `Δ <= 0.02` of `seq_acc(T=K)` for `K>=8` on carry tasks across ≥3 random seeds, engine claim is weakened/falsified.
- Meaning: no measurable dependence on iterative dynamics means the model is not gaining computation from repeated state updates.

2) **No repair despite budgeted compute**
- Test: D17-style corruption recovery with extra steps.
- Metric/threshold: if for corruption classes with ≥2 carries, `R_rec(20)` (final recovery at 20 extra steps) stays below `0.15` and never exceeds `R_rec(4)` by more than `0.02`, repeated updates are not functioning as error-correcting computation.
- Meaning: “parallel computation” is not being realized as iterative repair; performance is likely static/shortcut.

3) **Causal control test on carry features (strong)**
- Test: targeted latent interventions on high-carry examples (D8-style) and evaluate output perturbation persistence.
- Metrics:
  - `MI(intervention variable, final output)`  
  - `P_persist(output flip at horizon T)` (e.g., at T=10, T=20)
- Threshold: if `MI < 0.01 bits` and `P_persist < 0.20` while carry-in classifiers remain high, then carry-like states are not being causally used.
- Meaning: representation looks decodable but non-causal ⇒ not a working compute mechanism.

4) **Robustness inversion under increasing carry burden**
- Test: sweep corruption hardness (`k=1..K` carries, or carry-depth).
- Metrics: final accuracy and recovery energy vs carry count.
- Threshold: if higher carry count does not monotonically increase required recovery energy **and** recovery probability, but instead stays flat while accuracy remains high, then evidence points away from a genuine algorithmic correction regime.
- Meaning: would indicate brittle token-level matching or dataset-specific artifacts rather than scalable corrective dynamics.

5) **Wrong-attractor collapse under mild perturbation**
- Test: measure wrong-attractor (WA) rate on identical architecture/training when tiny latent noise is injected.
- Threshold: if `WA > 0.05` while task accuracy stays “good” and does not recover with extra steps, the system is not a stable iterative solver.
- Meaning: apparent accuracy could come from lucky basins rather than controlled computation.

6) **Bottleneck irrelevance**
- Test: vary `V` (thus softmax-capacity `log2(V)`), keep everything else fixed, and fit accuracy.
- Threshold: if accuracy and recovery are statistically flat while cap varies materially (`Δlog2(V) >= 2`) and no matching change in recurrent metrics (lyap/sigma/WA) is needed, then the proposed “bandwidth bottleneck + iterative bypass” story is unsupported.
- Meaning: architecture is probably not constrained by the proposed channel mechanism.

---

## 2) SCALING PREDICTIONS

Using the same bottleneck framing used in your files:

- Softmax-channel capacity per step: `C = log2(V)`.
- For `V=256`, `C=8` bits.

For `d=512, V=256, L=16`:
- If required latent information per input packet scales roughly as `R≈L * log2(V)` (same style used in bottleneck framing), then `R≈128` bits.
- Deficit ratio `≈ 1 - 8/128 = 0.9375` (i.e., still heavy bottleneck, but less extreme than older 16-bit-ish token-level deficits if those were the baseline comparisons).
- If your older convention uses larger effective `R` (the 4k/8k-style accounting), deficit remains near 0.996+.

What breaks first (most likely):
1) **Immediate carry recovery**
   - At low depths, carry-depth-2 to carry-depth-4 cases collapse into partial/unstable correction, similar to but worse than D17 phase-3 trends.
2) **Stability regime switches too early**
   - You likely see CE-style transient/rotation and E5-style compression diverge by hardness; one path will over-rely on short-horizon “fixes” while failing on longer/denser carry interactions.
3) **Wrong-attractor rise under compositional stress**
   - WA increases first on examples with stacked carries, then degrades sequence-level performance while token-level stays deceptively high.

Harder task: multi-digit multiplication
- Expect **earlier failure** than addition due to multi-step, cross-digit dependencies and non-local carry interactions.
- Likely signatures:
  - lower `final_seq_acc` than token-level,
  - much higher recovery energy per fixed token error,
  - stronger seed sensitivity,
  - quicker saturation of `R_rec(T)` and more non-normal collapse signatures.
- If any engine works here, it will likely need:
  - more depth or width, and/or
  - larger token budget/effective channel capacity, and
  - stronger Jacobian regularization to control WA under dense carry interactions.

---

## 3) TOP 5 NEXT EXPERIMENTS (not in current queue)

1) **Inference-only step-ablation test across all current checkpoints**
- What: evaluate `T=1,2,4,8,16,32` for all existing models; compare clean + corrupted + carry perturb variants.
- Why high info: one retrained model gives a direct falsifier of “iterative compute” with low cost.
- Decision power: if depth doesn’t move the needle, thesis collapses quickly.

2) **Counterfactual causal basis probing (low-rank edit test)**
- What: identify top latent directions predictive of carry-in; orthogonally project/replace them and rerun with identical output heads.
- Metrics: output flip rate, MI(carry-feature, output), persistence.
- Why high info: directly tests causality rather than correlation; very discriminator-heavy with moderate compute.
- Decision power: distinguishes representational echo from genuine control flow.

3) **Parametric sweep d/V/L with fixed 5e5-step cap and 3 seeds**
- What: small factorial on (`d∈{128,256,512}`) × (`V∈{64,128,256,512}`) × (`L∈{8,16,24}`) for one backbone + one task split.
- Why high info: locates true bottleneck-threshold surface (not task-specific anecdotes).
- Decision power: if no clear scaling law appears, “parallel engine via bottleneck compensation” weakens.

4) **Challenge generalization: same architecture, OOD finite-state tasks**
- What: train on addition/dedup, test on equivalent but unseen automata (e.g., subtraction, base-mixed affine updates, parity automaton).
- Metrics: zero-shot OOD sequence accuracy, recovery under corrupted states.
- Why high info: proves whether dynamics learned a repair algorithm vs memorized arithmetic surface.
- Decision power: if fails OOD while in-domain is good, thesis becomes narrower and probably memorization-heavy.

5) **Multi-digit multiplication with controlled carry-depth curriculum**
- What: 2→4→8 digits, fixed budget, fixed seeds, and same corruption protocol used in D17/D8.
- Why high info: most stress test for the engine hypothesis.
- Decision power: if multiplication destroys recovery dynamics while addition holds, then current mechanism is domain-bound, not general parallel computation.

---

## 4) THEORETICAL FRAMEWORK (single unifying framework)

Use a single framework:
### **Rate-limited recurrent error-correcting dynamical system**
Model:
- State update: `h_{t+1} = F_θ(h_t, x_t)`
- Output channel: `y_t = Π(h_t)` with `I(y_t; h_t) ≤ C = log2(V)` (softmax bottleneck).
- Target computation requires latent state entropy `H(S|x) = R` with `R >> C`.

Interpretation:
- Since `R > C`, one-step readout cannot contain full task state.
- The model must use `F_θ` to run an iterative solver on the error manifold:
  - Linearized error dynamics: `e_{t+1} = J_t e_t + η_t`.
  - `σ_max(J_t)` and non-normality control transient amplification (exploration of state space).
  - `ρ(J_t)`/alignment control contraction toward a solution manifold.
- Correctness emerges from alternating/competing mechanisms:
  - **Expansion phase** (high non-normal amplification) to escape wrong attractor basins.
  - **Convergence phase** (alignment/compression) to land on the right basin.
- Wrong-attractor rate is exactly basin leakage; phase-dependent instability is encoded by Jacobian spectra and their alignment.
- This framework explains all observed facts in one go:
  - bottleneck mismatch (information theory),
  - differing CE/E5 behavior (different J-geometry),
  - recovery energy/weak persistence (path dependence in error flow),
  - and fragility under deeper compositional stress.

---

## 5) CONFIRMATION-BIAS CHECK (specific)

1) **D7 as evidence of “thinking emergence” is inconsistent**
- You have near-`0.5` token/`0.0` seq in the referenced artifact and weak carry-depth correlations. Treating it as positive evidence requires explaining away core contradiction.
- Risk: narrative inflation from isolated successful plots rather than robust signal.

2) **Causality inferred from decodability**
- D8 has high carry-in predictability but weak/short-lived output changes after flips.
- Risk: calling this “parallel compute in hidden” overstates representation–causation equivalence.

3) **High final accuracy used to mask energy fragility**
- D17/D5 show near-perfect final scores but high correction-energy/instability signatures in some settings.
- Risk: interpreting accuracy alone as robust algorithmic competence.

4) **Architecture confound in compositional results**
- `exp_d_compositional` suggests dynamics matter, but encoder baselines and task-specific confounds (especially dedup) can hide whether gains are mechanism-level vs distributional.
- Risk: over-attributing gains to engine structure.

5) **Path-dependent evidence selection**
- Missing expected file (`exp_d10_adaptive_halting.json`) and ledger naming mismatch suggest incomplete artifact set.
- Risk: conclusions may be built on a curated subset that omits disconfirming runs by accident.

6) **Seed-fragility treated as variance**
- Some key comparisons rely on few seeds (or single-seed outlier patterns).
- Risk: false confidence from median-of-a-few and not adversarially verifying across failure regimes.

If you want, I can convert this into a one-page “go/no-go decision sheet” with explicit stop conditions for each falsification test and a scoring rubric (`PASS / Inconclusive / FALSIFIED`).