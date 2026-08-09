### 1) THESIS CONFIDENCE UPDATE (anti-overconfidence)

- **T1 — Dynamics are essential for compositionality:** **7.0/10** (down from ~9/10).
  - New evidence: D (addition) still shows encoder-only at 73.16% seq vs UESD ~100% with CE-dynamics [Exp D in `EXPERIMENTS.md`].
  - But D2b+D2d show this is **not strictly necessary**: properly seeded deep encoders (8L) recover addition reliably (5/5, ~99.98% seq) [EXPERIMENTS.md, `exp_d2d_depth_sweep.json` via `EXPERIMENTS.md`], so claim shifts from “necessary in principle” to “essential under compact parameter budgets / fixed depth.”
  - Net: strongest form weakened by depth-match controls, but still supported for the original efficiency framing.

- **T4 — E5 (self-consistency) beats CE:** **3.0/10** (slightly down).
  - Evidence: in D2b, CE-dynamics (no SC) is 5/5 successful (seq 0.9997–1.0000), while E5 remains 4/5 (seq ~0.7999 mean) with repeated wrong-attractor behavior [EXPERIMENTS.md].
  - SC increases wrong-attractor risk in D2c (`kappa` higher with E5 than CE), and Codex review already marked E5 as counterproductive in this regime [theory/proofs + `EXPERIMENTS.md` sections D2/D2c].

- **T5 — Dynamics perform parallel computation:** **8.5/10** (slightly down from 9/10).
  - Evidence: D7/D8 style step profiles show all positions improve early and monotonically, with near-synchronous stabilization and position-agnostic carry effects for CE-dynamics (not strict left-to-right wavefront), including 0% newly-wrong after step 4 [EXPERIMENTS.md, D3, D3b].
  - Caveat: self-attention architecture and data scale remain specific; this is still a supported mechanistic claim, not a universal one.

- **T6 — Dynamics can causally repair perturbations:** **4.5/10** (from ~2/10; conservative increase).
  - New evidence: D25 variable-T-only gives **consistent positive recovery at σ=0.2** (+27.86% mean across seeds at +10 steps) and positive at +5/+10/+20; this is real improvement relative to the same model’s pre-repair WA.
  - But it is not universal: no or negative recovery at σ=0.1,+20 and higher σ still mostly outside basin; this is partial rather than robust repair behavior.

- **Overall confidence:** **6.0–6.2/10**.
  - Stronger confidence than before due controlled D2b robustness + D25 replication signal.
  - Still below earlier 6/10 because key claims were diluted by depth/architecture confounds and incomplete D27/D28 evidence.

### 2) CROSS-DOMAIN EVALUATION (theory_summary §6)

- **Strongest (pursue):**
  1) **Canalization / variable-T intuition (6.3)** — partially confirmed: variable-T improves basin usage; D22-style and D25 σ=0.2 recovery are directionally aligned with implicit basin widening [recovery_impossibility, D25 json].
  2) **Edge-of-stability framing (6.1)** — moderate support through near-critical spectral behavior across runs (`rho` near marginal), but mostly empirical analogical grounding rather than strict cross-domain theorem-to-model transfer.

- **Weakest (reframe or drop as literal):**
  3) **Morphogenetic blueprint analogy (6.5)** — currently metaphorical (interesting biological framing, low direct validation).
  4) **Noise-induced order / recurrence resonance (6.4)** — partially consistent but still speculative without explicit matched perturbation protocol and structural-vs-random split.
  5) **Banach contraction (6.2 / Prop 25)** — **most vulnerable**: currently pre-empirical; D28 is still pending and this is the only hard falsification.

- **Should we pursue/drop?**
  - **Pursue (high priority):** Prop 25 (Banach ratio), because it is the cleanest single mechanism test for T_min universality and can disambiguate T5 interpretations.
  - **Pursue with narrowed scope:** reframe Nishimori-like claims as analogies unless matched by direct measurable thresholds.
  - **Drop as “theory claim,” keep as “heuristic bridge” unless D28 + coherence-gap test confirms:**
    morphogenetic energy-as-blueprint + some cross-domain mappings.

- **Is Banach prediction testing what we think?**
  - Partially: it tests one mathematically sufficient mechanism (geometric, D-independent contraction ratio), but does **not** by itself prove why universality arises.
  - Also confounded by current fixed training horizon (`T_train=10`) in scripts and by local/spectral-vs-global contractive differences (cf. D6/D3b findings).
  - So D28 should be interpreted as a targeted mechanistic probe, not full explanation.

### 3) D25 RECOVERY ANALYSIS

- **Is +27.9% real or artifact?**
  - Real and not a metric artifact by construction: both seeds show **same directional effect** at σ=0.2 (+27.47%, +28.25%, mean +27.86%) from `variable_t_only` [exp_d25_recovery_training.json].
  - Improvement is from a high-error operating point: WA@0 is ~60% at σ=0.2 and drops after perturbation steps, so effect is meaningful as basin widening/corrector behavior, not cosmetic.
  - That said, we only have two seeds and only one completed variant in the file, so this is **preliminary effect size**.

- **Non-monotonic +10 peak then worse at +20**
  - Pattern appears consistent with dynamics that are stable near, but not globally, and with state-dependent contraction:
    - early extra steps pull perturbed trajectories into a valid basin region,
    - later steps overrun into wrong-attractor structure or carry-propagation drift.
  - This aligns with D21/D20-style “wrong-attractor persistence” and with high-iterate degradation observed in other sweeps.

- **How does this change T6?**
  - Raises it from “non-repairable” to “conditional/canonical recovery exists”:
    - **T6 now supported at the narrow operating point:** σ≈0.2 and moderate extra steps (+10 strongest),
    - but still **not evidence for general causal repair** across noise scales or long-horizon invariance.
  - Hence T6 stays moderate-low and bounded.

### 4) D27 IMPLICATIONS

- **Cross-attention +28.4% (99.93 → 71.53 at σ=0 in seed42)**
  - Supports Prop 22 in a qualified way: re-reading encoder context is a substantial recurrent information channel, not a gimmick.
  - It also indicates the decoder is not “single-shot from context”; it exploits iterative refinement.

- **Encoder independence model (E_enc_seq / tok_acc^D)**
  - D27’s severe failure under σ=0.05–0.1 and lack of recovery across T suggest the channel is weak when corrupted and that dynamics cannot always invert it after the fact.
  - Combined with D23-style decomposition evidence, this is still consistent with D23’s encoder-independence framing: dynamics fix much of the carry structure internally, but only when context signal is not too damaged.

- **Attractor interpretation**
  - Noise does not just add random inference error; it appears to move the trajectory into alternate basins:
    - No recovery at σ=0.1 despite more steps,
    - “flattened” curves at noise.
  - This is exactly attractor-language consistent: basin of the correct attractor is narrow and context perturbation can shift the system to a wrong fixed point.

### 5) SINGLE MOST IMPORTANT NEXT LEVER

- **Run/complete D28 with multi-seed, and include a fixed-T control branch in the same script.**
  - A successful D28 run directly tests the core unresolved mechanism layer:
    1) whether `k_t` is geometric and D-independent,
    2) whether Banach-like contraction can explain T=5 saturation without circular threshold dependence,
    3) whether the universality is architecture-induced or merely `T_train=10`/horizon artifact.
  - This one experiment gives the largest confidence gain per run: it sharpens T5, resolves part of T1, and constrains several cross-domain connections at once.

### 6) RISK REGISTER (high-to-high impact)

1. **Residual seeding/sweep fragility**: some earlier sweeps were contamination-prone; still some conclusions lean on partial/early-complete runs (D27 incomplete, D25 partial variants).
2. **Mechanism confounds from fixed-horizon training (`T=10`)**: D23 universality and many D-min claims could be horizon-bias unless fixed-T ablation is added.
3. **Wrong-attractor regime in CE vs E5 overlap**: wrong attractor behavior is robustly observed, so precision gains can hide instability.
4. **Context-noise methodology**: D27 shows catastrophic degradation from tiny σ; small protocol differences (where perturbation is injected, same-step schedule, seeds) can invert conclusions on robustness.
5. **Small-N statistical coverage**: strong claims still often rest on few seeds and one architecture (d_model=128, heads=4, V=64, L mostly ≤24).
6. **Cross-domain over-interpretation risk**: some analogies (Nishimori, morphogenesis) are explanatory metaphors unless matched by direct falsifiable metrics in UESD.

### 7) PUBLICATION READINESS

- **Publishable subset now:** yes, with a conservative scope.
  - Solid core: CE-dynamics robustness + parameter efficiency vs deep encoder, wrong-attractor effects of SC, and finite-step dynamics behavior (D2b, D2d, D3b, D20-class scaling observations).
- **Not yet publish-ready:** global claims of universal T_min mechanism (Prop 24/25), strong E5 superiority, and broad “causal repair” claims.
- **What to strengthen before broader manuscript:**
  1) D28 finished with CI + fixed-vs-variable-T controls,
  2) full D27 seed completion (and preferably local-attention ablation),
  3) explicit recovery protocol with significance across at least 5 seeds and multiple σ bands.

If you want, I can convert this into a compact “evidence matrix + confidence ledger” with a one-page decision matrix (claim → evidence → status → required next test → publication fit).  

