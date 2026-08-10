# Repository Status

Audited snapshot through the terminal base-A/SVAMP adjudication, the single canonical base-B successor-gate `PASS`, and the single E2 mechanics-pilot attempt with a controlling post-evidence `VOID`. The immutable E2 artifact retains the runner-emitted `FAIL`, which the evidence gate found inadmissible because the preregistered 200-pair-per-seed denominator floor was 0/200 in both seeds. The base-B artifact is immutable band-placement evidence only; E2 is mechanics design-gate evidence only.

**This file is the sole authority for current state.** Experiment writeups, proofs, and reviews are point-in-time evidence; earlier reviews and evidence gates may contain predictions that later experiments falsified.

## Agent Landing Contract

1. Read `STATUS.md` (this file) before `README.md`, historical plans, proofs, or reviews.
2. Result JSONs and logged-result Markdown writeups are immutable evidence. Do not edit, regenerate, or relocate them.
3. `experiments/EXPERIMENTS.md` is the chronological lab notebook and `experiments/ledger.jsonl` the machine-readable run ledger. Neither is the current-status authority; this file is.
4. Frozen lines are not to be rerun or extended without an explicit reopening decision recorded here.
5. New results must update, in the same block of work: the result JSON, the ledger, the canonical synthesis doc, the relevant audit, and this file.
6. Full training scripts are provenance and must not be launched automatically. The default agent workflow is the lightweight validation commands below.

## Current Research Lines

| Line | State | Current verdict | Canonical evidence |
|---|---|---|---|
| 04 Circuit-viability pruning | FROZEN / AUDITED | Claim audit `234/234`; SynFlow pathology audit passes; mechanism supported within recorded scope. | `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md`, `docs/CLAIM_EVIDENCE_LEDGER.md`, `docs/CLAIM_AUDIT.md`, `results/04_criticality_pruning/` |
| 06 UESD — fixed-point arc | CONVERGENCE ARC CLOSED NEGATIVE | D40 falsifies the tested self-consistency fixed-point thesis; the system behaves as a finite-time transient solver. Surviving core: D22 variable-T k-suppression. | D40 ledger entry, `experiments/06_uesd/results/exp_d40_extended_convergence.json`, `docs/UNIFIED_ERROR_SPACE.md` |
| 06 UESD — semantic ratchet | BASE-A TASK-BAND TERMINAL VOID / BASE-B TASK-BAND PASS / E2 COMPLETE — CONTROLLING VOID | Base-B satisfied the separate band-placement prerequisite. E2's immutable runner artifact emitted `FAIL`, but both seeds supplied 0/200 required confidence-matched pairs; the registered denominator floor and `VOID > FAIL` precedence therefore control. Official test was not inspected. The 0.5B launch remains blocked, and no automatic Direction 2 route was unlocked. | `experiments/06_uesd/results/exp_e2_latch_mechanics.json`, `experiments/06_uesd/results/exp_e1_task_band_base_b.json`, prior base-A artifacts, `experiments/06_uesd/PREREGISTRATION.md` |
| 07 Best-of-N safe selection | PREFLIGHT STOP / CANONICAL NOT RUN | The runner, local manifest, cohort bindings, and four launch-blocking hash slots are complete. The retained 2-problem × 16-candidate calibration smoke passed generation/scoring integrity, but measured generation projects the 12,288-response bank at 11.75 GPU-hours versus the 2.5-hour cap. Work stopped as registered. A reviewed pre-data resize amendment and independent pre-launch review are required before any canonical launch. | `experiments/07_safe_selection/PREREGISTRATION.md`, `experiments/07_safe_selection/exp_f1_bon_safe_selection.py` |
| 01/02/03/05 pilots | FROZEN HISTORICAL | Single-run pilots; no active roadmap or runner. 01 failed; 02/03 modest wins on toy tasks; 05 efficiency win with failed DDM fit. | `results/01..05/pilot_result.json`, `results/pilot_suite_summary.json`, frozen-pilots table in `experiments/EXPERIMENTS.md` |

## 04 — Circuit-Viability Pruning

### Verdict

Severe global pruning can create hidden circuit cutsets: global SynFlow at 98-99% sparsity allocated zero weights to the dense classifier bridge in 3/3 CNN cases (mean after-fine-tuning delta vs magnitude: -42.80 points), and masked fine-tuning cannot recover. Capacity-reserve constraints rescue the collapse and beat magnitude in several severe-sparsity regimes (best: +6.57 points, full CIFAR-10 ResNet-20-style, 99%, 4/4 seeds). The evidence supports a hierarchy: preserve liveness first, then route-family balance, then degeneracy control.

### Supported Claim Boundary

Use only the public claim recorded in `docs/CLAIM_EVIDENCE_LEDGER.md`:

> Severe pruning can create hidden circuit cutsets. Capacity constraints inspired by circuit viability prevent these failures and can improve extreme-sparsity recovery across CNN and residual settings. The evidence supports a hierarchy: preserve liveness first, then route-family balance and degeneracy.

Do not claim: pruning is solved; route-capacity always beats magnitude; current penalty weights are theoretically derived; robust transformer or large-model transfer has been shown.

### Known Limitations

- All results are small-model scale (small CNNs, ResNet-20-class, TinyViT). The pretrained-ResNet-18/TinyImageNet external test mostly failed and was rescued only to magnitude parity.
- Selector generations V1-V7 were developed against the same growing seed pool they are evaluated on; V7's 15/15 oracle match is not prospective evidence of generality.
- Diversity/tradeoff penalty weights are hand-set, not derived.

### Canonical Documents

`docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md` (scientific synthesis, includes the neuroscience framing and route-deficit predictor appendices), `docs/CLAIM_EVIDENCE_LEDGER.md` (claim scope), `docs/CLAIM_AUDIT.md` (generated audit).

### Validation

```powershell
python experiments\04_criticality_pruning\synthesize_synflow_pathology.py
python experiments\04_criticality_pruning\audit_synflow_pathology.py
python experiments\04_criticality_pruning\synthesize_path_capacity.py
python experiments\04_criticality_pruning\audit_circuit_viability_claims.py
```

### Frozen vs Open

The line is frozen. Open questions (recorded, not active work):

1. Derive diversity/concentration penalty weights from measured route-family sensitivity instead of hand-setting them.
2. A predictive route-quality theory: rank masks before fine-tuning in the same order as after-fine-tuning accuracy, across methods and seeds.
3. Externally robust transfer (pretrained and transformer settings) beyond magnitude parity.

## 06 — Unified Error-Space Dynamics

### Verdict

- D40 (15/16 runs; the missing run is a CUDA crash, not a scientific gap): **key negative.**
- Correct outputs occur only in the trained finite-T compute window (VT range [4-16]): 99.90-100% per-run accuracy at T=10, collapse beyond T≈25-50.
- Stronger self-consistency (λ_sc) lowers residual monotonically in group means AND drives faster long-horizon accuracy collapse. When convergence finally occurs (λ_sc=3.0, seed 42: 91% converged at T=200), ~100% of converged examples are wrong attractors with 0% accuracy.
- CE-only (λ_sc=0) is the best long-horizon configuration; nothing in the SC stack improves any accuracy metric on the tested task.

### Defensible Surviving Core

D22 variable-T training (compute-window robustness: T=32 accuracy 88.5% -> 99.9%), with its mechanism established by D28-D37: contraction-rate suppression (Δk p=1.7e-5, 8/8 seeds at D=8), replicated across depths D=6-10, a non-arithmetic task (prefix-sum), and 6/6 d=128 pairs across three tested architectures — 48/51 pairs in the predicted direction across the full sweep; confidence 9/10. This is an anytime-solver robustness result. **It is not evidence for correct fixed-point convergence.**

### Rejected or Closed Claims

- `E(s)=||F_theta(s,c)||^2` is not a semantic correctness energy: its attractors do not decode to correct outputs.
- Early-iteration contraction rate k does not establish endpoint convergence (D39's T≈22-36 extrapolation falsified by D40).
- All pre-D40 "0% wrong-attractor" measurements were vacuous (converged fraction was 0).
- "Dynamics are necessary for addition" is weakened to parameter efficiency (larger encoders learn it).
- The thinking-generating continuum (confidence 2.5/10; dropped).

### Open Only If Explicitly Reframed

- Readout-coupled energy design (attractor = correct output, not state self-consistency).
- Anytime/transient-solver framing built on the D22 mechanism.
- Either path requires a fresh preregistered hypothesis. **Do not continue the old fixed-point arc (no "D41 by inertia").**

### Semantic-Ratchet Direction and Gates

The semantic-ratchet transient-solver hypothesis is preregistered in `experiments/06_uesd/PREREGISTRATION.md`; it does not reopen the D38-D40 fixed-point convergence arc. The primary E1 task-band gate is now **COMPLETE / ABORT-AND-SWAP**: frozen `base-A` scored 18/256 correct (7.03125%), with 238/256 valid extracted incorrect and 0/256 extraction failures. Fifteen of 256 responses (5.859375%) reached the token cap. Leakage preflight and batched-versus-unbatched equivalence passed. Total wall time was 168.23s; peak allocated/reserved VRAM was 1.17/1.45 GiB.

This result is below the preregistered 26-correct band edge and the 40-correct minimum critic population. The frozen mapping selected the one-time SVAMP fallback. Its canonical-mode initial-parser attempt scored 66/256 correct (25.78125% of 256), 177/256 valid extracted incorrect (69.140625% of 256), and 13/256 model-empty non-answers and extraction failures (5.078125% of 256), for 190/256 total exact-answer failures (74.21875% of 256). Zero of 256 responses reached the token cap. Total wall time was 208.19s; peak allocated/reserved VRAM was 0.78/0.84 GiB (842,769,408/901,775,360 bytes).

The correct count is inside the preregistered 26–217 band and both response populations exceed 40, but the extraction-failure rate is one example above the 5.000% ceiling. All 13 failed outputs are exactly `Answer:` with zero numeric content. The governing parser-repair-scope amendment therefore exhausts the repair branch as inapplicable: the base-A/SVAMP fallback gate is terminal **VOID** with reason `NO_RECOVERABLE_NUMERIC_CONTENT_FOR_PERMITTED_PARSER_REPAIR`. No parser change or repeated generation occurred or is permitted. The immutable initial-miss artifact remains unchanged with its point-in-time **PARSER-REPAIR-REQUIRED** status. This is band-placement evidence for frozen `base-A`, not a capability claim or evidence for the semantic-ratchet hypothesis.

The single frozen `base-B` successor gate is now **COMPLETE / PASS** on exactly 256 revision-pinned GSM8K official-test examples selected by the precommitted deterministic construction, disjoint from the base-A cohort, with no fallback. The four-category accounting is 143/256 correct numeric (55.859375%), 104/256 valid extracted incorrect (40.625%), 0/256 model-empty non-answers (0%), and 9/256 parser-recognition failures (3.515625%); the counts exhaust 256/256. The correct count is inside 26–217 and at least 40, usable incorrect is 104, both failure categories are at most 12/256, and leakage, provenance, repeat-determinism, cohort, and accounting checks pass. This is band-placement evidence only, not a capability result and not evidence for the semantic-ratchet mechanism.

The immutable artifact is `experiments/06_uesd/results/exp_e1_task_band_base_b.json` (checkout-stable canonical-LF SHA-256 `9c57a10f3aa64c43fa34819255f4cf4e004cc5ab74afaa838c02c4a06a271de2`; original runtime-byte SHA-256 `2c78f7ad83e6481a6abbdf82146945da5592cef24b91ea0bb1f007a424a3f0ca`). Total wall time was 21,412.46s (5h56m52.46s), generation wall time 21,391.56s, observed throughput 1.87186 generated tok/s, and peak allocated/reserved VRAM 3,278,870,016/3,716,153,344 bytes. Power telemetry varied during active generation; throughput is not frozen and the scientific protocol did not change. The wall time exceeded the 1.3–3.5h estimate, but no genuine stall occurred.

The 30M shortcut-resistant synthetic-deduction mechanics pilot remained formally **INDEPENDENT** of base-A, base-B, GSM8K, SVAMP, and the real-task task-band gate and is now **COMPLETE / CONTROLLING POST-EVIDENCE VOID** after the single reviewed attempt. The reviewed runner/config were unchanged from attested commit `fba181cd30f6465ce4149c10fd08eab459e1b0bf`; the run used the required `INDEPENDENT_PRETRAINING_REVIEW_CLEAN` launch token. Both seeds completed their 500,000-token common-controller and encoder-control budgets, selector fitting, and calibration. Seed 42 critic AUROC was 25,183,159/50,331,648 positive-negative pair units = 0.5003444155 over 4,096 positive and 12,288 negative states; confidence and on-policy critic AUROCs used the same 50,331,648-pair denominator and were 0.4999594887 and 0.5003444155. Seed 31415 critic AUROC was 25,275,202.5/50,331,648 = 0.5021731555 over the same 4,096/12,288 state populations; confidence and on-policy critic AUROCs, again over 50,331,648 pairs, were 0.4996897976 and 0.5021731555. Critic-minus-confidence advantages were 0.0003849268 and 0.0024833580 against 0.05, and critic-versus-confidence selection advantage was 0/1,024 in both seeds against 3 points.

Confidence-matched concordance was undefined with 0 qualifying pairs in each seed. This missed the preregistered minimum of 200 pairs per seed, a competence floor that must be satisfied before any `PROCEED`/`FAIL`; the configuration also freezes `VOID > FAIL > PROCEED` precedence. The runner's pretest branch failed to apply that denominator floor and emitted `FAIL / PRETEST_SELECTOR_PROVENANCE_GATE_MISSED`. The JSON is immutable and retains that raw output, but the controlling post-evidence adjudication is **`VOID / INSUFFICIENT_CONFIDENCE_MATCHED_PAIRS_PER_SEED`**, observed 0/200 in 2/2 seeds. The measurable near-chance selector results remain diagnostics, not an admissible `FAIL` or KILL.

Calibration no-latch accuracy was exactly 256/1,024 in each seed at every candidate horizon `T=1..15`, so the shared tie-break froze `t*=1` at pooled 512/2,048 and observed calibration gain was 0/1,024 per seed. Both informational arm-4 selectors froze `delta=0` with `calibration_constraint_miss=true`. The pretest branch stopped before official-test inspection. Therefore endpoint regression, regression reduction, gain retention, endpoint selector accuracy difference, matched-pair AUROC, competence floors, encoder-control test accuracy, arm-4 test summaries, and `O/F/H` headroom are not applicable rather than zero. The immutable artifact is `experiments/06_uesd/results/exp_e2_latch_mechanics.json` (checkout-stable canonical-LF SHA-256 `7842ca6f69ba3885fe7b03142b694e9c95950f195d31acd73f601d1e3f5a4075`). Artifact wall time was 1,181.121863s (19m41.12s); peak allocated/reserved VRAM was 1,076,839,936/1,256,194,048 bytes. The launch remained on AC/High performance with no power event or protocol deviation.

The current 0.5B semantic-ratchet program remains **LAUNCH BLOCKED BY E2 VOID**. The attempt did not satisfy the mechanics prerequisite, but its denominator failure also did not produce the registered `FAIL` that would automatically route the next full experiment to Direction 2. Direction 2 causal isolation and E2-CERT remain possible subjects for a fresh steering decision; neither is unlocked, authorized, or allowed to rescue or reinterpret this attempt. No E2 rerun, threshold change, or adaptive mechanics redesign is authorized without an explicit reopening decision recorded here.

Operationally, two earlier GSM8K canonical invocations were killed before GPU use while sandboxed dataset loading stalled on user-profile Hugging Face cache lock files; neither wrote an evidence artifact. The successful GSM8K attempt and the SVAMP initial attempt used the ignored workspace-local `.hf_cache/` fully offline. The SVAMP loader began GPU work immediately, so its 10-minute watchdog did not trigger.

### Canonical Documents and Evidence

`docs/UNIFIED_ERROR_SPACE.md` (synthesis with current verdict), `experiments/EXPERIMENTS.md` (chronology D1-D40), `experiments/06_uesd/proofs/theory_summary.md` (theorem catalog, supporting only), `experiments/06_uesd/results/` (JSONs + Codex review corpus).

### Validation

```powershell
python experiments\06_uesd\audit_uesd_claims.py
```

### Frozen vs Open

Current runner states (2026-08-10): E2 mechanics pilot COMPLETE / raw artifact `FAIL` / controlling post-evidence `VOID` after the single independently reviewed canonical attempt. Official test was not inspected. Base-B successor gate remains COMPLETE / PASS after its single reviewed canonical attempt. The current 0.5B semantic-ratchet launch is blocked; no full-experiment route is automatically unlocked by this `VOID`. The immutable E2 artifact must not be edited, regenerated, or relocated; local checkpoints are provenance only and do not authorize a rerun.

The fixed-point convergence arc is closed. D40 stands at 15/16; rerunning the crashed run (seed 512, λ_sc=3.0) is publication-completeness work, not a blocker to the negative conclusion. The base-A GSM8K task-band gate completed as ABORT-AND-SWAP, and its SVAMP fallback is terminal VOID under the parser-repair-scope amendment. The disjoint base-B successor gate completed as PASS; this satisfies only the band-placement prerequisite. The independent E2 mechanics attempt is controlling `VOID` because its 0/200 per-seed matched-pair populations missed a prerequisite floor; the registered full semantic-ratchet program was not run and cannot launch. Do not describe uninspected endpoint quantities as zero or failed, and do not treat line 07 or E2-CERT as a rescue. `exp_d14_scaling_laws.py` and `exp_d26_criticality_recovery.py` are reviewed-but-unexecuted designs with no result JSON; do not cite them as empirical evidence.

## Canonical Document Map

| Topic | Canonical file |
|---|---|
| Current repository state | `STATUS.md` |
| Project orientation | `README.md` |
| 04 scientific synthesis | `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md` |
| 04 claim scope | `docs/CLAIM_EVIDENCE_LEDGER.md` |
| 04 generated audit | `docs/CLAIM_AUDIT.md` |
| UESD scientific synthesis | `docs/UNIFIED_ERROR_SPACE.md` |
| UESD semantic-ratchet preregistration | `experiments/06_uesd/PREREGISTRATION.md` |
| E2 optimizer's-curse secondary analysis | `experiments/06_uesd/E2_SECONDARY_ANALYSIS_CURSE.md` (informational only; subordinate to E2 adjudication) |
| Best-of-N safe-selection preregistration | `experiments/07_safe_selection/PREREGISTRATION.md` |
| Experiment chronology | `experiments/EXPERIMENTS.md` + `experiments/ledger.jsonl` |
| UESD theorem catalog | `experiments/06_uesd/proofs/theory_summary.md` (supporting only) |
| Raw evidence | `results/` for 01-05 and 04; `experiments/06_uesd/results/` for 06 (do not relocate) |

## Canonical Validation Commands

```powershell
python experiments\04_criticality_pruning\synthesize_synflow_pathology.py
python experiments\04_criticality_pruning\audit_synflow_pathology.py
python experiments\04_criticality_pruning\synthesize_path_capacity.py
python experiments\04_criticality_pruning\audit_circuit_viability_claims.py
python experiments\06_uesd\audit_uesd_claims.py
```

These five commands are the default agent workflow: they read checked-in artifacts only and complete in seconds. The UESD audit currently reports that all checks pass, with the count sourced from the audit output. There is deliberately no `run_all_pilots.py` and no default full-GPU rerun. D40's runner is an expensive provenance script (~8.4h), not a landing command.

## Update Protocol

- One status owner: this file. No new `FINAL`, `LATEST`, `QUICK_WINS`, or parallel roadmap files.
- New evidence appends `experiments/ledger.jsonl`, updates `experiments/EXPERIMENTS.md`, and revises the canonical synthesis and this file in the same change.
- Deletion is preferred over accumulation; retired generators live in git history.
