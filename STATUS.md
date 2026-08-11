# Repository Status

Audited snapshot through the terminal base-A/SVAMP adjudication, the single canonical base-B successor-gate `PASS`, the single E2 mechanics-pilot attempt with a controlling post-evidence `VOID`, the completed E2-DIAG Stage-0 `STOP`, the terminal E3 preflight `PREFLIGHT_STOP`, and the E3 interface-supervision diagnostic with blocking review findings repaired, bindings and projection landed, no cell run, and launch still blocked on independent re-review. The base-B artifact is immutable band-placement evidence only; E2, E2-DIAG, E3 preflight, and the new diagnostic are diagnostic/design-gate surfaces only.

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
| 06 UESD — semantic ratchet | BASE-A TASK-BAND TERMINAL VOID / BASE-B TASK-BAND PASS / E2 CONTROLLING VOID / E2-DIAG STAGE-0 STOP / E3 PREFLIGHT STOP / INTERFACE-SUPERVISION DIAGNOSTIC REVIEW-FINDINGS-REPAIRED-HASH-BOUND-NOT-RUN | Base-B satisfied the separate band-placement prerequisite. E2 remains controlling `VOID`; E2-DIAG killed the from-scratch line. E3 stopped at `PREFLIGHT_STOP / PRETRAINED_INTERFACE_COMPETENCE_SMOKE_MISS`. The six blocking diagnostic-review findings are repaired, the strengthened payloads are hash-bound, and the GPU-free six-cell projection is 2,180/5,400 seconds. Cell I0 remains blocked on independent re-review and a clean-launch attestation. No diagnostic cell, E3 canonical stage, official test, or 0.5B launch is authorized yet. | `experiments/06_uesd/E3_INTERFACE_SUPERVISION_DIAGNOSTIC_PREREGISTRATION.md`, `experiments/06_uesd/exp_e3_interface_supervision_diagnostic.py`, `experiments/06_uesd/exp_e3_interface_supervision_diagnostic_projection.json`, `experiments/06_uesd/results/exp_e3_preflight.json`, `experiments/06_uesd/results/exp_e2_diag_stage0_instrumented.json`, `experiments/06_uesd/results/exp_e2_latch_mechanics.json`, `experiments/06_uesd/results/exp_e1_task_band_base_b.json` |
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

**E2-DIAG Stage 0 completed with registered `STOP` and routes `KILL_FROM_SCRATCH_LINE`.** T=4 training accuracy remained 32/128 at initialization and every 100-update checkpoint through 3,000 updates, below the frozen 122/128 gate. No later diagnostic stage, optimizer cell, model repair, selector path, or official test was authorized or accessed. Informational instrumentation found nonzero gradients throughout every registered parameter group, a 1.925192 L2 readout-weight change, and 3,000/3,000 clipped updates against a clip norm of 1.0; global pre-clip gradient norm had median 8.4168 and p95 15.5301. The all-one-class collapse oscillated between classes while full-set CE remained near `ln(4)`. This supports an **optimization-saturated training-regime interpretation** for the tested substrate and budget, not a wiring-bug diagnosis. The interpretation is successor-design fuel only and does not reopen the killed line.

The repaired invocation completed 3,000 updates over 2,786,051 non-padding input-plus-answer tokens in 98.873 GPU-seconds, with peak allocated/reserved VRAM of 637,935,104/696,254,464 bytes. The repaired immutable artifact is `experiments/06_uesd/results/exp_e2_diag_stage0_instrumented.json` (canonical-LF SHA-256 `0ad4fc5fafe343b37944b462972d32ff201749280a7bdbb506fa3b62536620d7`). The first operational-VOID artifact remains immutable at `experiments/06_uesd/results/exp_e2_diag.json` (canonical-LF SHA-256 `41c035dd79fe42e9edb891dcc02664077c7d491cd7a2857fe7cc9ae16dc5ab37`). The original E2 controlling `VOID`, blocked 0.5B launch, absence of an automatic Direction 2 route, and line-07 priority remain unchanged.

See the explicitly non-normative [`Failure Synthesis Retrospective — 2026-08-10`](docs/UNIFIED_ERROR_SPACE.md#12-failure-synthesis-retrospective-2026-08-10) for hypothesis-generation context subordinate to this status.

E3 is **PREFLIGHT STOPPED / CANONICAL NOT RUN**. The outcome-blind instrument probe completed exactly 50 throwaway updates with finite positive global pre-clip norms (median 13.6205, p95 23.4852) and the frozen mapping selected `C=14`. The one permitted exact-architecture smoke then completed exactly 500 updates and scored 128/512 = 25.0% at `T=4` (95% Wilson interval 21.4448%–28.9275%), below the registered 205/512 floor. Its training CE by 50-update window moved from 1.6670 to 1.4062 without above-chance validation competence.

The controlling outcome is **`PREFLIGHT_STOP / PRETRAINED_INTERFACE_COMPETENCE_SMOKE_MISS`**. It blocks the full E3 spend and authorizes no repeat, alternate horizon, changed interface, dense supervision, extra update, retained checkpoint, selector/calibration/test inspection, or canonical launch. It supports only the narrower statement that this registered representation/controller interface did not clear its pre-spend competence floor; it does not show that every frozen representation fails, that dense supervision is necessary, or that the frozen substrate lacks deduction knowledge.

The routed E3 interface-supervision diagnostic is now **PREREGISTERED / BLOCKING-REVIEW FINDINGS REPAIRED AND HASH-BOUND / NOT RUN / LAUNCH BLOCKED ON RE-REVIEW**. Its frozen six-cell grid tests three direct frozen-representation probes and three controller/supervision cells, with 900 seconds per cell and 5,400 seconds total. The runner, config, clarified registration protocol, reusable E2/E3 sources, template inventory, strengthened proof/candidate contract, tokenizer-containing substrate digest, prompt serializations, and six ordered payloads are SHA-256-bound. The all-record CPU self-test checks 3,584 miniature prompt rows, 512 process rows, 7,168 candidates, every route fixture, optimizer boundaries, and post-sync cap enforcement. The GPU-free projection assigns `150/140/850/80/260/700` seconds to I0/I1/I2/S0/S1/S2, totaling 2,180/5,400 seconds. No diagnostic metric or result exists. Cell I0 cannot launch until an independent re-review accepts the repairs and projection and an explicit clean-launch attestation is bound. This registration does not reopen E3, change its preflight stop, access line 07 or official tests, or unblock the 0.5B program.

The immutable artifact is `experiments/06_uesd/results/exp_e3_preflight.json` (SHA-256 `51b7e565fe13967948017c88b7df2ea4738cb35fdc2dba92a402673698130c9a`). Overall wall time was 104.162s; smoke wall time was 72.015s against 900s. Peak allocated/reserved VRAM was 3,453,325,824/3,873,439,744 bytes. Exactly 550 throwaway optimizer updates and zero canonical updates completed. All throwaway state was discarded; zero line-07, additional GSM8K/SVAMP official-test, or E2-test rows were accessed. The 0.5B launch remains blocked and line 07 remains independent.

**One-time queue recalibration (2026-08-11):** the already-in-flight E3 interface-supervision diagnostic may finish first to a routed verdict under its frozen six-cell protocol. This is a closure exception only: no E3 successor may leapfrog line 07, and line 07 resumes with uninterrupted priority immediately after the diagnostic lands or is frozen as not launchable. E3 permits one full independent re-review and at most one bounded local correction pass followed by delta re-review; a further design-level shortcut requiring dataset, label, or causal-comparison reconstruction bottoms out the diagnostic without a scientific conclusion and transfers the writer slot to line 07. Owner sweeps run on a 30-minute cadence as status heartbeats, not priority resets or automatic deep-review cycles.

**Line-07 verifier ruling:** reproduce the frozen golden replay in an isolated environment pinned to the compatible Transformers release supported by the published verifier implementation, preserving the exact frozen input, tokenizer, checkpoint, and BF16 conditions. Active diagnosis is bounded by 90 minutes or two predeclared compatibility-stack attempts, whichever comes first; the observed threshold is not relaxed. If the pinned path cannot pass, stop at a preflight implementation discrepancy. No verifier swap is authorized without fresh steering and preregistration.

Operationally, two earlier GSM8K canonical invocations were killed before GPU use while sandboxed dataset loading stalled on user-profile Hugging Face cache lock files; neither wrote an evidence artifact. The successful GSM8K attempt and the SVAMP initial attempt used the ignored workspace-local `.hf_cache/` fully offline. The SVAMP loader began GPU work immediately, so its 10-minute watchdog did not trigger.

### Canonical Documents and Evidence

`docs/UNIFIED_ERROR_SPACE.md` (synthesis with current verdict), `experiments/EXPERIMENTS.md` (chronology D1-D40), `experiments/06_uesd/proofs/theory_summary.md` (theorem catalog, supporting only), `experiments/06_uesd/results/` (JSONs + Codex review corpus).

### Validation

```powershell
python experiments\06_uesd\audit_uesd_claims.py
```

### Frozen vs Open

Current runner states (2026-08-11): E2 mechanics pilot COMPLETE / raw artifact `FAIL` / controlling post-evidence `VOID` after the single independently reviewed canonical attempt. E2-DIAG Stage 0 COMPLETE / `STOP` / `KILL_FROM_SCRATCH_LINE` after the single owner-authorized repair. E3 preflight COMPLETE / `PREFLIGHT_STOP`; canonical E3 is not run and cannot launch. The E3 interface-supervision diagnostic has all six blocking review findings repaired, is HASH-BOUND / PROJECTED / NOT RUN, and remains launch-blocked on independent re-review and clean attestation. No E2/E2-DIAG/E3 repeat, later stage, resume, official-test inspection, or 0.5B launch is authorized. The immutable E2, E2-DIAG, and E3 preflight artifacts must not be edited, regenerated, or relocated.

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
| E2-DIAG competence/learnability preregistration | `experiments/06_uesd/E2_DIAG_PREREGISTRATION.md` |
| E3 pretrained-substrate mechanics preregistration | `experiments/06_uesd/E3_PRETRAINED_MECHANICS_PREREGISTRATION.md` |
| E3 interface-supervision diagnostic preregistration | `experiments/06_uesd/E3_INTERFACE_SUPERVISION_DIAGNOSTIC_PREREGISTRATION.md` |
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
