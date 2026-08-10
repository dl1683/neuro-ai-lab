# Repository Status

Audited snapshot: commits `b1a3e5f` through `0ad75b5`, plus the SVAMP fallback implementation and smoke diagnostics described below.

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
| 06 UESD — semantic ratchet | SVAMP FALLBACK IMPLEMENTED / CANONICAL GATE BLOCKED | The primary GSM8K gate remains validly **ABORT-AND-SWAP** at 18/256 correct. The pinned SVAMP path is implemented and passed an 8-example smoke only; the canonical fallback has not run, and no mechanism training is authorized. | `experiments/06_uesd/results/exp_e1_task_band.json`, `experiments/06_uesd/exp_e1_task_band.py`, `experiments/06_uesd/PREREGISTRATION.md` |
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

### Fresh Preregistered Direction

The semantic-ratchet transient-solver hypothesis is preregistered in `experiments/06_uesd/PREREGISTRATION.md`; it does not reopen the D38-D40 fixed-point convergence arc. The primary E1 task-band gate is now **COMPLETE / ABORT-AND-SWAP**: frozen `base-A` scored 18/256 correct (7.03125%), with 238/256 valid extracted incorrect and 0/256 extraction failures. Fifteen of 256 responses (5.859375%) reached the token cap. Leakage preflight and batched-versus-unbatched equivalence passed. Total wall time was 168.23s; peak allocated/reserved VRAM was 1.17/1.45 GiB.

This result is below the preregistered 26-correct band edge and the 40-correct minimum critic population. The frozen mapping is SVAMP below band or GSM-Hard above band, so this result selects the one-time SVAMP fallback. The existing E1 runner now supports a revision-pinned SVAMP train/test snapshot, a seeded 256-of-300 test cohort, five fixed train demonstrations, the frozen parser and thresholds, and the separate immutable target `experiments/06_uesd/results/exp_e1_task_band_svamp.json`. The canonical fallback has not been launched or written. The GSM8K result supplies no evidence for the semantic-ratchet mechanism, and no mechanics or full-program training is authorized yet.

An independently reviewed 8-example SVAMP smoke passed after one diagnostic repair. The first smoke completed generation but failed the existing two-example batched-versus-unbatched equality check: one greedy response changed with padding/co-batch shape. SVAMP evaluation was therefore made per-example (`batch_size=1`) while GSM8K remains frozen at batch size 8; the review found no runner blocker for the smoke rerun. The passing smoke produced 3/8 correct, 4/8 valid extracted incorrect, and 1/8 extraction failures; all 8 stopped at end-of-message, none reached the cap, and both repeated responses and extracted answers matched. It projected 192.23 seconds for 256 examples. These are diagnostics, not task-band evidence. No smoke artifact was written, and the canonical SVAMP gate remains blocked pending its own independent pre-launch review and explicit launch decision.

Operationally, two earlier canonical invocations were killed before GPU use while sandboxed dataset loading stalled on user-profile Hugging Face cache lock files; neither wrote an evidence artifact. The successful canonical attempt kept the frozen runner and parser unchanged, copied the existing base-A and GSM8K cache subtrees into ignored workspace-local `.hf_cache/`, and ran fully offline with `HF_HOME` pointed there.

### Canonical Documents and Evidence

`docs/UNIFIED_ERROR_SPACE.md` (synthesis with current verdict), `experiments/EXPERIMENTS.md` (chronology D1-D40), `experiments/06_uesd/proofs/theory_summary.md` (theorem catalog, supporting only), `experiments/06_uesd/results/` (JSONs + Codex review corpus).

### Validation

```powershell
python experiments\06_uesd\audit_uesd_claims.py
```

### Frozen vs Open

The fixed-point convergence arc is closed. D40 stands at 15/16; rerunning the crashed run (seed 512, λ_sc=3.0) is publication-completeness work, not a blocker to the negative conclusion. The separately preregistered semantic-ratchet direction has not been run and is not evidence. `exp_d14_scaling_laws.py` and `exp_d26_criticality_recovery.py` are reviewed-but-unexecuted designs with no result JSON; do not cite them as empirical evidence.

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

These five commands are the default agent workflow: they read checked-in artifacts only and complete in seconds. The UESD audit currently expects 11/11 checks. There is deliberately no `run_all_pilots.py` and no default full-GPU rerun. D40's runner is an expensive provenance script (~8.4h), not a landing command.

## Update Protocol

- One status owner: this file. No new `FINAL`, `LATEST`, `QUICK_WINS`, or parallel roadmap files.
- New evidence appends `experiments/ledger.jsonl`, updates `experiments/EXPERIMENTS.md`, and revises the canonical synthesis and this file in the same change.
- Deletion is preferred over accumulation; retired generators live in git history.
