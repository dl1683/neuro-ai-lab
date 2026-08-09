# AGENTS.md — Standing Rules for All Agents in This Repository

Every agent (Codex, Claude, or any other) operating in this repo reads this file FIRST and treats it as binding. It encodes the rules that must never be forgotten. `STATUS.md` is the authority on *current state*; this file is the authority on *how agents behave*.

## 1. Read order (mandatory)

1. `AGENTS.md` (this file) — behavioral rules.
2. `STATUS.md` — current state, frozen/open boundaries, verdicts.
3. The canonical doc for your topic (see the Canonical Document Map in `STATUS.md`).

Never act from historical writeups, proofs, old reviews, or design docs alone — they are point-in-time evidence and may contain predictions that later experiments falsified. D40 is the controlling UESD result; the claim audits are the controlling 04 results.

## 2. Landing contract

- Result JSONs and logged-result Markdown writeups are **immutable evidence**. Never edit, regenerate, or relocate them.
- `experiments/EXPERIMENTS.md` (notebook) and `experiments/ledger.jsonl` (machine-readable) are the chronology. If it's not logged, it didn't happen.
- Frozen lines (04 pruning; 01/02/03/05 pilots; the UESD fixed-point arc) are not rerun or extended without an explicit reopening decision recorded in `STATUS.md`.
- Full training scripts are provenance, not landing commands. The default workflow is the five validation commands in `STATUS.md` (seconds, artifact-only).
- New results must update, in the same block of work: result JSON + ledger + EXPERIMENTS.md + canonical synthesis + `STATUS.md`.

## 3. Roles and gates (senior/junior model)

- **Codex (the real `codex` CLI binary — never a simulated review) is the senior architectural authority.** Claude implements, documents, and maintains hygiene. Invocation: `codex exec -s workspace-write --skip-git-repo-check -C "<dir>" -o "<scratch-file>" "<prompt>"`; follow-ups via `codex exec resume --last`. Point Codex at files; never paste file contents into prompts.
- **Design gate:** before non-trivial work, Codex proposes the structure.
- **Pre-launch gate:** before ANY experiment launch, Codex reviews the runner for correctness (this has caught real bugs in D38/D39 — CE-on-padding, flow distribution mismatch, detach errors).
- **Evidence gate:** before any claim, Codex confirms the metrics justify it. Beware vacuous metrics — the canonical lesson is the pre-D40 "0% wrong-attractor" claims, which were meaningless because converged_frac was 0. Every rate must state its denominator.
- **PR gate:** after a coherent block of work, Codex reviews holistically; loop until clean.

## 4. Anti-entropy rules

- No new files unless they create a reusable boundary, prevent duplication, or formalize a stable interface. Extend existing modules instead.
- One canonical doc per topic. No `FINAL`, `LATEST`, `v2`, `QUICK_WINS`, or parallel roadmap/status files — `STATUS.md` is the only status owner.
- Variation belongs in config, not new scripts. One-off analysis/monitoring/babysitting scripts are deleted after use (git history preserves them).
- Deletion is preferred over accumulation. If two files do the same thing, there should be one.
- Growth must increase clarity or capability; otherwise it is failure.

## 5. Research philosophy (the moonshot invariants)

- **Swing for the home run.** Question foundational assumptions; do not farm safe incremental wins or tune around a frame (the V1-V7 selector drift is the recorded counterexample). Rigor is assumed, not traded against risk.
- **The narrative test is a selection criterion, not an afterthought.** Every proposed direction must name the experiment that, if it works, produces a one-sentence story a stranger retells unprompted ("gossip-magazine test") — and that story must survive "isn't that obvious?" and "isn't that trivial?". A direction whose full success would demonstrate nothing story-worthy is mis-designed (recorded counterexample: base-64 addition as the UESD endpoint).
- **Kills are fuel, never destinations.** After a kill: the next question-loop output must propose ≥3 new hypotheses informed by WHY it died. After 3-5 accumulated kills: produce a FAILURE SYNTHESIS (pattern across failures → what the solution space must look like → untested assumptions → predicted-to-work directions). Never respond to a kill by polishing, publishing, or framing the kill as the deliverable.
- **Do not continue closed arcs by inertia.** New UESD work requires a fresh preregistered hypothesis (candidates on record: readout-coupled energy; anytime/transient-solver framing).
- **Steering is a dialogue.** Direction decisions (pivot / kill / which-way-next) require 2-3 genuine rounds with Codex (`resume --last`), never a unilateral call.
- **Evidence standard:** claims no stronger than surviving controls; matched budgets for comparisons; negative results are knowledge; costs and hardware reported.

## 6. Git and operational discipline

- Commit per logical change; message = one clear idea + `Committed by Devansh`.
- Never commit: datasets, checkpoints (`*.pt`), raw run logs, `.claude/`, `__pycache__` (all gitignored).
- OpSec: commit messages, branch names, and public docs never reveal model names, provider routing, or cost data.
- Before deleting or overwriting anything tracked, confirm it is not immutable evidence (Section 2).
- Hardware: single RTX 5090 (24 GB) is the default constraint; target ~80% utilization for long runs; quantize local LLMs unless precision is required; log wall time in the ledger.

## 7. Validation (run before and after any block of work)

```powershell
python experiments\04_criticality_pruning\synthesize_synflow_pathology.py
python experiments\04_criticality_pruning\audit_synflow_pathology.py
python experiments\04_criticality_pruning\synthesize_path_capacity.py
python experiments\04_criticality_pruning\audit_circuit_viability_claims.py
python experiments\06_uesd\audit_uesd_claims.py
```

All must pass; expected pass counts are recorded only in `STATUS.md`. A failing audit blocks any claim and any commit that touches evidence.
