# Codex Synthesis Agent: Unified Strategic Direction for UESD Research

You are the synthesis agent for a multi-agent research survey. Your job is to read
ALL of the following source documents and produce ONE unified strategic document
that identifies the single most promising research direction and a concrete next
experiment.

## Source Documents to Read

1. **_synthesis_input.md** (this directory) — Combined findings from two Claude
   research agents: (a) web research covering 9 papers (2025-2026), and (b) repo
   survey covering 12 cross-domain findings from the _meta and Open Exploration
   research repositories. Also contains the current empirical state of UESD
   (confirmed results, weak/stalled areas, blind spots).

2. **_codex_meta_review.md** (this directory) — Codex review of the _meta
   research repository (C:\Users\devan\OneDrive\Desktop\Projects\_meta), which
   contains theoretical frameworks, cross-domain connections, and research notes.

3. **_codex_openexp_review.md** (this directory) — Codex review of the Open
   Exploration repository (C:\Users\devan\OneDrive\Desktop\Projects\Market Reports\Open Exploration),
   which contains broader research explorations and paper analyses.

4. **theory_summary.md** (this directory) — The current UESD theory state:
   34 propositions, calibration scores, confirmed/falsified predictions.

5. **bottleneck_depth_scaling.md** (this directory) — The deepest ongoing
   theoretical analysis (Prop 30-34, rho scaling, strain model, solvability).

## Current Empirical State (Critical Context)

The D28 contraction ratio experiment is now 11/12 complete. Key results:

**FT rho(D) sequence (COMPLETE):**
D=2: 1.0018, D=4: 1.0026, D=6: 1.0024, D=8: 1.0016, D=10: 1.0042, D=12: 1.0039

Key pattern: FT rho PEAKS at D=10 and declines at D=12. Non-monotonic with
anomalous dip at D=8.

**VT constant-delta_rho (4/5 depths confirmed):**
delta_rho = {-0.0024, -0.0025, -0.0028, +0.0014, -0.0025}
             D=2      D=4      D=6      D=8(!)   D=10

Mean (excl D=8): -0.0026 +/- 0.0002. D=8 is 20-sigma outlier.

**D30 rho monotonic sequence (VT regularization strength):**
T_min=2: rho=0.9992, T_min=4: 1.0001, T_min=6: 1.0006, T_min=8: 1.0017

**Confirmed high-confidence findings:**
- Readout-stable manifold (NOT fixed point) — Prop 28
- FTLE decomposition: lambda_R < 0, lambda_null > 0 — Prop 31, D29b
- T_99 = max(T_min, D_intrinsic) — Prop 32
- Variable-T enables unlimited depth extrapolation (T=48: 99.95% at D=12) — D22
- Subtraction transfer 24x speedup from addition pretraining — D24
- Encoder completely fails at D=12 (0% accuracy), dynamics achieve 99.6% — D23

**Stalled/weak:**
- Recovery (T6): 2/10, stalled since D25
- Strain model (Prop 30): 4/10, multiple falsifications
- Single task (addition only), single scale (d=128, V=64)
- No information-theoretic measurements (MI, Fisher, IB)

## Your Task

Produce a document with the following sections:

### 1. Signal Extraction
What are the TOP 5 signals from across ALL source documents that are most
relevant to UESD's next research direction? For each signal, cite the specific
source and explain WHY it matters.

### 2. Pattern Recognition
What patterns emerge when you combine findings across all sources that NO
SINGLE SOURCE captures alone? Look for:
- Cross-domain convergences (same structure appearing in different fields)
- Contradictions between sources
- Gaps that multiple sources independently identify

### 3. The One Experiment
If we could run EXACTLY ONE experiment next (after D31 multi-seed replication),
what should it be? Requirements:
- Must be computationally feasible on a single RTX 5090
- Must test something PARADIGM-SHIFTING (not incremental rho measurement)
- Must have clear success/failure criteria
- Must connect to at least 2 of the sources above
- Must address one of the identified blind spots

Describe: what to measure, what to predict, what it means if the prediction holds,
what it means if it fails.

### 4. Theory Consolidation
Our theory has 34 propositions. This is too many. Propose a MINIMAL set of
core principles (3-5) from which the confirmed results follow. Which propositions
are actually the same thing viewed from different angles?

### 5. Risk Assessment
What is the SINGLE BIGGEST RISK that our entire research program is an artifact
of the specific task/architecture/scale? How would we test for this?

### 6. Publication Strategy (Ambitious Version)
Not "what's safe to publish" but "what would be the most impactful paper we
could write with 2 more months of experiments?" What claim would make this
work significant to the ML community?

## Output Format

Write your synthesis to a file. Be specific and concrete — cite exact rho values,
proposition numbers, paper titles, and experiment IDs. No hand-waving.
Total length: 2000-3000 words.
