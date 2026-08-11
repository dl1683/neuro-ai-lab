# E3 Pretrained-Substrate Latch-Mechanics Preregistration

Status: **PREREGISTERED / NOT RUN / BUILD AUTHORIZED / LAUNCH BLOCKED ON IMPLEMENTATION, HASH BINDINGS, PREFLIGHT, AND INDEPENDENT REVIEW**

Experiment ID: `exp_e3_pretrained_latch_mechanics`

Preflight ID: `exp_e3_preflight`

This is a fresh successor registration after E2-DIAG assigned
`KILL_FROM_SCRATCH_LINE` to the current from-scratch 30M E2 substrate and
training pipeline. It does not reopen E2, E2-DIAG, or the D38-D40 fixed-point
arc. It cannot alter their immutable artifacts or verdicts.

E3 tests a materially different substrate: one frozen pretrained text model
with a separately trained pilot-scale recurrent controller, answer readout,
and latent critic. E3 is a mechanics gate, not the 0.5B semantic-ratchet
experiment and not publishable confirmation of the full hypothesis.

## 1. Hypothesis and narrative test

### Hypothesis

A frozen pretrained text substrate can supply enough representation and
deduction competence for a small recurrent controller to produce informative
horizon-dependent answer trajectories, allowing a provenance-gated latent
critic to preserve earlier correct states better than confidence plus schedule
alone.

### Conditional one-sentence story

> A frozen language model learned no new language weights, yet a small latent
> controller learned when its own extra computation had gone wrong and retained
> the earlier correct state.

This story survives only if every competence, denominator, provenance,
regression, gain-retention, and selector-comparison gate below passes. Merely
beating chance, learning the synthetic task, or producing a useful confidence
selector is not sufficient.

## 2. Controlling prior evidence

The current from-scratch line is closed:

- E2 is controlling `VOID / INSUFFICIENT_CONFIDENCE_MATCHED_PAIRS_PER_SEED`,
  with 0 qualifying pairs against the required 200 in each seed.
- E2-DIAG Stage 0 remained at 32/128 training examples correct through 3,000
  updates and registered `STOP → KILL_FROM_SCRATCH_LINE`.
- Instrumentation found healthy downstream gradients, a changed readout, and
  3,000/3,000 clipped updates. This supports an optimization-saturated
  training-regime interpretation for the tested configuration, not a wiring
  failure and not impossibility under every optimizer or budget.
- The full 0.5B semantic-ratchet launch remains blocked.
- Line 07 remains independent and cannot be altered, delayed, rescued, or
  interpreted by E3.

The canonical failure synthesis is recorded in
`docs/UNIFIED_ERROR_SPACE.md`. E3 inherits its central constraint:
**competence and informative denominators must precede mechanics claims.**

## 3. Frozen substrate and cohort isolation

Use the already revision-fixed line-06 checkpoint alias `base-B`.

The exact repository identity, revision, tokenizer revision, resolved local
path, license, file map, and local content digest remain bound through the
gitignored line-06 local manifest. Tracked files use only the alias `base-B`.

The substrate is entirely frozen:

- no full or partial fine-tuning;
- no LoRA, adapters inside the substrate, prompt tuning, or embedding updates;
- no checkpoint comparison;
- no layer selection using observed competence or mechanics outcomes;
- evaluation mode throughout, with dropout disabled;
- gradients prohibited on every substrate parameter.

E3 consumes:

- zero line-07 calibration examples;
- zero line-07 test examples;
- zero additional GSM8K official-test examples;
- zero additional SVAMP official-test examples;
- zero E2 test examples.

All E3 training, calibration, smoke, and test examples come from fresh,
procedurally generated synthetic-deduction namespaces.

The line-07 `base-C` checkpoint is not an E3 fallback. No alternative
pretrained checkpoint may be substituted after registration.

## 4. Data generator and split contract

Reuse the verified typed-Datalog deduction construction class from E2:

- four balanced answer choices;
- proof depths 2 through 5;
- unique shortest proof and exactly one target choice;
- atomic counterfactual pairs;
- direction and late lures;
- balanced answer positions;
- equal option-context occurrences;
- disjoint symbolic skeletons and name combinations across splits;
- no identifier, rendered-question, skeleton, or counterfactual group overlap.

Freeze a new E3 generator seed and bind the generator source, template
inventory, rendered serialization, tokenizer, and every split by SHA-256 before
the instrument probe.

### Canonical retained splits

| Split | Examples | Use |
|---|---:|---|
| `controller_train` | 8,192 | Controller and matched-control training |
| `selector_harvest` | 2,048 | Critic replay and on-policy state construction |
| `selector_calibration` | 1,024 | Selector fitting, `t*`, and arm-4 delta |
| `test` | 4,096 | One-time final mechanics adjudication |

### Preflight-only splits

| Split | Examples | Use |
|---|---:|---|
| `instrument_probe_train` | sufficient for 50 fixed batches | Gradient-scale probe only |
| `competence_smoke_train` | 2,048 | Throwaway smoke training |
| `competence_smoke_validation` | 512 | One-time smoke adjudication |

All six namespaces are pairwise disjoint. Probe and smoke examples are excluded
from canonical E3 training, calibration, harvesting, and test.

The 512 smoke-validation examples must contain exactly 128 examples in each
answer class and preserve the registered depth, lure, template, and
counterfactual balance construction.

## 5. Frozen representation interface

Serialize each deduction using one fixed, hash-bound prompt template and the
native frozen `base-B` tokenizer. The answer label is never serialized into the
input.

Extract the frozen final-layer per-token hidden states after the substrate’s
final normalization. These states are the only substrate representation
available to the trainable system.

The trainable common model consists of:

- one projection from the frozen substrate width to width 512;
- eight learned latent plan slots;
- two pre-LN transformer controller layers;
- controller weights tied across recurrent iterations;
- no absolute iteration embedding;
- cross-attention from plan slots to frozen token states;
- one answer decoder/readout producing exactly four logits.

The projection, plan slots, controller, decoder, and readout together must have
at most 30 million trainable parameters.

The latent critic must have between 0.7 million and 0.9 million trainable
parameters. Frozen-base parameters are reported separately and never counted as
trainable overhead.

Train controller horizons in the exact repeating sequence `T={1,2,4}`.
Evaluate every integer horizon `T=1,...,32`. Horizons 17 through 32 are
inference-only.

## 6. Stage -1: pre-data instrument-calibration probe

### Purpose

Select the controller/readout global gradient-clip norm from the actual E3
trainable parameter scale without using any retained state or any validation,
calibration, or test outcome.

This probe is outcome-blind but not label-blind. Synthetic training labels are
used solely to compute the registered four-way training loss and its raw
gradient. No accuracy, prediction, class-collapse, validation loss, or
competence result may be computed or inspected.

### Execution

Use two fixed throwaway initialization seeds, distinct from the smoke and
canonical model seeds.

For each throwaway initialization:

1. instantiate the exact E3 frozen substrate, projection, controller, decoder,
   and readout;
2. run exactly 25 optimizer updates on fixed, disjoint
   `instrument_probe_train` batches;
3. measure the global pre-clip L2 gradient norm over all trainable projection,
   controller, decoder, and readout parameters before every update;
4. use provisional safety clip `16.0` only to make the throwaway optimizer
   steps operationally bounded;
5. use the registered optimizer, peak LR, weight decay, precision, batch size,
   loss, and the first 25 steps of the registered canonical warmup schedule.

The two initializations produce exactly 50 finite raw gradient-norm
observations. Both models, optimizers, dataloader states, caches, checkpoints,
and RNG states are then discarded and may never be resumed.

The latent critic is not part of this probe. Its clip remains separately fixed
at `1.0`, with raw critic gradient norms reported during canonical fitting.

### Frozen mapping

Let `m` be the ordinary median of the 50 global pre-clip norms, computed as the
mean of the 25th and 26th sorted observations.

Define the E3 controller/readout clip norm:

```text
C = min(16, max(2, floor(m + 0.5)))
```

Thus rounding is nearest integer with exact half values rounded upward. The
mapping has no tunable choice.

The lower bound of `2` prevents a sub-signal clip inherited from a very small
or initialization-specific median. The upper bound of `16` preserves a hard
safety limit. The probe estimates a robust central scale only; it is not
evidence about learning, optimization quality, or tail behavior.

Only the following may cross from the probe into later stages:

- the scalar `C`;
- the probe-integrity token;
- the immutable audit record containing the 50 raw norms, their quantiles, and
  the deterministic mapping result.

No weight, checkpoint, optimizer state, selected example, loss trajectory, or
RNG state may cross.

### Probe outcomes

**PASS:** exactly 50 finite, strictly positive norms exist; all hashes and
parameter boundaries match; `C` is derived exactly by the frozen mapping.

**PREFLIGHT_STOP:** any norm is nonfinite or nonpositive, the count is not 50,
the trainable parameter boundary is wrong, a forbidden metric is computed, or
the mapping is not reproduced exactly.

A probe stop has reason `INSTRUMENT_CALIBRATION_INVALID`. It authorizes no
alternative clip, repeated probe, or canonical training. Repair requires a
fresh pre-data amendment and review.

## 7. Stage 0: competence-spend smoke gate

### Purpose

Test the largest single E3 risk before the full spend: whether the frozen
`base-B` representation and registered controller interface expose enough
signal to learn the hard four-way deduction task.

This is a spend gate, not the E3 mechanics adjudication.

### Architecture and state isolation

Use the exact canonical E3 projection, controller, decoder, and readout
architecture—not a reduced-capacity proxy—with one fixed throwaway smoke seed.

The instrument-derived clip `C` is frozen before the smoke begins.

The smoke initialization, optimizer, checkpoints, cached representations,
dataloader state, and RNG state are discarded after the gate and may never
initialize or resume canonical E3.

### Training

- 2,048 `competence_smoke_train` examples;
- batch size 8;
- exactly 500 completed optimizer updates;
- exact repeating `T={1,2,4}` horizon schedule;
- four-way cross-entropy only;
- AdamW with peak LR `3e-4`;
- 50-update linear warmup followed by a constant peak-LR plateau;
- registered weight decay and betas;
- controller/readout clip `C`;
- frozen substrate throughout.

The smoke wall-time cap is 15 minutes from first substrate loading through the
single validation evaluation.

### Adjudicating metric

After update 500, evaluate the 512 smoke-validation examples once at `T=4`.

No intermediate validation accuracy, alternative horizon, best checkpoint,
curriculum, threshold selection, or repeated seed may be inspected.

**PASS:** at least 205/512 correct, i.e. at least 40.0390625%.

**PREFLIGHT_STOP:** at most 204/512 correct.

**VOID_NO_ROUTE:** the endpoint is not reached because of a hash, accounting,
hardware, nonfinite-loss, cache, or 15-minute-cap failure.

Integer counts control. Report the exact numerator, denominator, percentage,
and 95% Wilson interval. Passing does not establish canonical competence or
mechanics.

### Frozen smoke branch table

| Outcome | Route |
|---|---|
| `PASS` | Authorize canonical E3 only after all remaining launch gates pass. |
| `PREFLIGHT_STOP / PRETRAINED_INTERFACE_COMPETENCE_SMOKE_MISS` | Do not spend the full E3 budget. Route only to drafting `REGISTER_INTERFACE_SUPERVISION_DIAGNOSTIC`. |
| `VOID_NO_ROUTE` | No scientific interpretation and no successor route. |

`REGISTER_INTERFACE_SUPERVISION_DIAGNOSTIC` does not authorize a run. Its fresh
registration must separate at least:

1. representation-interface choice; and
2. answer-only versus verified dense process supervision.

The smoke miss cannot by itself establish that all frozen pretrained
representations fail, that dense supervision is necessary, or that `base-B`
lacks deduction knowledge.

No alternative interface, denser labels, checkpoint, extra updates, or second
smoke seed may rescue E3 after a miss.

## 8. Canonical training contract

Canonical E3 may launch only after:

- instrument-probe PASS;
- competence-smoke PASS;
- exact runner/config/generator/split/model hash binding;
- lightweight self-tests;
- measured compute-cap preflight;
- independent pre-training review of the full pipeline;
- every blocking review finding resolved;
- an explicit clean launch attestation.

### Controller training

For each canonical model seed `42` and `31415`:

- initialize independently from scratch above the same frozen substrate;
- train one shared variable-depth controller checkpoint;
- batch size 8;
- exactly 2,500 optimizer updates;
- exactly 20,000 example exposures;
- `T={1,2,4}` in an exact repeating balanced schedule;
- four-way cross-entropy only;
- AdamW, peak LR `3e-4`;
- betas `(0.9, 0.95)`;
- epsilon `1e-8`;
- weight decay `0.01`, excluding biases, normalization parameters, and latent
  plan-slot parameters;
- BF16 with TF32 enabled;
- controller/readout clip `C`;
- 500-update linear warmup;
- constant peak-LR plateau through update 2,000;
- linear decay from update 2,001 through 2,500 to `3e-5`;
- no curriculum;
- no checkpoint selection: update 2,500 is controlling.

Record raw gradient norms and clipped-update numerators/denominators over every
completed update, separately for projection, controller, decoder/readout, and
the global trainable set.

Harvest checkpoints are initialization and updates 500, 1,250, 2,000, and
2,500. They exist only for registered critic replay and cannot replace the
final controller checkpoint.

### Non-recurrent control

For each model seed, train a parameter-matched non-recurrent control over the
same frozen token representations.

It must:

- use the same projection and four-way answer interface;
- have trainable parameters within 5% of the recurrent common model;
- receive the same examples, example exposures, supervision, optimizer family,
  schedule, precision, and model seed;
- use one fixed-depth feed-forward/attention computation rather than tied
  recurrent iteration;
- use the same instrument-derived controller clip `C`;
- use its final update-2,500 checkpoint without selection.

If either non-recurrent-control seed exceeds 80.0% test accuracy, E3 is `VOID /
NONRECURRENT_CONTROL_TOO_COMPETENT`; the task does not isolate recurrent
mechanics at pilot scale.

## 9. Selector and critic provenance

Use four inference arms over the same canonical controller checkpoint:

1. final available horizon / no latch;
2. confidence-plus-schedule latch;
3. latent progress critic plus latch;
4. hysteretic latent-critic latch, informational only.

### State harvesting

For initialization and controller updates 500, 1,250, 2,000, and 2,500,
harvest states at horizons `T={1,2,4,8,16}` using:

- one greedy rollout;
- two fixed-seed stochastic rollouts.

During critic fitting, exactly 50% of the fitting corpus must come from
checkpoint replay and 50% from the final on-policy controller.

At least 50% of critic minibatches must be confidence-matched,
opposite-correctness pairs.

### Latent critic boundary

The critic may receive:

- prompt-conditioned pooled latent state;
- prompt-conditioned pooled update;
- frozen prompt representation;
- update norm;
- consecutive-update cosine;
- cross-horizon latent agreement;
- normalized schedule coordinate `t/16`.

It may not receive:

- raw answer logits;
- answer probabilities;
- top-two margin;
- entropy;
- answer identity;
- sampled-answer frequency;
- response log-probability.

The confidence baseline receives only:

- maximum answer probability;
- top-two probability margin;
- entropy;
- `t/16`.

Both selectors are fitted and frozen using only harvest and selector-calibration
data before test evaluation. At horizons 17 through 32, `t/16` may exceed 1;
no selector may be refitted.

### Provenance floors

Before an admissible mechanics `PROCEED` or `FAIL`, require per seed:

- at least 20,000 correctness-labeled selector-training states;
- at least 10,000 correctness-labeled selector-calibration states;
- at least 200 confidence-plus-schedule-matched, opposite-correctness
  evaluation pairs.

Critic gates are:

- overall correctness AUROC at least 0.75;
- matched-pair concordance at least 0.70;
- AUROC advantage over confidence plus schedule at least 0.05;
- critic selection accuracy at least 3 points above confidence plus schedule;
- on-policy-only correctness AUROC at least 0.70.

All selectors are frozen before test metrics are computed. Final adjudication
applies `VOID > FAIL > PROCEED` precedence, so a competence or denominator VOID
cannot be hidden by a critic-gate miss.

## 10. Calibration-frozen horizon and competence floors

Using selector-calibration data only, choose:

\[
t^* =
\arg\max_{t\in\{1,\ldots,15\}}
\frac{1}{2}\sum_s A_{\mathrm{no},s}^{\mathrm{cal}}(t).
\]

Break ties toward the smaller horizon. Freeze one shared `t*` before test
inspection.

For each seed `s`, define:

\[
S_s=\{i:\hat y^{\mathrm{no}}_{i,s,t^*}=y_i\}.
\]

Before any admissible `PROCEED` or `FAIL`, require in each seed:

- no-latch test accuracy at `t*` of at least 60%, meaning at least 2,458/4,096
  correct;
- no-latch `t*`-minus-`T=1` gain of at least 5 absolute points, meaning at
  least 205 additional correct examples;
- `|S_s| >= 500`;
- every selector population and matched-pair floor in Section 9.

A miss on any competence or population floor is `VOID`. Chance flicker,
underpowered matching, or a competent pretrained base without useful recurrent
gain cannot adjudicate latch mechanics.

## 11. Endpoint and regression definitions

For endpoint `H` in `{16,32}`:

\[
R_{\mathrm{no},s}(H)=
\frac{\#\{i\in S_s:\hat y^{\mathrm{no}}_{i,s,H}\ne y_i\}}{|S_s|}
\]

and

\[
R_{\mathrm{critic},s}(H)=
\frac{\#\{i\in S_s:\hat y^{\mathrm{critic}}_{i,s,H}\ne y_i\}}{|S_s|}.
\]

Pooled metrics concatenate example records across seeds. They never average
seed-level percentages without their numerators and denominators.

When `R_no(H) >= 10%`, regression reduction is:

\[
1-\frac{R_{\mathrm{critic}}(H)}{R_{\mathrm{no}}(H)}.
\]

Gain retention is:

\[
\frac{A_{\mathrm{critic}}(H)-A_{\mathrm{no}}(1)}
     {A_{\mathrm{no}}(t^*)-A_{\mathrm{no}}(1)}.
\]

Endpoint selection is frozen:

1. If no-latch regression is at least 10% pooled and separately in both seeds
   at `H=16`, adjudicate at 16.
2. If it misses at 16, inspect the already-generated `H=32` endpoint.
3. If it is at least 10% pooled and separately in both seeds at 32, adjudicate
   at 32.
4. Otherwise regression is unmeasurable. No intermediate endpoint, subset, or
   alternative denominator may be selected.

## 12. Arm 4: non-adjudicating hysteretic critic

For each seed, use the post-sigmoid critic score and select one delta from:

```text
{0.00, 0.02, 0.05, 0.10}
```

using only selector-calibration trajectories through `T=16`.

A challenger replaces the incumbent only when:

```text
challenger_score - incumbent_score > delta
```

Require calibration gain retention of at least 90%. Among feasible deltas,
select lexicographically:

1. smallest harmful-switch-example rate;
2. highest `T=16` accuracy;
3. larger delta.

If none is feasible, freeze `delta=0` and record
`calibration_constraint_miss=true`.

Arm 4 is informational only. It cannot:

- rescue a FAIL;
- veto a PROCEED;
- satisfy a competence or denominator floor;
- enter a regression-only exception;
- select the endpoint;
- alter any 0.5B decision.

Its values are excluded from every E3 `PROCEED`, `FAIL`, and `VOID`
computation.

## 13. Final outcome rules

Decision precedence is:

1. integrity or operational `VOID`;
2. non-recurrent-control VOID;
3. competence or denominator VOID;
4. selector-provenance FAIL;
5. mechanics outcome.

### PROCEED

E3 receives `PROCEED` only if all conditions hold pooled and separately in both
seeds at the frozen endpoint:

- no-latch correct-to-wrong regression at least 10%;
- critic-latch regression reduction at least 80%;
- critic-latch gain retention at least 90%;
- critic-latch accuracy at least 3 absolute points above confidence plus
  schedule;
- confidence-plus-schedule-matched critic concordance at least 0.70;
- every competence, population, provenance, integrity, and control gate passes.

`PROCEED` supports only the statement that this frozen pretrained substrate and
pilot interface produced an admissible positive mechanics gate. It does not
automatically satisfy the old E2 prerequisite or authorize the 0.5B launch. A
fresh steering decision must explicitly define any replacement relationship.

### FAIL

If every integrity, control, competence, and denominator floor passes but any
independently measurable selector or mechanics condition misses, assign
`FAIL / PRETRAINED_LATCH_MECHANICS_MISSED`.

A provenance-gate miss with otherwise valid competence and denominators is an
admissible FAIL. The test split may not be used to repair or refit the critic.

FAIL kills this registered pretrained-substrate interface and latch design. It
does not kill all pretrained substrates, recurrent controllers, critics, or
outcome banking.

### Regression-only VOID

If regression is below 10% at both 16 and 32, assign
`VOID / UNMEASURABLE_REGRESSION` only if:

- all integrity, control, competence, denominator, and critic-provenance gates
  pass;
- gain retention is at least 90% at 32;
- critic accuracy exceeds confidence plus schedule by at least 3 points at 32;
- matched critic concordance is at least 0.70 at 32.

This outcome does not validate the latch and does not automatically unblock
the 0.5B program.

### Other VOID outcomes

Assign VOID for:

- representation, generator, split, tokenizer, code, config, or checkpoint
  hash mismatch;
- frozen-base parameter change;
- probe or smoke boundary violation;
- train/calibration/test overlap;
- missing canonical seed;
- non-recurrent control above 80%;
- competence or sample-size floor miss;
- compute mismatch;
- incomplete required stage;
- nonfinite training;
- unresolved review finding;
- overwritten or regenerated evidence;
- any official GSM8K, SVAMP, E2-test, or line-07 cohort access.

A VOID authorizes no retry, threshold change, extra seed, alternative
checkpoint, interface repair, or post-result dense-supervision variant.

## 14. Compute and resource contract

The instrument probe uses at most 50 total throwaway updates.

The competence smoke uses at most 15 wall-clock minutes.

The retained canonical E3 invocation has a hard cap of 9,000 seconds from first
CUDA allocation, including:

- frozen-representation construction or loading;
- both recurrent-controller seeds;
- both non-recurrent controls;
- critic and confidence fitting;
- trajectory evaluation through `T=32`;
- result serialization.

Before launch, a retained throughput/memory preflight must project completion
within 9,000 seconds on the registered hardware. If the projection exceeds the
cap, canonical E3 does not launch and requires a pre-data resize amendment.

If the next required stage cannot complete inside the remaining canonical
budget, it must not start. An incomplete canonical suite is
`VOID_NO_ROUTE / COMPUTE_CAP_EXHAUSTED`, not a scientific miss.

Frozen-representation caches must:

- live in a gitignored workspace-local directory;
- be keyed by checkpoint, tokenizer, prompt, generator, and split hashes;
- have a preregistered maximum disk footprint of 32 GiB;
- refuse stale or partial cache reuse;
- be deleted after evidence landing or retained only as ignored reproducibility
  cache;
- never be committed.

Report wall time, CUDA-active time where measurable, peak allocated/reserved
VRAM, cache bytes, processed examples, generated base tokens, and all update
denominators.

## 15. Evidence artifacts

The preflight writes exactly one immutable artifact:

`experiments/06_uesd/results/exp_e3_preflight.json`

It contains:

- all probe hashes and 50 raw gradient norms;
- median `m` and derived clip `C`;
- confirmation that no forbidden probe metric was computed;
- smoke split hashes;
- smoke final numerator, denominator, percentage, and Wilson interval;
- wall time and hardware telemetry;
- terminal `PASS`, `PREFLIGHT_STOP`, or `VOID_NO_ROUTE`;
- confirmation that every throwaway state was discarded.

Only preflight PASS permits the canonical run.

The canonical run writes exactly one immutable artifact:

`experiments/06_uesd/results/exp_e3_pretrained_latch_mechanics.json`

It contains:

- code, config, generator, split, substrate, tokenizer, cache, and checkpoint
  hashes;
- exact trainable and frozen parameter counts;
- every raw numerator and denominator;
- controller, control, critic, and selector training accounting;
- gradient and clipping telemetry;
- accuracy grids through `T=32`;
- trajectory transitions and selector-switch hazards;
- arm-4 informational results;
- per-example test records;
- compute and hardware telemetry;
- final `PROCEED`, `FAIL`, or `VOID`;
- confirmation of zero forbidden cohort access.

Both artifacts use same-directory fsync, atomic no-clobber publication.
Checkpoints, caches, and raw logs remain untracked provenance.

## 16. Landing and review contract

### Registration landing

The registration becomes active only when one coherent pre-data block:

1. adds this document;
2. appends `E3_PRETRAINED_MECHANICS_PREREGISTERED` to
   `experiments/ledger.jsonl`;
3. appends a preregistered/not-run E3 entry to `experiments/EXPERIMENTS.md`;
4. updates `STATUS.md` to record:
   - the from-scratch kill;
   - the bounded optimization-saturation interpretation;
   - E3 registration/build authorization;
   - the continuing 0.5B block;
   - line-07 independence and priority;
5. adds the full failure synthesis to `docs/UNIFIED_ERROR_SPACE.md`;
6. records that no E3 probe, smoke, training, calibration, or test outcome
   existed when this protocol was frozen.

### Preflight landing

A completed preflight updates, in one block:

- the immutable preflight artifact;
- ledger;
- notebook;
- `STATUS.md`;
- the UESD claim audit.

A PASS authorizes only reviewed canonical launch. A stop or VOID lands its
terminal route and does not leave E3 marked active.

### Canonical-result landing

A completed or terminal canonical E3 updates, in one block:

- immutable canonical result artifact;
- ledger;
- notebook;
- `docs/UNIFIED_ERROR_SPACE.md`;
- `STATUS.md`;
- `experiments/06_uesd/audit_uesd_claims.py`.

Before any launch, the full pipeline must receive the mandatory independent
pre-training review for implementation bugs, crashes, unbounded VRAM/RAM/disk
use, process leaks, checkpointing, accounting, split leakage, selector
provenance, and system-shutdown risk.

After the coherent result block, perform the holistic PR/evidence/overclaim
review and loop until clean.

## 17. Frozen prohibition on adaptive rescue

After any E3 probe, smoke, calibration, training, or test observation, no agent
may:

- change the clip mapping or bounds;
- choose another clip from the observed quantiles;
- repeat the probe;
- repeat the smoke;
- change the smoke threshold;
- substitute a substrate or representation layer;
- change the interface;
- add dense supervision;
- add controller updates;
- change the schedule or LR;
- add a seed;
- select a non-final checkpoint;
- change `t*`, an endpoint, or a denominator;
- fit a selector on test data;
- promote arm 4;
- inspect a line-07 cohort;
- reinterpret a VOID as a FAIL or a FAIL as promising evidence.

Any such change requires a fresh registration and cannot be represented as E3.

---

**Frozen before any E3 instrument probe, smoke, training, calibration, or test
outcome exists.**
