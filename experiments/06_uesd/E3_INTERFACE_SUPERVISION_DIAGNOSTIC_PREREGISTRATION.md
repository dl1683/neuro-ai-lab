# E3 Interface-Supervision Diagnostic Preregistration

Status: **FROZEN_NOT_LAUNCHABLE / NOT RUN / NO SCIENTIFIC CONCLUSION**

Experiment ID: `exp_e3_interface_supervision_diagnostic`

Route token: `REGISTER_INTERFACE_SUPERVISION_DIAGNOSTIC`

## 1. Purpose and controlling boundary

E3 stopped at its registered pre-spend competence gate:

`PREFLIGHT_STOP / PRETRAINED_INTERFACE_COMPETENCE_SMOKE_MISS`

The outcome-blind gradient probe passed with 50/50 finite positive norms and
froze controller clip `C=14`. The exact registered 13,671,428-parameter
recurrent controller then completed 500 throwaway updates. Its mean four-way
training CE moved from 1.6670 in the first 50-update window to 1.4062 in the
last, but its single registered validation measurement was exactly 128/512
correct, or 25.0%, against the 205/512 floor.

This miss does not distinguish:

1. whether the frozen representation exposes the answer-relevant deduction at
   the registered interface; from
2. whether sparse answer-only CE fails to provide a learning foothold that
   verified step-level supervision would provide.

This diagnostic separates those factors. It is not an E3 retry, interface
rescue, mechanics experiment, or 0.5B launch gate.

## 2. Accumulated evidence to be explained

The diagnostic is motivated by three bounded observations:

- The current from-scratch 30M line could not memorize 128 balanced full-hard
  examples. Accuracy remained 32/128 through 3,000 updates, CE remained at
  approximately `ln(4)`, every post-update checkpoint predicted one class for
  all examples, and 3,000/3,000 updates were clipped. Healthy downstream
  gradients rule against a severed computation path.
- The frozen-pretrained E3 controller reduced training CE from 1.6670 to 1.4062
  over 500 updates with calibrated clip `C=14`, but validation remained exactly
  128/512. The observed loss reduction is compatible with learning calibration
  or marginal statistics without learning example discrimination.
- Across the tested substrate/controller configurations, the shortcut-resistant
  balanced 2-5-hop task has produced no demonstrated answer discrimination
  under sparse answer-only CE.

The task has four balanced answer positions, unique shortest proofs,
counterfactual pairing, direction and late lures, disjoint symbolic skeletons
across splits, and sparse answer-only supervision. Its generator can therefore
supply mechanically verified proof traces without relying on generated
chain-of-thought text.

## 3. Diagnostic hypotheses

### H-Interface

Frozen `base-B` final-layer representations expose at least one of:

- the answer to the complete hard prompt;
- atomic entailment at early proof steps;
- atomic entailment throughout the unique 2-5-step proof.

### H-Supervision

Holding the frozen substrate, controller core, initialization, hard examples,
example order, optimizer, answer loss, and compute budget fixed, verified
rule-application supervision creates answer discrimination that answer-only CE
does not.

### Falsifiers

- `H-Interface` misses its registered floor if the direct probe bank fails the
  hard-answer floor and the fact-trace probe fails at the relevant proof
  distances.
- `H-Supervision` misses its registered floor unless the dense arm learns
  verified process targets, clears hard-answer competence, and beats its paired
  answer-only arm by the registered causal margin.

Neither miss establishes representational impossibility or universal necessity.

## 4. Frozen substrate and access boundary

Use the already revision-bound frozen checkpoint alias `base-B` and its existing
tokenizer manifest.

The substrate remains entirely frozen:

- no full or partial fine-tuning;
- no LoRA, adapter, prompt-tuning, or embedding updates;
- no checkpoint or layer search;
- evaluation mode with dropout disabled;
- no substrate gradients;
- final-layer post-normalization token states only.

No cell may access:

- line-07 calibration or test examples;
- GSM8K or SVAMP official-test examples;
- the E2 official test;
- E2/E3 selector, critic, or latch outcomes;
- an E3 smoke example or retained E3 state.

All diagnostic examples use fresh synthetic namespaces.

## 5. Generator and verified-process contract

Reuse the typed-Datalog construction class and the full hard-task controls:

- balanced four-way answers;
- proof depths 2-5;
- unique shortest proof;
- exactly one target choice;
- counterfactual pairs;
- clean, direction-lure, and late-lure strata;
- balanced answer positions and option-context occurrences;
- name-, skeleton-, prompt-, and counterfactual-group-disjoint splits.

Extend the verifier only enough to reconstruct the unique shortest proof.
Before hashing any split, independently replay every trace and require:

1. every premise is available before its use;
2. every rule application is type-valid;
3. every conclusion follows from its registered premises and rule;
4. the terminal conclusion entails exactly the gold answer;
5. the trace length equals the verifier's minimum proof cost;
6. the shortest-proof count remains exactly one.

For each proof step, construct four type- and arity-matched candidate rule
applications. Exactly one must be valid at that step. Candidate positions,
rule schemas, entity mentions, and predicates must be balanced within each
split. The three distractors must fail symbolic replay at that step.

These candidates are supervision objects only. They are never inserted into
the controller's answer prompt or cross-attended by its recurrent core.

## 6. Frozen splits

| Split | Size | Use |
|---|---:|---|
| `easy_train` | 2,048 | One-hop probe and controller floor |
| `easy_validation` | 1,024 | One-hop endpoint adjudication; 256 per answer position |
| `hard_train` | 2,048 | Hard-answer probe and both paired controller arms |
| `hard_validation` | 512 | Hard endpoint comparison; 128 per answer position and 128 per proof depth |
| `fact_probe_train` | 2,048 queries per distance | Atomic-entailment probe training |
| `fact_probe_validation` | 512 queries per distance | Atomic-entailment adjudication at distances 0-5 |

The answer-only and dense-supervision hard cells use exactly the same
`hard_train` and `hard_validation` rows, initialization, minibatch permutation,
and example exposures.

All six namespaces are hash-bound and disjoint from E2, E3 preflight,
official-test, and line-07 cohorts.

## 7. Frozen six-cell grid

Every cell has a hard 900-second GPU wall-time cap. The complete invocation has
a hard 5,400-second cap, including substrate loading, representation
construction, training, evaluation, and result publication.

| Cell | Axis | Trainable component | Task | Supervision | Endpoint |
|---|---|---|---|---|---|
| `I0_ONE_HOP_LINEAR_FLOOR` | Interface | Affine probes only | E2-DIAG one-hop unary task | Answer labels | 1,024 examples |
| `I1_HARD_ANSWER_LINEAR` | Interface | Affine probes only | Original full-hard task | Answer labels | 512 examples |
| `I2_FACT_TRACE_LINEAR` | Interface | Binary affine probe | Atomic entailment, distances 0-5 | Entailed/not entailed | 512 per distance |
| `S0_ONE_HOP_CONTROLLER_FLOOR` | Supervision/control | E3 controller and answer readout | One-hop unary task | Answer CE only | 1,024 examples |
| `S1_HARD_ANSWER_ONLY` | Supervision | E3 controller and answer readout | Original full-hard task | Answer CE only | 512 examples |
| `S2_HARD_PROCESS_DENSE` | Supervision | Paired E3 controller, answer readout, process head | Original full-hard task | Answer CE plus verified process CE | Same 512 examples |

All six scientific cells run unless an integrity or operational failure makes
the suite `VOID_NO_ROUTE`. Scientific outcomes are not inspected between cells.

Shared representation construction is charged to the first cell that
constructs each cache. No unmetered setup or post-grid GPU work is permitted.

## 8. Interface-axis protocols

### 8.1 Fixed probe views

Cells I0 and I1 train three independent affine four-way probes over these
preregistered final-layer views:

1. final non-padding token;
2. masked mean of all prompt tokens;
3. masked mean of the rendered `Query` and `Choices` span.

The views, layers, training schedule, and thresholds are frozen. No best view
is selected for downstream use, and no probe weight may initialize a successor.

Each probe uses:

- fixed initialization seed;
- exactly 1,000 optimizer updates;
- batch size 64;
- AdamW with fixed LR `1e-3`;
- weight decay `0`;
- clip norm `1`;
- final-update checkpoint only;
- no intermediate validation inspection.

### 8.2 I0 one-hop interface floor

Use the unrun E2-DIAG Stage-1 construction:

- exactly one supporting unary fact;
- exactly one unary implication;
- no binary relation, conjunction, lure, or counterfactual reversal;
- fixed templates and shared entity vocabulary;
- exact train/validation prompt disjointness;
- exact label balance.

`I0_PASS` requires at least one frozen probe view to reach 922/1,024 correct.

A miss is scoped to this frozen representation and serialization floor.

### 8.3 I1 hard-answer exposure

Train the same three probe views directly on full hard-prompt representations.

`I1_PASS` requires at least one view to reach 205/512 correct.

This threshold matches the E3 competence-spend floor. A pass establishes only
registered linear decodability at one view. A miss does not establish that
deduction knowledge is absent.

### 8.4 I2 depth-localized atomic entailment

For each prompt, append a fixed query asking whether one candidate atomic fact
is entailed. Positives and matched negatives must be balanced for entities,
predicates, lexical occurrence, proof depth, and answer relationship.

Train one shared binary affine probe and report separately for:

- distance 0: explicitly stated input facts;
- distance 1;
- distance 2;
- distance 3;
- distance 4;
- distance 5.

Each distance has exactly 256 entailed and 256 non-entailed validation queries.

A distance passes at 308/512 correct or better.

Classify the fact interface as:

- `FACT_FULL`: all distances 0-5 pass;
- `FACT_PARTIAL`: distances 0 and 1 pass, but at least one distance 2-5 misses;
- `FACT_NONE`: distance 0 or 1 misses.

Report exact confusion matrices and Wilson intervals at every distance. Do not
collapse the depth curve into an unweighted average.

## 9. Supervision-axis protocols

### 9.1 Common controller

Cells S0-S2 use the frozen E3 common architecture:

- frozen `base-B` final-layer token states;
- projection to width 512;
- eight learned plan slots;
- two tied pre-LN recurrent controller layers;
- no absolute iteration embedding;
- cross-attention to frozen token states;
- four-way answer readout;
- controller/readout clip fixed at the prior measured `C=14`.

S1 and S2 begin from byte-identical core and answer-head initialization. They
receive identical hard examples in identical order and complete exactly 500
optimizer updates with batch size 8. Both evaluate once on the same 512
examples.

The hard comparison runs five recurrent iterations so the process arm has one
registered state per possible proof step. This diagnostic alignment is not an
E3 horizon-mechanics result.

### 9.2 S0 one-hop controller floor

Train for exactly 500 updates at `T=1`, answer CE only.

`S0_PASS` requires at least 922/1,024 validation examples correct.

This separates "the frozen representation exposes the easy task" from "the
registered controller can consume that representation within the smoke budget."

### 9.3 S1 hard answer-only arm

Train at fixed `T=5` with:

\[
L_{\text{S1}}=L_{\text{answer}}.
\]

`S1_COMPETENT` requires at least 205/512 correct at the final endpoint.

This is a causal control for S2. It cannot repeat, rescue, or replace the
stopped E3 smoke.

### 9.4 S2 verified dense-process arm

Instantiate the same controller core, answer readout, initialization, examples,
minibatch order, optimizer, clip, and update count as S1.

At recurrent step \(k\), a shared auxiliary scorer receives controller state
\(z_k\) and frozen representations of the four registered candidate rule
applications. Candidate representations do not enter the controller core or
answer forward path.

Use:

\[
L_{\text{S2}}
=
L_{\text{answer}}
+
\frac{1}{d}\sum_{k=1}^{d}L_{\text{rule},k},
\]

where \(d\) is the unique proof length and each `rule` loss is four-way CE
over the verified candidate applications. The answer-loss coefficient remains
exactly one.

The process head is also instantiated in S1 but has coefficient zero, receives
no optimizer update, and cannot affect its answer forward pass. This preserves
byte-identical answer paths while isolating the additional supervision in S2.

Define:

- `PROCESS_FULL`: each active proof-step stratum reaches at least 60% correct
  and at least 308/512 complete proof traces are predicted exactly;
- `PROCESS_PARTIAL`: step 1 passes, but a later step or the complete-trace floor
  misses;
- `PROCESS_NONE`: step 1 misses.

For the mixed-depth hard validation set, the per-step 60% integer floors are:

| Step | Denominator | Correct floor |
|---|---:|---:|
| 1 | 512 | 308 |
| 2 | 512 | 308 |
| 3 | 384 | 231 |
| 4 | 256 | 154 |
| 5 | 128 | 77 |

`S2_COMPETENT` requires at least 205/512 final answers correct.

### 9.5 Dense-supervision causal gain

On the paired hard validation examples, let:

- \(n_{01}\): S1 wrong and S2 correct;
- \(n_{10}\): S1 correct and S2 wrong.

`DENSE_GAIN` requires all of:

1. `S2_COMPETENT`;
2. `PROCESS_FULL`;
3. S2 minus S1 correct count at least 77/512, or at least 15 absolute points;
4. one-sided exact McNemar/binomial \(p \le 0.01\) over \(n_{01}\) versus
   \(n_{10}\).

Training-loss improvement alone cannot satisfy `DENSE_GAIN`.

## 10. Frozen branch table

Integrity and operational VOIDs have precedence. Rows are evaluated top to
bottom.

| Observed pattern | Route | Meaning |
|---|---|---|
| Any hash, split, verifier, access, nonfinite, accounting, publication, per-cell cap, or total-cap failure | `VOID_NO_ROUTE` | No scientific interpretation and no successor |
| I0 and S0 pass, and S2 satisfies `S2_COMPETENT + PROCESS_FULL + DENSE_GAIN` | `REGISTER_DENSE_SUPERVISION_E3B` | Verified process supervision created a competent discrimination foothold under the paired diagnostic |
| I0 misses; or I0 passes while S0 misses; or I1/`FACT_FULL` exposes hard information while both hard controller arms miss competence | `REGISTER_INTERFACE_REDESIGN` | Representation serialization, pooling, controller access, or answer extraction is the leading registered bottleneck |
| S1 is competent but dense supervision lacks its causal gain | `REGISTER_INTERFACE_REDESIGN` | Answer-only learning is possible under the diagnostic scaffold; dense supervision is not isolated as necessary |
| I0 and S0 pass, I1 misses, no hard arm is competent, and either `FACT_PARTIAL` or `PROCESS_PARTIAL` occurs | `REGISTER_TASK_FAMILY_CHANGE` | A local foothold exists but the full anti-shortcut 2-5-hop family exceeds the demonstrated compositional interface |
| I0 and S0 pass, I1 misses, both hard arms miss competence, `FACT_NONE`, and `PROCESS_NONE` | `KILL_SYNTHETIC_DEDUCTION_FAMILY` | Retire the current full-hard synthetic family as the mechanics substrate under the registered bounded program |
| Any other valid mixed pattern | `VOID_NO_ROUTE / MIXED_DIAGNOSTIC_PATTERN` | The grid does not uniquely separate interface from supervision; no post-hoc threshold or branch is allowed |

`KILL_SYNTHETIC_DEDUCTION_FAMILY` is scoped to the current shortcut-resistant
2-5-hop construction as a bounded semantic-ratchet mechanics substrate. It is
not a claim that synthetic deduction, pretrained reasoning, recurrent
computation, or process supervision is impossible.

A family kill must be followed by a failure synthesis and at least three
outside-family hypotheses before another direction is registered.

## 11. Non-adjudication boundaries

This diagnostic cannot:

- change or rescue the E3 preflight outcome;
- authorize canonical E3, E3b, or the 0.5B program;
- establish latch, critic, banking, regression, or horizon mechanics;
- establish that `base-B` contains or lacks deduction knowledge generally;
- establish that dense supervision is universally necessary;
- select a representation layer, pooling, checkpoint, optimizer, loss weight,
  or curriculum post-result;
- describe a one-seed result as robust or reproducible;
- treat a falling CE curve as discrimination;
- reinterpret missing quantities as zero or failed;
- modify line 07 or consume any of its cohorts.

All rates must include exact numerators, denominators, percentages,
populations, and checkpoints. Held-out binomial metrics include 95% Wilson
intervals. Zero denominators are `undefined`.

## 12. Evidence and landing contract

### Registration companions

A registration becomes active only through one reviewed pre-data block that:

1. adds `experiments/06_uesd/E3_INTERFACE_SUPERVISION_DIAGNOSTIC_PREREGISTRATION.md`;
2. implements `experiments/06_uesd/exp_e3_interface_supervision_diagnostic.py`;
3. adds one config file for hashes and frozen constants;
4. binds runner, config, generator, proof-trace verifier, templates, tokenizer,
   substrate, prompts, and all splits by SHA-256;
5. appends `E3_INTERFACE_SUPERVISION_DIAGNOSTIC_PREREGISTERED` to
   `experiments/ledger.jsonl`;
6. appends a preregistered/not-run entry to `experiments/EXPERIMENTS.md`;
7. updates `STATUS.md` without marking E3 reopened;
8. updates `docs/UNIFIED_ERROR_SPACE.md` only with the registered question and
   existing claim boundary;
9. updates `audit_uesd_claims.py`;
10. records that no diagnostic run or metric existed when the protocol was
    frozen.

No parallel roadmap, result memo, or one-off analysis script is created.

### Pre-launch gates

Because the complete invocation may exceed 30 minutes, launch requires:

- design review;
- full pipeline review for correctness, leakage, proof reconstruction, loss
  masking, gradient paths, memory/disk bounds, process cleanup, and atomic
  publication;
- measured completion projection within 5,400 seconds;
- all blocking findings resolved;
- explicit clean-launch attestation.

### Result artifact

A completed suite writes exactly one immutable artifact:

`experiments/06_uesd/results/exp_e3_interface_supervision_diagnostic.json`

It must contain:

- all hashes and access attestations;
- exact trainable/frozen parameter counts;
- per-cell updates, examples, wall time, CUDA time, and peak VRAM;
- all probe counts, confusion matrices, and intervals;
- fact-probe counts by distance;
- controller answer counts;
- process-step and exact-trace counts;
- \(n_{01}\), \(n_{10}\), paired gain, and exact \(p\)-value;
- gradient norms and clipped-update numerators/denominators;
- the frozen route token;
- confirmation that every checkpoint, optimizer, probe, process head, cache,
  and RNG state was discarded.

Publication uses same-directory fsync and atomic no-clobber creation.

### Result landing companions

A completed or terminal suite lands in one coherent block:

- immutable result JSON;
- ledger entry;
- `experiments/EXPERIMENTS.md`;
- `docs/UNIFIED_ERROR_SPACE.md`;
- `STATUS.md`;
- `audit_uesd_claims.py`;
- holistic evidence and overclaim review.

A route token authorizes only drafting and steering review of its named
successor. It never authorizes that successor's implementation or launch.

## Pre-data clarification and blocking-review amendment — 2026-08-11

This amendment resolves the ambiguity between Section 5's general
name-disjoint language and Section 8.2's I0 shared-vocabulary requirement
before any diagnostic cell or scientific metric exists.

- For I0 only, `easy_train` and `easy_validation` use the same fixed 32-entity
  vocabulary in full. They remain exactly prompt-disjoint and use explicit
  name-free symbolic skeletons that are disjoint across the two splits. The
  runner independently reconstructs and verifies the single fact, single
  unary rule, unique one-step proof, unique entailed choice, and label for
  every easy row before any split hash is computed. Section 5's name-disjoint
  requirement continues to govern the hard and fact namespaces.
- I2 matched negatives are assigned jointly at each distance by a deterministic
  capacity-constrained matching, rather than by lexical first choice. Queried
  predicate marginals are identical under both labels; every positive/negative
  pair uses the same queried entity and base answer position; both query words
  occur in the prompt. The runner asserts and serializes the label contingency
  tables for queried entity, queried predicate, entity occurrence, lexical
  occurrence, and answer-position relationship at every distance in both fact
  splits before hashing.
- S2 candidates are now genuine groundings of rules registered in the example.
  Every candidate's premises, types, and arities are reconstructed from its
  registered rule; all three distractors are inapplicable under the facts
  available at that step. The runner asserts exactly one valid candidate per
  row, exact candidate-position/validity counts, and exact 3:1 invalid-to-valid
  schema, concrete conclusion-entity, and body-predicate marginals within each
  split and step. It serializes the full split/step/position/validity tables.
- Cell completion now synchronizes the device and then enforces both the
  900-second cell cap and 5,400-second suite cap. Result serialization, file
  fsync, no-clobber link, directory sync where supported, and the final cap
  check execute inside the suite timer.
- Launch now requires an ordered six-cell projection whose cell and total
  values are finite, nonnegative, runner/config hash-bound, individually below
  900 seconds, sum exactly to the suite projection, and remain below 5,400
  seconds. The projection is GPU-free and derives its conservative values from
  the immutable E3 preflight measurement plus the fully generated audited
  workload; it computes no diagnostic outcome.
- The fast self-test iterates every miniature easy, fact, hard-process, trace,
  and candidate record; replays every registered invariant; verifies all probe
  and controller optimizer boundaries; checks the frozen thresholds; exercises
  adversarial fixtures for every branch-table route; checks post-sync cap
  enforcement; and checks atomic no-clobber publication.
- Each affine head now records exact total/trainable/frozen and optimizer-bound
  parameter counts and names. The three four-way heads each contain 6,148
  trainable parameters; the shared binary fact head contains 3,074. Frozen
  substrate parameters and cached representations remain outside every probe
  optimizer.

These are pre-data correctness clarifications and contract-strengthening
changes. They do not reopen E3, authorize I0, issue either launch attestation,
or alter any scientific threshold or route.

## Implementation binding attestation

This pre-data slot-filling block records implementation details that were
implicit in the frozen probe contract and binds the executable surface. No
diagnostic cell, validation endpoint, scientific metric, result artifact, or
line-07/official-test access existed when these slots were filled.

- I2 uses the masked mean of the appended atomic-query span and the same single
  1,000-update affine-probe schedule frozen in Section 8.1. One shared binary
  probe is fit across all six distances; it is evaluated once and then reported
  by distance without refitting or distance selection.
- The tracked config binds the registration protocol body, E2 generator source,
  E3 reusable runner source, template inventory, three prompt/serialization
  templates, proof-replay contract, tokenizer-containing frozen substrate
  digest, and all six ordered split payloads by SHA-256.
- `E3-interface-diagnostic-runner-sha256`: `c282538ce28e6373eb8f71e273c4146fbc48db9518cbfdc39d296561bb21a12e`
- `E3-interface-diagnostic-config-sha256`: `c9695f6a439bd77aa1ba28335071abe1e4828f046fe1d6593f653f0539d1a157`
- The implementation remains launch-blocked. A separate reviewer must inspect
  the complete pipeline, all blocking findings must be resolved, a reviewed
  non-scientific projection must fit 5,400 seconds, and the clean-launch token
  must be issued before I0.

## Terminal design-gate record — 2026-08-11

The permitted correction cycle and full independent re-review are exhausted.
The re-review found a design-level I2 lexical shortcut that requires
reconstructing the matched-negative assignment and rebinding the fact
datasets:

- queried predicate identities are balanced, but the queried label-word's
  context occurrence count differs under the two labels at every distance in
  both full fact splits;
- the matcher requires only nonzero occurrence and the serialized balance
  assertions omit the actual occurrence count;
- on miniature validation, a one-feature affine classifier using only that
  count reached **69.73%** accuracy. Ordinary query-word and full-prompt
  unigram probes were exactly 50%, localizing the shortcut to the unbalanced
  query/context relationship.

Under the controlling queue recalibration's bottom-out rule, this registration
is therefore `FROZEN_NOT_LAUNCHABLE`. No diagnostic cell ran, no result artifact
or scientific metric exists, and no scientific conclusion may be drawn about
interface exposure, representation content, controller competence, or dense
supervision.

The mechanics line's representation-versus-supervision routing question
remains **OPEN**. It may be revisited only through fresh steering and a fresh
registration; this document authorizes no repair, rerun, successor, or launch.
The following independently verified pieces may be reused as design assets in
that future work: S2's grounded registered-rule distractor construction and
shortcut checks, the frozen-substrate and access boundaries, and the
exhaustively exercised branch table.
