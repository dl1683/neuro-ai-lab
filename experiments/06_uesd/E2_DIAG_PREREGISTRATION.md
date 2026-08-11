# E2-DIAG Competence and Learnability Diagnostic Preregistration

**Date:** 2026-08-10  
**Status:** PREREGISTERED / NOT RUN / DIAGNOSTIC ONLY  
**Canonical landing path:** `experiments/06_uesd/E2_DIAG_PREREGISTRATION.md`

## 1. Purpose and controlling boundary

The completed E2 mechanics pilot remains under its controlling post-evidence adjudication:

- **`VOID / INSUFFICIENT_CONFIDENCE_MATCHED_PAIRS_PER_SEED`**;
- the immutable runner-emitted `FAIL` remains execution evidence only;
- the official E2 test split was not inspected;
- endpoint regression, gain retention, selector advantage, matched-pair AUROC, encoder-control test accuracy, and latch outcomes remain not applicable;
- the current 0.5B semantic-ratchet program remains launch-blocked;
- no automatic Direction 2 route, E2 rerun, E2-CERT launch, or threshold change was unlocked.

E2-DIAG investigates the narrower post-mortem diagnosis that the 29.5M-parameter from-scratch substrate received too little effective supervision to learn the hard deduction task. The original 500,000-token run supplied approximately 2,181 examples, 551 optimizer updates, and 0.266 passes through the controller-training corpus before the learning rate reached its minimum. Both recurrent and encoder-control validation losses remained near four-way chance.

E2-DIAG asks, in order:

1. Can the current recurrent model and training pipeline memorize 128 full-hard examples?
2. Can the same pipeline generalize on a deliberately easy one-hop deduction task?
3. Are easy-task outcomes materially controlled by gradient clipping or learning-rate schedule?
4. Does continuing one frozen E2 seed to 2M and 5M tokens produce hard-task competence?

This is a diagnostic program only. It does not test the semantic latch, adjudicate the E2 mechanics hypothesis, alter the prior E2 outcome, satisfy the original mechanics prerequisite, or authorize any full experiment.

## 2. Non-adjudication and non-authorization contract

E2-DIAG:

- **adjudicates nothing** about the semantic-ratchet mechanism;
- **unlocks nothing by itself**, including E2b, Direction 2, E2-CERT, or the 0.5B program;
- cannot rescue, replace, reinterpret, or amend the immutable E2 result;
- cannot inspect or report the original E2 official-test split;
- cannot inspect GSM8K or SVAMP official-test examples;
- trains no critic, confidence calibrator, latch, selector, reranker, or encoder control;
- reports no regression, selector, matched-pair, AUROC, oracle-headroom, or latch metric;
- cannot change the original E2 competence, denominator, PROCEED, FAIL, or VOID thresholds;
- cannot supply publishable mechanism evidence.

A branch outcome below identifies only the next subject that may be taken to steering and written as a fresh preregistration. No successor launches until that fresh registration, its `STATUS.md` decision, implementation, and required review have landed.

## 3. Line-07 priority boundary

E2-DIAG does not change line 07’s scientific, implementation, or GPU-queue priority.

Line 07 retains first claim on the GPU once its frozen launch blockers are resolved. E2-DIAG may run only when doing so does not delay line-07 implementation, review, preflight, or retained generation. If the two lines contend for the same execution window, line 07 wins.

Nothing in E2-DIAG may change line 07’s cohorts, hashes, policies, thresholds, verdict rules, compute cap, or claim language.

## 4. Shared frozen setup

### 4.1 Model

Use the existing E2 common recurrent architecture without structural modification:

- five-layer bidirectional pre-LN input encoder;
- width 512;
- eight attention heads;
- FFN width 2048;
- eight latent plan slots;
- two-layer controller tied across recurrent iterations;
- no absolute iteration embedding;
- one-layer four-choice answer decoder receiving only the projected plan;
- total parameter count within the existing 28–30M envelope.

No new module, auxiliary loss, process supervision, certificate target, tokenizer feature, or architectural ablation is permitted.

### 4.2 Seed and precision

Use model seed `42`, the first frozen E2 seed, for every E2-DIAG model fit. This is a one-seed diagnostic. No seed may be substituted, added, dropped, or selected from observed results.

Use the existing E2 precision and deterministic settings:

- bfloat16;
- TF32 enabled;
- fixed data-order seeds;
- deterministic evaluation;
- workspace-local `HF_HOME=<repo>/.hf_cache`.

### 4.3 Base optimizer

Except in the frozen optimizer matrix, use the original E2 optimizer family and constants:

- AdamW;
- peak learning rate \(3\times10^{-4}\);
- minimum learning rate \(3\times10^{-5}\);
- betas \((0.9,0.95)\);
- epsilon \(10^{-8}\);
- weight decay \(0.1\);
- 5% linear warmup;
- cosine decay by processed training budget;
- gradient clip norm \(1.0\);
- target 1,024 non-padding input-plus-answer tokens per update;
- cross-entropy only;
- exactly balanced repeating training horizons \(T\in\{1,2,4\}\).

### 4.4 Integrity

Before GPU training, the runner must verify and record:

- code and config hashes;
- model parameter count;
- model-initialization hash;
- dataset and split hashes;
- tokenizer round-trip and unknown-token checks;
- exact label counts;
- example and symbolic-skeleton disjointness where required;
- deterministic repeated evaluation on a fixed sample;
- checkpoint identity and optimizer-state identity for continuation;
- the absence of any official-test loader or evaluation call in the diagnostic path.

An integrity failure has precedence over every scientific or routing result and yields `VOID_NO_ROUTE`.

## 5. Compute and stopping contract

The complete E2-DIAG program is capped at **90 minutes of cumulative GPU wall time**.

For this contract, GPU wall time begins when a process first initializes CUDA and ends when that process exits. The result must report both per-run and cumulative wall time, along with peak allocated and reserved VRAM.

Additional rules:

- every individual training invocation must terminate within **30 minutes**;
- the runner must contain its own wall-time stop and must not depend on an external process kill;
- no agent may kill a process it did not spawn;
- no run may be repeated because its result is unfavorable;
- unused time from a stopped branch cannot be used to add an unregistered run;
- a run interrupted by hardware failure, integrity failure, or a wall-time cap does not contribute a scientific miss;
- if the next registered run would exceed the remaining 90-minute budget, it must not launch;
- an incomplete required stage yields `VOID_NO_ROUTE / COMPUTE_CAP_EXHAUSTED` or the applicable operational reason.

The pre-launch review gate applies before execution. The reviewer must confirm task construction, label accounting, checkpoint resumption, learning-rate behavior, wall-time enforcement, absence of selectors, and absence of official-test access.

## 6. Stage 0 — Tiny-set memorization gate

### 6.1 Question

Can the current recurrent model and training implementation fit a tiny sample of the original full-hard task?

A failure here means that additional generalization training is not interpretable. No later diagnostic is justified.

### 6.2 Dataset

Generate the original frozen E2 hard-task `controller_train` split using the existing generator, tokenizer, templates, and seed.

Construct the 128-example memorization set as follows:

1. group examples by answer position;
2. within each answer-position group, sort by the existing canonical symbolic-example hash;
3. take the first 32 examples from each of the four groups.

The resulting set contains exactly 128 full-hard examples and exactly 32 examples per answer position. It retains the original 2–5-hop deductions, full entity and rule ranges, pseudonames, counterfactual construction, and clean/late/directional-lure mixture present in the selected records.

The 128 example IDs and the ordered-set hash are frozen before training.

### 6.3 Training and metric

Initialize the recurrent model from scratch with seed `42`. Repeatedly train on only these 128 examples for at most 3,000 optimizer updates using the base optimizer.

Evaluate deterministic exact four-choice training accuracy at \(T=4\) every 100 updates and at the final update. Record:

- correct count out of 128;
- exact accuracy;
- cross-entropy sum and mean, with example denominator;
- updates and processed tokens;
- first update at which the threshold is reached, if any;
- pre-clip gradient-norm distribution;
- clipped-update count over total updates;
- wall time and peak VRAM.

### 6.4 Frozen gate

**PASS:** \(T=4\) training accuracy is at least 95%, meaning at least **122/128** correct, at or before update 3,000.

**STOP:** fewer than **122/128** are correct after 3,000 completed updates.

**VOID:** the registered endpoint is not reached because of an integrity, checkpoint, hardware, accounting, or compute-cap failure.

Only PASS advances to Stage 1. STOP ends E2-DIAG immediately and assigns the routing token `KILL_FROM_SCRATCH_LINE`.

This kill is scoped to the current E2 from-scratch 30M recurrent substrate and training pipeline. It does not establish that recurrent reasoning, semantic latching, or pretrained-controller successors are impossible.

## 7. Stage 1 — One-hop learnability probe

### 7.1 Question

Can the same architecture generalize when deduction has a clear gradient foothold?

### 7.2 Frozen easy task

Construct balanced one-hop unary deductions with:

- exactly one supporting unary fact;
- exactly one unary implication required to derive the answer;
- four answer choices;
- no binary relations;
- no conjunction rules;
- no lures;
- no counterfactual reversal;
- fixed rendering templates;
- a shared, fixed entity vocabulary across train and validation;
- exact prompt and symbolic-example disjointness between train and validation;
- exact label balance within each split.

Use:

- 2,048 training examples: 512 per answer position;
- 1,024 diagnostic-validation examples: 256 per answer position.

Every example must be symbolically verified before training. These diagnostic splits are not the original E2 test split.

### 7.3 Training and metrics

Initialize a fresh recurrent model with seed `42`. Train for at most 3,000 optimizer updates using the base optimizer and the same balanced \(T\in\{1,2,4\}\) horizon schedule.

At fixed 100-update intervals, report \(T=4\):

- training correct count and denominator;
- validation correct count and denominator;
- training and validation cross-entropy sums and means;
- answer-position confusion matrix;
- answer-frequency distribution;
- clipped-update count over total updates;
- processed examples, updates, tokens, wall time, and peak VRAM.

### 7.4 Frozen classification

**ONE_HOP_LEARNED:** validation accuracy reaches at least 90%, meaning at least **922/1,024** correct, by update 3,000.

**ONE_HOP_NOT_LEARNED:** the run completes 3,000 updates without reaching **922/1,024**.

**VOID:** the registered endpoint is unavailable for an operational or integrity reason.

A valid Stage-1 result advances to Stage 2 whether or not the 90% threshold is reached. Stage 2 determines whether the easy-task conclusion depends on the two preregistered optimizer factors.

## 8. Stage 2 — Easy-task optimizer 2×2

### 8.1 Scope

The optimizer comparison uses only:

1. the 128-example full-hard memorization task; and
2. the balanced one-hop train/validation task.

It may not train on the full hard generalization corpus, inspect the original E2 test split, continue the original E2 checkpoint, or fit any selector.

### 8.2 Frozen matrix

Run the following matrix:

| Cell | Gradient clip | Learning-rate schedule |
|---|---:|---|
| A | 1 | cosine |
| B | 10 | cosine |
| C | 1 | plateau |
| D | 10 | plateau |

Cell A is exactly the completed Stage-0 and Stage-1 configuration and is not repeated.

For the plateau schedule:

- use the same 5% linear warmup to \(3\times10^{-4}\);
- hold \(3\times10^{-4}\) constant for the remainder of the 3,000 updates;
- do not decay to the original minimum.

All other optimizer, architecture, initialization, data-order, precision, batch-token, loss, and evaluation settings remain identical. Cells B–D use the same initial model weights and data order as Cell A for each task.

### 8.3 Cell metrics

For each cell, report:

- memorization correct count out of 128 at \(T=4\);
- update at first attainment of 122/128, or `not_reached`;
- one-hop validation correct count out of 1,024 at \(T=4\);
- update at first attainment of 922/1,024, or `not_reached`;
- train and validation CE sums, means, and denominators;
- pre-clip gradient-norm summaries;
- clipped-update numerator and update denominator;
- processed tokens, updates, wall time, and peak VRAM.

A cell is **EASY-COMPETENT** only if it satisfies both:

- memorization accuracy at least 122/128;
- one-hop validation accuracy at least 922/1,024.

### 8.4 Optimizer classification and tie-break

Classify the matrix as:

- **OPTIMIZER-ROBUST:** all four cells are EASY-COMPETENT;
- **OPTIMIZER-SENSITIVE:** one to three cells are EASY-COMPETENT;
- **EASY-GENERALIZATION-ABSENT:** no cell is EASY-COMPETENT.

If at least one cell is EASY-COMPETENT, choose one informational easy-task winner using this fixed lexicographic rule:

1. highest one-hop validation correct count;
2. lowest one-hop validation mean CE;
3. fewest updates to 122/128 memorization accuracy;
4. prefer clip 1 over clip 10;
5. prefer cosine over plateau.

This winner is informational only. It may be named in a future fresh preregistration, but it does not alter Stage 3.

### 8.5 Advancement

- `OPTIMIZER-ROBUST` or `OPTIMIZER-SENSITIVE` advances to Stage 3.
- `EASY-GENERALIZATION-ABSENT` stops E2-DIAG and routes to `REGISTER_PRETRAINED_SUBSTRATE`.
- A Stage-2 VOID yields `VOID_NO_ROUTE`.

## 9. Stage 3 — Hard-task competence continuation ladder

### 9.1 Question

Was the original E2 substrate stopped before learning the full-hard task?

### 9.2 Starting state

Use only the original E2 recurrent checkpoint for seed `42` at exactly 500,000 processed non-padding input-plus-answer tokens.

Before continuation, verify that the model, optimizer, data-order, processed-token, and checkpoint hashes match the identities recorded by the retained E2 run. If the exact checkpoint and optimizer state are unavailable or do not match, assign:

`VOID_NO_ROUTE / START_CHECKPOINT_UNAVAILABLE_OR_MISMATCHED`

The checkpoint may not be reconstructed by rerunning the first 500,000 tokens.

### 9.3 Continuation protocol

Continue training on the unchanged original hard `controller_train` split with:

- the same recurrent architecture;
- the same seed and data order;
- the same optimizer state;
- the same \(T\in\{1,2,4\}\) schedule;
- the same CE-only loss;
- the same 1,024-token update target;
- the same clip norm of 1;
- the learning rate held at the already-reached minimum \(3\times10^{-5}\).

Do not rewind warmup, renormalize the original cosine schedule, restart Adam moments, or apply the Stage-2 winner. This is a continuation diagnostic, not a hard-task optimizer retuning experiment.

Run the ladder as two resumable invocations:

1. 500K to exactly 2M cumulative training tokens;
2. 2M to exactly 5M cumulative training tokens.

Write recovery checkpoints during execution, but the only scientific checkpoints are 500K, 2M, and 5M. The 500K values are imported from the existing calibration record rather than regenerated.

Even if the 2M checkpoint reaches the competence threshold, continue to 5M unless a registered operational cap prevents completion.

### 9.4 Evaluation data

Evaluate only on the original 1,024-example `selector_calibration` split. This split is reused solely as a diagnostic-validation set.

Do not access:

- the original 4,096-example E2 test split;
- GSM8K official test;
- SVAMP official test;
- any selector-harvest label for selector fitting.

### 9.5 Competence-only metrics

At 500K, 2M, and 5M, report deterministic no-latch metrics at fixed horizons \(T=\{1,2,4,8,16\}\):

- correct count and exact accuracy at every horizon;
- CE sum and mean at every horizon;
- \(T=4\) correct-count change from the prior checkpoint;
- \(T=4\) accuracy gain over \(T=1\);
- answer-frequency distribution;
- per-proof-depth and per-lure correct counts and denominators;
- processed tokens and optimizer updates;
- gradient-norm and clipping summaries;
- wall time and peak VRAM.

No best horizon may replace \(T=4\) in the advancement calculation. No confidence model, critic, latch, selector, regression analysis, matched-pair construction, or AUROC calculation is permitted.

### 9.6 Frozen competence threshold

The continuation supports the E2b-registration route only if the **5M checkpoint** satisfies both:

1. \(T=4\) validation accuracy is at least 60%, meaning at least **615/1,024** correct;
2. \(T=4\) accuracy exceeds \(T=1\) accuracy by at least five absolute percentage points, meaning the \(T=4\) correct count exceeds the \(T=1\) correct count by at least **52 examples**.

These thresholds mirror the scale of the original E2 competence floor without adjudicating it on the uninspected test set.

Outcomes at 2M are diagnostic trajectory evidence only. Passing at 2M and missing at 5M does not qualify for the E2b route.

## 10. Frozen branch table

Integrity and operational VOIDs have precedence over all rows below.

| Observed diagnostic outcome | Frozen route token | Meaning |
|---|---|---|
| Stage 0 completes below 122/128 | `KILL_FROM_SCRATCH_LINE` | End the current 30M from-scratch E2 substrate line. Scaling its generalization budget is unjustified when it cannot memorize 128 hard examples. |
| Stage 0 passes, but no Stage-2 cell reaches both 122/128 memorization and 922/1,024 one-hop validation | `REGISTER_PRETRAINED_SUBSTRATE` | The model can memorize but does not obtain reliable easy-task generalization under the frozen optimizer matrix. The next eligible subject is a fresh pretrained-substrate successor. |
| At least one Stage-2 cell is EASY-COMPETENT, and Stage 3 completes with the 5M checkpoint satisfying both competence thresholds | `REGISTER_E2B` | The cheap diagnostics support writing a fresh full E2b preregistration. They do not authorize or constitute E2b. |
| At least one Stage-2 cell is EASY-COMPETENT, but the completed 5M checkpoint misses either competence threshold | `REGISTER_LEARNABILITY_STAIRCASE` | An easy foothold exists, but compute-only continuation does not establish the hard-task competence required to expose latch mechanics. |
| Any required stage is invalid, incomplete, hash-mismatched, interrupted, or blocked by the 30/90-minute caps | `VOID_NO_ROUTE` | No scientific branch is selected. The 0.5B program stays blocked and the incomplete diagnostic may not be interpreted as a miss. |

No metric may be rounded across a threshold. Integer counts control.

## 11. Successor meanings

### 11.1 `REGISTER_E2B`

This token permits only the drafting and steering review of a fresh E2b preregistration. The candidate described by the post-mortem is:

- two seeds;
- training from scratch;
- unchanged hard generator and recurrent architecture;
- a larger fixed competence budget, provisionally 10M tokens;
- checkpoints at 0.5M, 1M, 2M, 5M, and 10M;
- competence validity before selector fitting or test inspection;
- fresh denominator, baseline, review, and outcome rules.

The future registration may use the Stage-2 easy-task optimizer winner only if that choice is declared before E2b training. E2-DIAG checkpoints cannot become E2b evidence or substitute for either registered seed.

### 11.2 `REGISTER_LEARNABILITY_STAIRCASE`

The fresh staircase registration must proceed in this conceptual order:

1. balanced one-hop unary deductions with fixed templates, shared entity vocabulary, and no lures;
2. 2–3-hop mixed-rule deductions;
3. reintroduction of name- and skeleton-disjoint generalization;
4. the original 2–5-hop counterfactual and lure task.

Each stage requires a frozen budget, held-out competence threshold, advancement rule, and label-balance audit. Failure to reach the original hard-task competence floor leaves mechanics unmeasurable and the 0.5B program blocked.

### 11.3 `REGISTER_PRETRAINED_SUBSTRATE`

The fresh successor must test whether a frozen small pretrained text substrate supplies the missing representation and deduction foothold. It must include:

- a frozen pretrained substrate;
- a separately trained recurrent controller and readout;
- the hard deduction task;
- a parameter-, supervision-, and compute-appropriate non-recurrent control;
- competence gates before selector fitting;
- fresh review and outcome rules.

Success would not validate E2 or automatically transfer to the 0.5B program. Failure would be scoped to the registered substrate and interface.

### 11.4 `KILL_FROM_SCRATCH_LINE`

This token ends further scaling of the current from-scratch 30M E2 substrate and training pipeline. It is not a claim that all from-scratch models, recurrent controllers, or semantic latches are impossible.

Any subsequent work must enter through a materially different, freshly preregistered substrate, such as pretraining or verified dense process supervision.

## 12. Claim-language and denominator contract

All repository house rules apply.

Every reported rate must include:

- numerator;
- denominator;
- exact percentage;
- the population and checkpoint to which it applies.

For held-out binomial accuracies, report an observed point estimate and a 95% Wilson interval. Training-set memorization accuracy is descriptive and must be labeled as training accuracy, not generalization.

Additional rules:

- zero-denominator quantities are `undefined`, never zero;
- missing or uninspected quantities are `not applicable`, never failed;
- means must report their example, token, or update denominator;
- clipped-gradient fractions must report clipped updates over total completed updates;
- results from one model seed cannot be described as seed-robust or generally reproducible;
- easy-task success does not establish hard-task competence;
- hard-task competence does not establish latch quality;
- a continuation trend does not establish that more compute alone is sufficient at other seeds or scales;
- the tokens `PASS`, `STOP`, and `EASY-COMPETENT` are internal stage-control labels only;
- no E2-DIAG outcome may be described as an E2 `PROCEED`, `FAIL`, or mechanics `VOID`;
- no narrative may claim that overthinking was prevented, regression was reduced, selectors worked, or latent reasoning was validated.

Permitted top-level language is limited to statements such as:

> “In a one-seed diagnostic, the current substrate did or did not cross the preregistered competence threshold by 5M training tokens.”

or:

> “The diagnostic routed the next design discussion to a fresh E2b, staircase, pretrained-substrate, or line-kill registration.”

## 13. Result and landing contract

### 13.1 Preregistration landing

This document becomes effective only when the same pre-data change also:

1. appends an `E2_DIAG_PREREGISTERED` entry to `experiments/ledger.jsonl`;
2. adds a preregistered/not-run entry to `experiments/EXPERIMENTS.md`;
3. updates `STATUS.md` to authorize only this bounded diagnostic while preserving:
   - the original E2 controlling VOID;
   - the blocked 0.5B launch;
   - the absence of an automatic Direction 2 route;
   - line 07’s priority.

The registration entry must state that no diagnostic run or metric existed when the protocol was frozen.

### 13.2 Execution landing

A completed or terminally stopped E2-DIAG suite writes exactly one immutable result artifact:

`experiments/06_uesd/results/exp_e2_diag.json`

The artifact must use atomic no-clobber publication and contain:

- every stage status and reason;
- the final route token;
- all raw numerators and denominators;
- the optimizer matrix;
- checkpoint identities;
- code, config, generator, split, and checkpoint hashes;
- per-run and cumulative compute;
- all protocol deviations or operational stops;
- confirmation that no official test or selector path was accessed.

In the same evidence-landing block:

- append the run to `experiments/ledger.jsonl`;
- update `experiments/EXPERIMENTS.md`;
- update `STATUS.md`;
- update `docs/UNIFIED_ERROR_SPACE.md`;
- extend `experiments/06_uesd/audit_uesd_claims.py` for the new artifact and claim boundaries.

Checkpoints, raw logs, and caches remain untracked provenance and are not committed.

## 14. Frozen precedence and prohibition on adaptive rescue

The decision precedence is:

1. integrity or operational `VOID_NO_ROUTE`;
2. Stage-0 stop;
3. Stage-2 easy-task classification;
4. Stage-3 competence classification;
5. final route token.

After any result is observed, the following are prohibited:

- changing a threshold;
- adding updates or tokens;
- changing the seed;
- restarting from a different checkpoint;
- substituting an optimizer cell;
- moving the competence horizon away from \(T=4\);
- adding an intermediate curriculum stage;
- inspecting an official test;
- fitting selectors;
- repeating an unfavorable run;
- choosing a branch from an unregistered metric.

Any such change requires a fresh preregistration and cannot be represented as E2-DIAG.

---

**Frozen before E2-DIAG execution. No result is claimed here.**

## 15. Post-VOID operational-repair authorization

**Governance date:** 2026-08-10
**Nature:** owner-directed reopening after the terminal operational VOID; not a
pre-data amendment and not a change to the frozen scientific configuration.

The first Stage-0 invocation reached all 3,000 updates but failed during
post-endpoint result serialization because `torch.quantile` received a
quantile tensor whose dtype did not match the float64 gradient-norm input.
The owner has explicitly authorized exactly one operationally repaired Stage-0
invocation. This is not a repeat because the observed result was unfavorable;
it repairs a post-endpoint operational failure covered by Section 5. The
original `exp_e2_diag.json` remains immutable operational evidence.

The repair invocation must preserve the registered seed, selected 128
examples, initialization, optimizer, horizon schedule, 3,000-update limit,
122/128 gate, and every access prohibition. It may change only:

- the quantile serialization path so input and quantile tensors share a dtype;
- diagnostic-only telemetry, sampled every 100 updates, for disjoint parameter
  groups `encoder`, `controller`, `plan_slots`, `prefix_projector`,
  `answer_decoder`, and `readout_head`;
- prediction-flip counts, full-set cross-entropy decrease, and readout-head
  weight-delta norms.

The added telemetry is informational and cannot choose a scientific branch.
The registered Stage-0 count still controls the branch table. The one repaired
invocation lands atomically and without clobbering the prior artifact at:

`experiments/06_uesd/results/exp_e2_diag_stage0_instrumented.json`

No second repair invocation, resume, Stage 1 launch, E2 runner change, or
line-07 change is authorized by this addendum.
