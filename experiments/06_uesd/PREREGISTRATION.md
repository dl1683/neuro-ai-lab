# Semantic-Ratchet Transient Solver Preregistration

Status: **PREREGISTERED / NOT YET RUN**

This is the single canonical preregistration for the fresh semantic-ratchet transient-solver hypothesis. It does not reopen or continue the closed D38-D40 fixed-point convergence arc. No result is claimed here.

## Hypothesis

A frozen small language model augmented with a variable-depth latent controller and outcome-trained best-state memory can convert additional recurrent computation into improved exact-answer accuracy while preventing overthinking, and can outperform compute-matched independent sampling plus learned reranking.

## Target Headline (Conditional on CONFIRM)

**“A tiny local reasoner can think longer without thinking itself out of a correct answer—and beats sampling more answers with the same compute.”**

## Prior Art and Differentiation

Geiping et al., [“Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach” (arXiv:2502.05171)](https://arxiv.org/abs/2502.05171), already establish that a weight-tied recurrent block can improve reasoning quality as latent loop count increases. Therefore “more loops improve answers” is prior art, not this experiment's novelty claim. Fu et al., [“Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models” (arXiv:2511.08577)](https://arxiv.org/abs/2511.08577), already identify latent overthinking and use a learned token-level decider to apply extra iterations selectively. This preregistration instead tests whether an outcome-trained memory can choose the best state along one problem-level latent trajectory, using trajectory geometry and problem-conditioned latent content rather than decoder-confidence features, decode only one final response, prevent correct-to-wrong regression, and beat compute-matched independent sampling with a learned reranker. Success is not novelty by mechanism label alone; it requires every **CONFIRM** condition below.

## Scope and Sequence

Direction 1 remains ranked first, but the full 0.5B run is conditional on two cheap gates: task-band validation and mechanism validation. The task-band gate runs before training. The embedded mechanics pilot is a design gate before the full semantic-ratchet program. No full Direction 1 work starts after a task-band **VOID**, and no full 0.5B run starts after a mechanics-pilot **FAIL** or **VOID**.

Any run expected to exceed 30 minutes requires an independent pre-training code review of the full pipeline, with every blocking finding resolved before launch.

## Primary Task and Transfer Set

Use **GSM8K, standard five-shot exact-numeric evaluation**, with SVAMP as the held-out cross-family transfer set.

The checkpoint is identified in tracked files only as `base-A`. Its exact identifier is kept in the gitignored local manifest `experiments/06_uesd/_local_manifest.md`.

## Sub-Hour Task-Band Gate

Before training:

- Evaluate exactly 256 held-out GSM8K examples.
- Fixed five-shot prompt, greedy decoding, fixed extraction code.
- Maximum 256 generated tokens.
- Record exact-answer accuracy, extraction-failure rate, response length, and answer-frequency distribution.
- Expected wall time: 20–45 minutes.

**PASS:**

- Correct count is 26–217 inclusive, corresponding to the stipulated \([10\%,85\%]\) band.
- At least 40 correct and 40 incorrect responses exist, ensuring usable critic populations.
- Extraction failures are ≤5%.

**ABORT-AND-SWAP:**

- If below 10% or fewer than 40 correct: swap to SVAMP using the same protocol.
- If above 85% or fewer than 40 incorrect: swap to GSM-Hard.
- Rerun the same 256-example gate once.

**VOID:**

- The fallback also misses the band.
- Extraction failures exceed 5% after one parser repair.
- Contamination or answer leakage is detected.

No full Direction 1 work starts after a VOID.

## Frozen-Base Architecture

The 0.5B base is entirely frozen:

- token embeddings frozen;
- all backbone transformer blocks frozen;
- LM head frozen;
- no base LoRA.

Trainable components:

- shared two-layer recurrent controller: approximately 12–18M parameters;
- soft-prefix projector: approximately 2–4M;
- progress critic: approximately 1–2M;
- total trainable overhead capped at 5% of base parameters.

The frozen base encodes the question. The controller iteratively refines a compact latent plan and maps each horizon to a soft prefix for the frozen decoder. The critic selects the best latent state; the decoder generates exactly one public-facing response from that banked state.

BF16 is justified despite the normal quantization preference because gradients must pass through the frozen decoder into the soft prefix, and quantization-dependent gradient behavior would introduce another attribution variable.

## Embedded Mechanics Pilot

Use a 30M recurrent transformer on shortcut-resistant generated natural-language deductions. Train to horizons \(T\in\{1,2,4\}\); evaluate through \(T=16\).

Three arms share everything except selection:

1. final-horizon/no latch;
2. confidence latch;
3. latent progress critic plus latch.

Two seeds, 0.5M tokens per arm.

**PROCEED:**

- No-latch arm exhibits at least 10% correct-to-wrong regressions among examples correct at an earlier horizon; otherwise the test cannot measure overthinking.
- Semantic latch reduces that regression rate by at least 80%.
- It retains at least 90% of the no-latch arm’s \(T=1\) to best-horizon gain.
- Critic exceeds confidence-latch final accuracy by ≥3 points.
- Critic AUROC on confidence-matched pairs is ≥0.70.

**FAIL:**

A valid pilot misses any PROCEED condition. The 0.5B run does not launch. Direction 2 becomes the next full experiment for causal isolation.

**VOID:**

- Encoder-only control exceeds 80%.
- No-latch correct-to-wrong regression is below 10%.
- Dataset shortcuts, seed mismatch, or accounting failure are found.

This pilot is a design gate, not publishable evidence and not a substitute for full Direction 2.

Because its expected duration exceeds 30 minutes, it still requires the independent pre-training code review.

## Data Splits

Partition the public training data before any adaptation:

- 70% controller training;
- 15% critic-harvest;
- 15% critic calibration/validation;
- official test set untouched until final evaluation;
- SVAMP untouched for transfer evaluation.

No problem may cross partitions through paraphrases or shared identifiers.

## Critic Provenance

The critic must exploit information confidence does not contain, and the data construction must force it to do so.

### State Harvesting

Harvest reachable states from controller checkpoints at:

- initialization;
- 10%;
- 30%;
- 60%;
- 100% of controller training.

At each checkpoint and harvest problem:

- horizons \(T=\{1,2,4,8,16\}\);
- one greedy rollout;
- two fixed-seed stochastic rollouts;
- retain latent plan \(z_t\), update \(\Delta z_t\), prompt representation, candidate response, decoder confidence, and exact correctness.

This initial replay corpus is off-policy relative to the final controller but covers failure modes that disappear late in training.

During critic training, refresh 50% of each epoch’s corpus with rollouts from the current controller. The other 50% remains checkpoint replay. Thus:

- checkpoint replay supplies diversity and regressions;
- current-policy data prevents train/inference distribution mismatch—the exact error that killed the flow head.

### Pair Construction

Pairs are always within the same problem where possible:

- wrong horizon versus later correct horizon;
- correct horizon versus later regressed horizon;
- high-confidence wrong versus confidence-matched correct;
- low-confidence correct versus confidence-matched wrong.

At least 50% of critic minibatches must be confidence-matched:

- same confidence decile;
- absolute normalized confidence difference ≤0.02;
- opposite correctness labels.

Oversample high-confidence wrong and low-confidence correct states. These are precisely the cases ordinary confidence cannot solve.

### Critic Information Boundary

The latent critic is prohibited from seeing:

- raw decoder logits;
- softmax margin;
- response log-probability;
- entropy;
- sampled-answer frequency.

It may see:

- problem-conditioned latent plan \(z_t\);
- update \(\Delta z_t\);
- cosine alignment between consecutive updates;
- cross-horizon latent agreement;
- frozen prompt representation;
- step index.

The additional information source is therefore the **trajectory geometry and problem-conditioned latent content**, not output confidence.

A separate confidence baseline receives margin, entropy, mean token log-probability, and sample agreement. A separate sequence reranker receives the decoded candidate and prompt. Both train on the same number of correctness-labeled candidates.

### Critic Gates

Before it controls the latch:

- Overall correctness AUROC ≥0.75.
- AUROC on confidence-matched pairs ≥0.70.
- AUROC advantage over confidence-only ≥0.05.
- Critic-latch selection accuracy ≥3 points above confidence latch on critic validation.
- The on-policy half alone must achieve AUROC ≥0.70.

A valid miss on any criterion is **KILL for the current semantic-ratchet design**. The full test must not proceed with a critic known to be a confidence surrogate.

## Exact Baseline Definitions

- **Frozen base, no controller:** `base-A` uses the frozen five-shot prompt and the same extraction and evaluation code, with one greedy decoded response and no learned controller, critic, or reranker.
- **Semantic-ratchet controller:** the variable-depth controller banks the preregistered horizon states, the provenance-gated latent critic chooses one banked state, and exactly one public-facing response is decoded from that state.
- **Same checkpoint, final horizon/no latch:** reuse the semantic-ratchet checkpoint and always select the last available horizon state; do not use the critic for selection.
- **Same checkpoint, confidence latch:** reuse the semantic-ratchet checkpoint and select the horizon with the strongest confidence-baseline score. The confidence score uses only margin, entropy, mean token log-probability, and sample agreement; it is calibrated on the critic calibration/validation split and frozen before final evaluation.
- **Same checkpoint, conventional early exit:** reuse the semantic-ratchet checkpoint and stop at the first horizon whose confidence-baseline score crosses a threshold selected on the critic calibration/validation split; if none crosses, use the final horizon. Freeze the threshold before final evaluation.
- **Parameter- and training-token-matched fixed-depth controller:** train a controller with the same trainable-parameter cap, data split, correctness supervision, total controller-training tokens, seeds, and frozen `base-A`, but with a single fixed recurrent depth rather than variable-depth training or best-state memory. Compare it at the preregistered equal-compute budgets.
- **Compute-matched stochastic best-of-\(N\) with confidence reranking:** draw independent stochastic responses from frozen `base-A`, use the confidence-baseline features to choose one response, and match the semantic-ratchet inference budget by total layer-token FLOPs.
- **Compute-matched stochastic best-of-\(N\) with learned sequence reranking:** draw the same kind of independent responses from frozen `base-A`; a learned reranker that sees only the prompt and decoded candidate selects one. Its correctness-label budget equals the ratchet critic's.
- **Oracle best-of-\(N\):** report whether any independently sampled candidate is exactly correct at each matched budget. This is an unavailable upper bound and never a deployable baseline.

No-latch, confidence-latch, final-horizon, and early-exit arms reuse the semantic-ratchet checkpoint; they are inference interventions, not six independent 0.5B fine-tunes.

## FLOP and Budget Matching

- Match by total layer-token FLOPs within 5%; report wall time and decoded tokens separately.
- Count frozen-base question encoding, all recurrent-controller steps, soft-prefix decoding, candidate generation, and reranker/critic inference in the method's total inference compute.
- Use the same fixed evaluation examples, generation cap, extraction code, and seed schedule for every matched comparison.
- Where the FLOP budget permits fractional \(N\), distribute \(N\) and \(N+1\) samples across examples so the dataset-level compute matches rather than rounding every example down.
- Freeze the example allocation and matched-budget table before official-test evaluation.
- Give the learned sequence reranker the same number of correctness-labeled candidates as the ratchet critic. Report oracle best-of-\(N\) only as an unavailable upper bound.

Without compute-matched sampling plus reranking, the trivial dismissal survives. Direction 1 must beat the strongest practical version of that alternative, not merely confidence selection.

**Revised decisive criterion:** semantic ratchet accuracy must exceed learned best-of-\(N\) reranking by at least 3 absolute points, with a paired-bootstrap 95% confidence interval whose lower bound is above zero. A valid run missing this threshold is **KILL**, even if the ratchet beats every other baseline.

That makes the claim precise: not “we reranked horizons,” but “one latent trajectory and one decoded response outperform spending the same compute on independent sampled responses.”

## Revised Compute Budget

| Component | Training/candidate tokens | Seeds | Estimated 5090 time |
|---|---:|---:|---:|
| Task-band pilot | ≤0.10M input + ≤0.07M generated | 1 | 0.3–0.8h |
| 30M latch-mechanics pilot | 0.5M per arm × 3 arms | 2 | 1.5–2.5h |
| Semantic-ratchet controller | 3.0M per seed | 2 | 8–12h |
| Matched fixed-depth controller | 3.0M per seed | 2 | 6–9h |
| Progress-critic harvest | ≈1.2M generated/cached | shared | 2–4h |
| Best-of-\(N\) reranker | 0.75M per seed | 2 | 1–2h |
| All inference ablations and final evaluation | no training | 2 | 3–5h |
| **Total** | **≈15–17M processed/labeled tokens** |  | **22–35h** |

The earlier 14–20h estimate was too optimistic. The full program should be preregistered as approximately **one to one-and-a-half GPU-days**, excluding review or repairs.

## Final Outcome Tokens

**CONFIRM** only if all are true:

- \(T=1\) to \(T=8\) gain ≥7 absolute points.
- No aggregate accuracy drop greater than 1 point from \(T=8\) through \(T=32\).
- Correct-to-wrong regression among examples correct at an earlier horizon ≤2%.
- Ratchet exceeds learned compute-matched best-of-\(N\) by ≥3 points, with paired-bootstrap 95% CI lower bound \(>0\).
- Ratchet exceeds matched fixed-depth and early-exit baselines by ≥3 points at some preregistered equal-compute budget.
- Critic satisfies every provenance gate above.
- At least 50% of the in-domain improvement survives on SVAMP.
- Trainable overhead ≤5%.
- Only one final response is decoded by the ratchet at inference.

**KILL** means the run is valid but any CONFIRM condition fails. There is no post-hoc “promising” category.

**VOID** is reserved for:

- task-band failure after one fallback;
- leakage or split contamination;
- extraction failure above 5%;
- unmatched compute or token budgets;
- missing seed;
- broken baseline;
- critic/controller train–inference distribution mismatch;
- unresolved review finding;
- hardware failure preventing the preregistered grid.

## AMENDMENTS

### 2026-08-09 — Mechanics-pilot design-gate clarifications (pre-data)

This section was appended before mechanics-pilot implementation or execution. It amends ambiguities in the committed preregistration without silently rewriting its original normative text. Where this section conflicts with earlier wording, this section controls.

#### Shared controller checkpoint

For each model seed, train exactly one common recurrent controller checkpoint on 500,000 processed non-padding input-plus-answer tokens. Reuse that exact checkpoint for the final-horizon/no-latch, confidence-latch, and latent-critic-latch conditions. These are selector interventions, not independently trained controller arms.

Record both:

- `shared_controller_tokens = 500000`;
- `logical_arm_exposure = 500000`.

Do not report 1.5M unique controller-training tokens per seed. Selector fitting and critic fitting are recorded separately from controller-training tokens.

#### Meaning of semantic latch

Every occurrence of “semantic latch” in the mechanics-pilot PROCEED criteria means the latent progress critic plus latch. It does not mean the confidence latch.

#### Encoder-only control

Use a strong parameter-matched non-recurrent control with:

- 9 bidirectional pre-LN transformer encoder layers;
- width 512;
- 8 attention heads;
- FFN width 2048 with GELU;
- the same tokenizer, rendered examples, split, 500,000-token training budget, optimizer family, model seeds, and four-way answer interface as the recurrent model;
- total trainable parameters within 5% of the common recurrent model.

Select its checkpoint once using the fixed validation rule and evaluate it once on the fixed test set. Report exact parameter counts and per-seed results.

The pilot is VOID if either encoder-control seed exceeds 80.0% test accuracy.

#### Calibration-frozen earlier horizon and regression denominator

Using only the calibration split, define one shared earlier horizon

\[
t^\* =
\arg\max_{t\in\{1,\ldots,15\}}
\frac{1}{2}\sum_s A_{\mathrm{no},s}^{\mathrm{cal}}(t),
\]

where \(s\) indexes the two model seeds. Break ties toward the smaller horizon. Freeze \(t^\*\) before inspecting any test result. The same \(t^\*\) is used for both the \(H=16\) primary adjudication and any permitted \(H=32\) adjudication.

For each seed \(s\), define

\[
S_s=\{i:\hat y^{\mathrm{no}}_{i,s,t^\*}=y_i\}.
\]

For endpoint \(H\in\{16,32\}\), define

\[
R_{\mathrm{no},s}(H)=
\frac{
\#\{i\in S_s:\hat y^{\mathrm{no}}_{i,s,H}\ne y_i\}
}{
|S_s|
},
\]

and

\[
R_{\mathrm{critic},s}(H)=
\frac{
\#\{i\in S_s:\hat y^{\mathrm{critic}}_{i,s,H}\ne y_i\}
}{
|S_s|
}.
\]

The critic prediction at endpoint \(H\) is the critic-selected state among horizons \(1,\ldots,H\). Pooled rates concatenate the corresponding per-example records across both seeds; rates are never averaged without their numerators and denominators.

When \(R_{\mathrm{no}}(H)\ge10\%\), define regression reduction as

\[
1-\frac{R_{\mathrm{critic}}(H)}{R_{\mathrm{no}}(H)}.
\]

#### Competence and sample-size validity floors

Before any PROCEED/FAIL adjudication, require:

- no-latch test accuracy at the calibration-frozen \(t^\*\) of at least 60% in each seed;
- no-latch gain
  \[
  G_s=A_{\mathrm{no},s}(t^\*)-A_{\mathrm{no},s}(1)
  \]
  of at least 5 absolute points in each seed;
- \(|S_s|\ge500\) test examples in each seed;
- at least 20,000 correctness-labeled selector-training states per seed;
- at least 10,000 correctness-labeled selector-calibration states per seed;
- at least 200 qualifying confidence-plus-schedule-matched, opposite-correctness evaluation pairs per seed.

A miss on one of these floors is VOID. Random answer flicker, a weak controller, or an underpowered matched-pair set cannot satisfy the scientific mechanics gate.

At endpoint \(H\), gain retention is

\[
\frac{A_{\mathrm{critic}}(H)-A_{\mathrm{no}}(1)}
     {A_{\mathrm{no}}(t^\*)-A_{\mathrm{no}}(1)}.
\]

#### Step-index fairness

The confidence calibrator receives the same normalized step coordinate \(t/16\) available to the latent critic.

The pilot confidence calibrator therefore receives:

- maximum answer probability;
- top-two probability margin;
- entropy;
- \(t/16\).

The latent critic retains its allowed latent-content and trajectory-geometry features and also receives \(t/16\). It remains prohibited from seeing logits, answer probabilities, margins, entropy, or answer identity.

Both selectors are fitted and frozen before test evaluation. At horizons 17–32, \(t/16\) is allowed to exceed 1; neither selector may be refitted, recalibrated, or retuned using those horizons or any test result.

For the later 0.5B design, the latent critic, confidence latch, and confidence-based early-exit calibrator must receive the same normalized horizon coordinate, using the preregistered maximum selector-calibration horizon as the denominator. Any comparison previously described as “critic versus confidence-only” is interpreted as “critic versus confidence plus the same schedule coordinate.” The critic’s additional information is limited to problem-conditioned latent content and trajectory geometry.

#### Confidence-matched concordance

Confidence matching uses the frozen confidence-plus-schedule calibrator’s predicted-correctness score, including \(t/16\). A qualifying pair must:

- come from the same problem;
- contain one correct and one incorrect state;
- fall in the same calibrated-score decile;
- have an absolute calibrated-score difference no greater than 0.02.

For \(N\) preregistered qualifying pairs, define confidence-matched critic AUROC as

\[
\frac{1}{N}\sum_{j=1}^{N}
\left[
\mathbf{1}(c_j^+>c_j^-)
+\frac{1}{2}\mathbf{1}(c_j^+=c_j^-)
\right],
\]

where \(c_j^+\) and \(c_j^-\) are the critic scores for the correct and incorrect members of pair \(j\). Thus the metric is the critic’s concordance probability after matching on both output confidence and schedule. Pair construction and any fixed subsampling seed must be independent of critic scores.

#### Evaluation through T=32 and formal endpoint selection

Evaluate and record every integer recurrent horizon

\[
T=1,2,\ldots,32.
\]

Horizons 17–32 are inference-only. They do not add controller-training tokens and may not trigger selector refitting, new calibration, generator modification, example selection, or checkpoint selection.

The formal decision endpoint is selected as follows:

1. Apply all integrity, leakage, accounting, encoder-control, competence, denominator, and pair-count validity checks.
2. If \(R_{\mathrm{no}}(16)\ge10\%\) pooled and separately in both seeds, set \(H=16\). Apply every numeric PROCEED condition at \(H=16\). A valid miss is FAIL and may not be rescued using \(H=32\).
3. Only if the regression floor misses at \(H=16\), inspect the already-generated \(H=32\) result.
4. If \(R_{\mathrm{no}}(32)\ge10\%\) pooled and separately in both seeds, set \(H=32\) and apply every numeric PROCEED condition at \(H=32\), using the same frozen \(t^\*\), denominator definition, thresholds, and per-seed-plus-pooled requirements.
5. If the regression floor misses at both endpoints, correct-to-wrong regression is unmeasurable at pilot scale. No intermediate horizon, alternative denominator, selected subset, generator retuning, or changed threshold may be used for adjudication.

At the selected endpoint \(H\), all of the following remain required pooled and separately in both seeds:

- \(R_{\mathrm{no}}(H)\ge10\%\);
- regression reduction at least 80%;
- gain retention at least 90%;
- critic-latch accuracy minus confidence-plus-schedule-latch accuracy at least 3 absolute points;
- confidence-plus-schedule-matched critic AUROC at least 0.70.

#### Consequence of regression-only VOID

If regression remains below 10% at both \(H=16\) and \(H=32\), assign final token `VOID` with reason code `UNMEASURABLE_REGRESSION` only when:

- every integrity and competence validity gate passes;
- the regression denominator and matched-pair floors pass;
- gain retention is at least 90% at \(H=32\);
- critic-latch accuracy exceeds confidence-plus-schedule-latch accuracy by at least 3 absolute points at \(H=32\);
- confidence-plus-schedule-matched critic AUROC is at least 0.70.

This pure regression-only VOID is not evidence against the latch. It does not route to Direction 2 or to an adaptively redesigned mechanics pilot. Conditional on a PASS from the sub-hour task-band gate and satisfaction of the full design and independent pre-training review gates, it permits the 0.5B real-task experiment to proceed, with regression measured there under the full preregistered criteria.

If any independently measurable selector criterion above fails, record FAIL and route to Direction 2; absence of measurable regression does not erase that valid negative. Any other VOID reason—including leakage, shortcut control failure, insufficient competence, insufficient denominator, missing seed, accounting failure, or unresolved review finding—continues to block the 0.5B launch.

The sub-hour task-band gate remains controlling: a pending task-band result permits implementation and review but not launch, and a task-band VOID blocks both the mechanics-pilot launch and the full 0.5B experiment.

### 2026-08-09 — Task-band parser-repair scope
(post-SVAMP-initial-data; conservative adjudication)

This amendment is recorded after the immutable SVAMP initial-parser
attempt. It narrows the permissible parser-repair branch and may not
convert any previously recorded extraction failure into a valid extracted
answer or valid extracted incorrect response.

A parser repair is eligible only when the frozen scored response segment
contains explicit numeric answer content, but a deterministic defect in
recognition, segmentation, or numeric normalization prevents that content
from being extracted. A repair may recover only numeric content already
present in the scored response segment.

A parser repair may not:

- infer or impute missing numeric content;
- recompute an answer from the question;
- consult the gold answer;
- change the prompt, decoding, stopping, cohort, or generated response;
- classify a response containing no numeric answer as a valid extracted
  incorrect response.

A response whose scored segment contains an answer marker but no complete
numeric candidate is a model-empty non-answer. It is recorded as:

- `correct = false`;
- `extraction_failed = true`;
- `valid_extracted_incorrect = false`;
- `model_empty_non_answer = true`.

Model attribution and extraction status are separate properties: identifying
the model as the cause does not turn an unextractable response into an
extracted answer.

The one-parser-repair allowance is applicable only if independent review
identifies at least one extraction failure containing recoverable explicit
numeric answer content. If no such record exists, no qualified parser repair
can be performed; the repair branch is exhausted as inapplicable and the
fallback task-band gate receives terminal `VOID` with reason
`NO_RECOVERABLE_NUMERIC_CONTENT_FOR_PERMITTED_PARSER_REPAIR`. No ceremonial
source change or repeated deterministic generation is required or permitted
to manufacture a repaired attempt.

Applied to the immutable SVAMP initial attempt, all 13 extraction failures
are the literal response `Answer:` and contain zero numeric content.
The controlling accounting therefore remains:

- 66/256 correct;
- 177/256 valid extracted incorrect;
- 13/256 model-empty non-answers and extraction failures;
- 190/256 total exact-answer failures;
- 13/256 = 5.078125% extraction failures.

Because 5.078125% exceeds the preregistered 5.000% ceiling and no eligible
parser defect exists, the task-band verdict is terminal `VOID`. The immutable
initial-miss artifact remains unchanged.

### 2026-08-09 — Successor base-B task-band gate and mechanics independence
(post-base-A evidence; pre-base-B data)

This amendment is recorded after the terminal base-A task-band VOID and
before any base-B generation, smoke evaluation, or canonical evaluation.
It establishes a new independent prerequisite; it does not reinterpret,
pool, or rescue either base-A result.

#### Decision and checkpoint freeze

The successor task-band gate uses one frozen checkpoint identified publicly
as `base-B`. `base-B` must be in the same nominal small-model class as
`base-A`, compatible with the frozen-controller design and the RTX 5090
resource envelope, and must not be selected by comparing newly generated
GSM8K, SVAMP, or other task-band results across candidate checkpoints.

Before any base-B generation:

- record the exact checkpoint identifier, revision, and local content digest
  in the gitignored local manifest;
- freeze the prompt messages, demonstrations, decoding settings, parser,
  cohort-selection implementation, and result path;
- obtain independent pre-launch review of the runner;
- record that no base-B task-band response existed before this amendment.

Only one base-B checkpoint is permitted under this amendment. A miss does
not authorize base-C, another cohort, another task, or another fallback
without returning to steering.

#### Task and frozen protocol

Evaluate exactly 256 held-out examples from the revision-pinned GSM8K
official test split:

- dataset: `openai/gsm8k`;
- config: `main`;
- revision: `740312add88f781978c0658806c59bc2815b9866`;
- fixed five-shot prompt content and demonstrations;
- serialization through base-B's frozen native chat template;
- greedy decoding;
- maximum 256 generated tokens;
- the already-qualified exact-rational numeric parser;
- no task-specific adaptation before the gate.

No SVAMP fallback exists in this successor gate.

#### Disjoint cohort construction

The pinned GSM8K test split contains 1,319 examples.

Let `E` be the sorted set of the 256 dataset indices recorded in
`experiments/06_uesd/results/exp_e1_task_band.json`. Require:

- `|E| = 256`;
- comma-joined, ASCII-encoded sorted-index SHA-256:
  `0cc16e5a27c42ed8cab155006c25883a2695cac4131150b7e1d61334e912192b`.

Let `P = {0, ..., 1318} \ E`, giving `|P| = 1063`.

Use the fixed selection string:

`semantic-ratchet-base-b-gsm8k-v1-2026-08-09`

For each `i` in `P`, compute:

`SHA256(dataset_revision + "\n" + selection_string + "\n" + decimal(i))`

Sort by the raw 32-byte digest ascending, breaking any tie by integer index,
and select the first 256 indices in that order. The comma-joined,
ASCII-encoded ordered-index SHA-256 must be:

`670705ea2936f75f0e90a4048d3f5b5ec3a63b42577c0d7a9df87253b77444ff`

Before generation, assert and record:

- selected count = 256;
- selected unique count = 256;
- overlap with `E` = 0;
- remaining unconsumed official-test count = 807;
- selected-row content digest;
- dataset split fingerprint.

Neither the 256 base-A gate examples nor the 256 base-B gate examples may
be used for controller training, critic harvesting, calibration, checkpoint
selection, threshold selection, or final in-domain evaluation. The remaining
807 official-test examples are reserved for final evaluation.

#### Frozen outcome taxonomy

Every response belongs to exactly one category:

1. `correct_numeric`;
2. `valid_extracted_incorrect`;
3. `model_empty_non_answer`;
4. `parser_recognition_failure`.

A model-empty non-answer contains no complete numeric candidate in the
frozen scored segment, including an empty response, whitespace-only
response, or bare answer marker. Record it as:

- `correct = false`;
- `extracted_answer = null`;
- `extraction_failed = true`;
- `valid_extracted_incorrect = false`;
- `model_empty_non_answer = true`;
- `parser_recognition_failure = false`;
- `exact_answer_failure = true`.

The parser-recognition-failure ceiling measures harness recognition defects
only. Model-empty non-answers do not enter its numerator. They instead have
their own independently frozen suitability ceiling.

Define:

- `usable_incorrect_count =
  valid_extracted_incorrect_count + model_empty_non_answer_count`;
- `parser_recognition_failure_rate =
  parser_recognition_failure_count / 256`;
- `model_empty_non_answer_rate =
  model_empty_non_answer_count / 256`.

The four category counts must sum to 256. Report every numerator and
denominator.

#### PASS

The successor task-band gate passes only if all are true:

- correct count is 26–217 inclusive;
- correct count is at least 40;
- usable incorrect count is at least 40;
- model-empty non-answers are at most 12/256, i.e. no more than 5%;
- parser-recognition failures are at most 12/256 after at most one eligible
  parser repair governed by the preceding parser-repair-scope amendment;
- leakage, provenance, determinism, and accounting checks pass.

#### VOID

Any valid miss is terminal `VOID`, including:

- below or above the task band;
- fewer than 40 correct or usable incorrect examples;
- more than 12 model-empty non-answers;
- parser-recognition failures above 5% after the permitted repair branch;
- contamination, cohort overlap, provenance failure, or accounting failure.

There is no fallback, repeat cohort, checkpoint substitution, or
post-result threshold change under this amendment.

#### Model-empty semantics in the full program

If the base-B gate passes, the same model-empty definition applies to every
decoded candidate at every controller horizon, during state harvesting,
critic training, calibration, ablation, and final evaluation.

- Model-empty candidates receive an incorrect outcome label and may be used
  as negative critic examples.
- A correct earlier response followed by a model-empty response counts as a
  correct-to-wrong regression.
- A latch-selected model-empty state counts as an incorrect final answer.
- Report model-empty counts and rates at every horizon and for every arm.
- Report correct-to-empty and empty-to-correct transitions separately.
- No empty response may be imputed, regenerated, or removed from an accuracy
  denominator.

To prevent trivial format detection from satisfying the semantic-critic
claim, report the critic provenance metrics both:

1. over all candidates; and
2. after excluding every model-empty candidate.

The preregistered critic AUROC, advantage-over-confidence, on-policy, and
confidence-plus-schedule-matched thresholds must pass on the nonempty
candidate population. The existing matched-pair minimum must also be met
using nonempty pairs alone. Failure to supply that population is VOID;
failure of a properly powered nonempty critic comparison is KILL.

#### Transfer-set consequence

SVAMP is no longer an untouched confirmatory transfer set: 256/300 official
test examples were consumed by the base-A gate. SVAMP may be reported only
as a disclosed secondary diagnostic and may not satisfy the preregistered
cross-family CONFIRM condition.

Before full 0.5B training, a separate pre-data amendment must freeze a new
untouched cross-family transfer set and its evaluation cohort. The full
program remains blocked until that replacement is registered.

#### Mechanics-pilot independence

This section supersedes earlier language that made mechanics-pilot launch
conditional on the task-band result.

The 30M shortcut-resistant synthetic-deduction mechanics pilot is independent
of base-A, base-B, GSM8K, SVAMP, and the real-task task-band gate. It may
proceed after its own independent pre-training code review, regardless of
the base-B gate's state or verdict.

A mechanics PROCEED result does not authorize the 0.5B program by itself.
The full program requires both:

- an admissible mechanics-pilot outcome under its frozen rules; and
- a PASS from the base-B task-band gate.

A mechanics FAIL or blocking VOID still prevents the 0.5B launch even if
the base-B task-band gate passes.

### 2026-08-09 — Base-B selection compute-budget revision (pre-data)

This note is recorded after selecting the single frozen `base-B` checkpoint
and before any base-B generation, smoke evaluation, canonical evaluation, or
full-program training. It revises time estimates only. The token budgets and
the 5% trainable-overhead ceiling remain unchanged. A conservative 4.3×
multiplier is applied to frozen-base-dominated rows.

| Component | Revised estimate |
|---|---:|
| Task-band gate | 1.3–3.5h |
| Mechanics pilot | unchanged, 1.5–2.5h |
| Semantic-ratchet controller | 34–52h |
| Matched fixed-depth controller | 26–39h |
| Critic harvesting | 9–17h |
| Learned reranker | 4–9h |
| Final inference/ablations | 13–22h |
| **Full program** | **approximately 90–150 GPU-hours** |

### 2026-08-10 — Exploration-adopted selector and interface amendments (pre-pilot-data)

This amendment is recorded before any mechanics-pilot data or execution. It
extends the pilot with an informational selector and mandatory diagnostics,
registers a separate follow-up, and replaces the readout interface for the
future 0.5B program only. It does not change the existing E1 gates or
reinterpret any E1 evidence.

#### Arm 4 — hysteretic incumbent-replacement latch (informational only)

Arm 4 operates over the exact shared frozen recurrent-state bank used by the
other three arms. Let \(q_{i,t}\in[0,1]\) be the frozen post-sigmoid critic
score for example \(i\) at horizon \(t\). For model seed \(s\), initialize

\[
j_{i,1}=1,
\]

and, for \(t\ge2\), apply the sequential rule

\[
j_{i,t}=
\begin{cases}
t,&q_{i,t}-q_{i,j_{i,t-1}}>\delta_s,\\
j_{i,t-1},&\text{otherwise}.
\end{cases}
\]

The arm-4 prediction at budget \(B\) is the prediction from state
\(j_{i,B}\). The strict inequality retains the incumbent on a tie. The rule
compares every challenger with the current incumbent, not with the previous
horizon and not with a separately generated state.

Select one \(\delta_s\) independently for each trained critic/model seed from
the frozen grid

\[
\mathcal D=\{0.00,0.02,0.05,0.10\},
\]

using only that seed's selector-calibration trajectories through \(T=16\).
Freeze the shared \(t^*\) first under the existing rule. For every candidate
\(\delta\), simulate the complete sequential latch and record at \(B=16\):

- gain retention relative to the no-latch \(T=1\) accuracy and the frozen
  no-latch \(t^*\) gain;
- accuracy;
- examples with at least one accepted harmful challenge divided by all
  calibration examples;
- accepted harmful challenges divided by offered challenges whose incumbent
  was correct and challenger was incorrect;
- rejected beneficial challenges divided by offered challenges whose
  incumbent was incorrect and challenger was correct;
- total replacements divided by all offered challenges.

A candidate is feasible only when its calibration gain retention is at least
90%. Among feasible candidates select lexicographically: (1) the smallest
harmful-switch-example rate; (2) the highest \(B=16\) accuracy; and (3) the
larger \(\delta\). If none is feasible, freeze \(\delta_s=0\) and record
`calibration_constraint_miss=true`. Only the selected \(\delta_s\) is applied
to test data; test results for unselected grid values are not reported. The
selected constant is frozen without refitting, recalibration, or retuning
through \(T=32\).

Arm 4 is informational only. It cannot rescue an arm-3 FAIL, veto an arm-3
PROCEED, enter the regression-only-VOID exception, or create a best-selector
clause. Its values are excluded from every PROCEED/FAIL/VOID computation.
Interpret it only as follows:

- improved late-horizon accuracy with fewer harmful switches and retained
  gain indicates that repeated low-margin replacement was a failure
  mechanism and favors hysteresis for the 0.5B design;
- no improvement indicates high-margin critic mistakes, biased scores,
  schedule drift, or insufficient semantic information rather than a
  low-margin switching problem;
- preserved correctness with lost gain indicates a real
  plasticity-stability tradeoff rather than merely an extreme-value bug.

#### Trajectory diagnostics

For every budget \(B=1,\ldots,32\), every seed \(s\), every selector arm
\(a\) (including arm 4), and the pooled records formed by concatenating
examples across seeds, report

\[
O_s(B)=
\frac{\#\{i:\exists t\le B,\ \hat y_{i,s,t}=y_i\}}{N_s},
\]

\[
F_{a,s}(B)=
\frac{\#\{i:\hat y^a_{i,s,B}=y_i\}}{N_s},
\qquad
H_{a,s}(B)=O_s(B)-F_{a,s}(B).
\]

The oracle term is shared because all arms select from the same state bank.
Every reported value includes its numerator and denominator. For the raw
state dynamics, report at every transition \(t\to t+1\):

\[
h_{C\to W,s}(t)=
\frac{\#\{i:\hat y_{i,s,t}=y_i,\ \hat y_{i,s,t+1}\ne y_i\}}
     {\#\{i:\hat y_{i,s,t}=y_i\}},
\]

\[
h_{W\to C,s}(t)=
\frac{\#\{i:\hat y_{i,s,t}\ne y_i,\ \hat y_{i,s,t+1}=y_i\}}
     {\#\{i:\hat y_{i,s,t}\ne y_i\}}.
\]

For every selector arm, report transition-by-transition and aggregate switch
hazards with numerators and denominators: accepted harmful challenges,
rejected beneficial challenges, correct-incumbent survival, and total
replacements. Wherever model-empty outputs are possible, also report raw and
selected correct-to-empty and empty-to-correct transitions. A challenge is
every state offered after \(t=1\); acceptance means it replaces the incumbent.

The evaluator must assert the accounting identities

\[
F_a(B)\le O(B),\qquad H_a(B)\ge0,
\]

for all arms, budgets, seeds, and the pooled records, and at an adjudicated
endpoint \(H\),

\[
H_{\mathrm{no}}(H)\ge A_{\mathrm{no}}(t^*)R_{\mathrm{no}}(H).
\]

Violation is an evaluator bug and produces `VOID` for integrity; it is not a
scientific floor. No minimum-headroom validity floor exists. These diagnostics
may not select \(t^*\), \(\delta\), an endpoint, a subset, generator settings,
or an outcome.

#### E2-CERT follow-up registration

The following is a registered follow-up, not a fifth E2 pilot arm:

> **E2-CERT — verified-evidence latch follow-up.** Using the same frozen generator family, skeleton-disjoint splits, model seeds, parameter envelope, training-token budget, and \(T=1,\ldots,32\) grid as E2, train a separately budgeted joint answer-and-certificate model that emits a canonical sequence of rule identifiers and grounded substitutions at every state. A deterministic verifier, blind to the gold answer, accepts a certificate only when every premise appears in the prompt or a prior verified step, every rule application type-checks, and the final derived proposition exactly matches the state’s emitted answer choice. The verified latch permanently banks the first accepted state; if no state verifies by budget \(B\), it falls back to the calibration-frozen asymmetric critic latch. A capacity- and token-matched control receives the same certificate supervision but ignores verifier acceptance during selection. At \(H=16\) and \(H=32\), report verified coverage, verifier false-accept count, overall accuracy and regression, oracle headroom, and the fraction of residual asymmetric-latch headroom closed by verification. Support requires zero verifier false accepts, at least 50% verified coverage per seed, and closure of at least 50% of the asymmetric latch’s positive residual headroom pooled and separately in both seeds. E2-CERT cannot alter or reinterpret the E2 outcome.

E2-CERT cannot change, delay, rescue, or reinterpret E2 and requires its own
implementation, budget matching, review, and launch decision.

#### 0.5B typed terminal interface

For the 0.5B program only, replace the free-text-parser readout with a shared
two-channel output:

```text
rationale: <free text>
decision:
  kind: answer | abstain
  value: <canonical numeric value, only when kind=answer>
  certificate: <nullable; inert in the main experiment>
```

The sequence reranker receives the prompt, full free-text rationale, and typed
terminal record for every candidate. The primary scorer reads only the typed
terminal decision. `abstain` is incorrect in primary exact-answer accuracy;
coverage and selective risk conditional on answering are secondary metrics.
Failure to emit a terminal record before the token limit remains a distinct
model-empty outcome and is incorrect; it is not converted into abstention.
The nullable certificate field is inert and cannot affect selection,
adjudication, or the main experiment's result.

Rationale generation remains unconstrained. Grammar masking begins only after
an explicit terminal-decision marker and uses the smallest terminal grammar
that expresses `answer` with a canonical numeric value or `abstain`. Every arm
uses the same terminal grammar, marker, stopping rules, token limit, numeric
normalization, and accounting. Grammar-mask computation and terminal tokens
enter the matched layer-token FLOP budget for every arm, including every
best-of-\(N\) candidate. Confidence features are computed from the same
constrained terminal distribution in every applicable arm.

After freezing the interface, run a paired masked-versus-unmasked diagnostic
on calibration data only with matched random seeds. Report answer changes,
model-empty changes, and likelihood changes where available. This diagnostic
characterizes grammar-mask distortion but cannot choose the interface, change
the primary protocol, or select any outcome. Conclusions are scoped to the
common typed-decoding regime.

This interface amendment applies only to the future 0.5B program. It does not
alter the E1 task-band gates, their free-text parser, or any existing E1
evidence or verdict.
