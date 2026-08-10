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
