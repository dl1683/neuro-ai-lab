# Frozen Best-of-N Safe Selection Pilot Preregistration

Status: **PREREGISTERED / NOT YET RUN / LAUNCH BLOCKED ON HASH SLOTS, MANIFEST IDENTITIES, RUNNER BUILD, AND INDEPENDENT REVIEW**

This is the single canonical preregistration for the frozen best-of-\(N\) safe-selection pilot. It tests an inference-layer selection law over independent stochastic samples. It is not a UESD-lineage experiment, does not reopen the D38–D40 fixed-point arc, and cannot alter, rescue, delay, reinterpret, or adjudicate E2 or the preregistered 0.5B semantic-ratchet experiment. No result is claimed here.

## Hypothesis and inference-layer law

### Hypothesis

For a frozen generator–verifier pair, sequential incumbent-based selection with a calibration-frozen, multiplicity-aware replacement margin can use a larger independent candidate bank while reducing verifier-induced harmful replacement, retaining at least 90% of the beneficial acquisitions supplied by ordinary verifier argmax, and improving selected exact-answer accuracy.

### Law being tested: safe acquisition under candidate expansion

For problem \(i\) and candidate position \(n\), let:

- \(c_{i,n}\in\{0,1\}\) be exact-answer correctness;
- \(q_{i,n}\in[0,1]\) be the frozen verifier score;
- \(j^a_{i,n}\) be the incumbent retained by policy \(a\) after observing candidates \(1,\ldots,n\).

An offered challenge is:

- **harmful** when \(c_{i,j^a_{i,n-1}}=1\) and \(c_{i,n}=0\);
- **beneficial** when \(c_{i,j^a_{i,n-1}}=0\) and \(c_{i,n}=1\).

A harmful or beneficial switch occurs only if the corresponding challenge replaces the incumbent.

The tested inference-layer law is:

> Increasing candidate supply should not increase the probability of replacing a correct incumbent with a verifier-induced false positive when replacement requires multiplicity-aware evidence, provided the selector retains at least 90% of the beneficial acquisitions available to verifier argmax.

This is an empirical law for the frozen generator–verifier–task system, not a theorem and not a claim that one numerical margin transfers unchanged to arbitrary models or verifiers. The proposed transferable object is the protocol class:

1. preserve an incumbent;
2. require calibration-frozen evidence for replacement;
3. expose harmful and beneficial switches with explicit denominators;
4. increase the margin with candidate multiplicity or stop the budget;
5. retain acquisition rather than achieving apparent safety by refusing nearly all changes.

## Target headline, conditional on CONFIRM

**“A simple incumbent rule beat verifier argmax and self-consistency while
cutting harmful answer switches on a frozen math cohort.”**

## Claim-language contract

All result writeups, status summaries, abstracts, captions, and public-facing
descriptions are governed by this section.

CONFIRM does not license “never gets worse,” “cannot abandon a correct answer,”
“eliminates harmful replacement,” “virtually never,” “monotonic,” or
“anytime-safe.” The registered conditions permit nonzero harmful switching,
up to 10% relative loss of beneficial acquisitions versus verifier argmax,
and do not require accuracy to be nondecreasing at every prefix.

A successful writeup may state only the measured facts, including:

- the exact \(N=16\) accuracy differences against every adjudicating baseline;
- that at least 90% of verifier argmax's observed beneficial acquisitions were
  retained, with numerator and denominator;
- the observed relative and absolute reduction in harmful-switch examples,
  with numerator, denominator, and paired interval;
- the observed first-correct-to-final-wrong regression rate, if reported with
  its exact event definition, numerator, denominator, and interval.

Ratio language must name both rates. Prefer “the observed regression rate was
\(r\) versus \(r_0\), a relative reduction of \(x\%\)” over “\(k\) times less
likely.” If the numerator is zero, report “0/\(D\) observed” and a one-sided
confidence bound; do not report an infinite reduction or use “never.”

Every claim must say “on this frozen generator–verifier pair and frozen GSM8K
train-split cohort.” It may not say “held-out GSM8K,” “official-test
performance,” or imply generalization to unseen GSM8K problems.

### “Isn’t that obvious?” defense

It is known that maximizing an imperfect verifier over more candidates can produce overestimation and degradation. It is not established that a frozen, no-training incumbent rule can reverse that failure over independent language-model samples while retaining at least 90% of the useful acquisitions and adding at least three test points over verifier argmax.

### “Isn’t that trivial?” defense

The result cannot be obtained by always preserving the first sample or merely hiding later candidates. CONFIRM requires positive accuracy gain over argmax, retained beneficial acquisition, actual incumbent replacement, reduced optimism and harmful switching, a positive paired confidence interval, and survival across 1,000 candidate-order permutations.

## Prior art and differentiation

Dorner et al., [“ROC-n-reroll: How verifier imperfection affects test-time scaling” (arXiv:2507.12399)](https://arxiv.org/abs/2507.12399), characterize Best-of-\(N\) and rejection-sampling accuracy through verifier ROC geometry, show that rejection sampling can outperform Best-of-\(N\) at fixed compute, and show that low-compute observations generally do not determine high-compute behavior. That work characterizes degradation; it does not construct the incumbent-based replacement protocol tested here.

Dalal et al., [“More Test-Time Compute Can Hurt: Overestimation Bias in LLM Beam Search” (arXiv:2603.15377)](https://arxiv.org/abs/2603.15377), derive and demonstrate scorer overestimation bias as beam width grows. Their object is branching beam search and a scorer-dependent useful-width diagnosis, not calibration-frozen incumbent hysteresis over an independently sampled stream.

Yu, Li, and Wang, [“Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning” (arXiv:2502.00271)](https://arxiv.org/abs/2502.00271), demonstrate across models, tasks, and outcome/process verifiers that widening verifier-guided search can increasingly misrank or prune valid paths and eventually lose to repeated sampling. This is large-scale evidence for an adjacent pathology, but its candidates arise from branching search and it does not construct incumbent-based anytime safety.

Huang et al., [“Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment” (ICML 2025)](https://proceedings.mlr.press/v267/huang25c.html), are the closest conceptual control: they analyze reward hacking under Best-of-\(N\) and introduce a pessimistic inference-time algorithm with scaling-monotonic guarantees. Therefore this pilot must not claim to be the first monotonic inference-time selection method. Its narrower differentiation is a simple incumbent-based, acquisition-constrained protocol tested without weight training and with the same harmful/beneficial switch ledger across independent sampling and, separately, a correlated recurrent trajectory.

The three papers named in the ruling characterize or document the failure but do not construct this specific incumbent-based anytime-safety protocol. Success here would support transfer of a failure decomposition and policy class, not universality across arbitrary generators, verifiers, or tasks.

Wang et al., “Self-Consistency Improves Chain of Thought Reasoning in Language
Models,” establish answer-level aggregation over independently sampled
reasoning paths as a strong training-free baseline, including substantial
reported GSM8K gains. Verification-weighted answer voting is also established
prior art. This pilot therefore does not claim novelty from combining
candidate multiplicity with verification.

ROC-n-reroll does not determine when answer voting beats verifier selection.
Its verifier analysis conditions on binary candidate correctness and score
geometry; answer plurality additionally depends on the complete distribution
of normalized answer identities, especially the collision structure of wrong
answers. A verifier AUC and a scalar answer-agreement rate are insufficient:
AUC discards the ROC-tail geometry governing extreme selection, while a scalar
agreement rate discards which answers agree and whether repeated wrong answers
share a systematic mode.

The contribution tested here is consequently narrower: whether a
calibration-frozen sequential incumbent rule improves over verifier argmax,
unweighted answer aggregation, and verifier-weighted answer aggregation on
one frozen candidate distribution while preserving explicitly measured
beneficial acquisition.

## Frozen components and OpSec boundary

Tracked files identify the models only as:

- `base-C`: one frozen small public math-instruct generator;
- `verifier-V`: one frozen public process-reward verifier.

Their exact repository identifiers, revisions, tokenizer revisions, resolved local paths, file digests, licenses, and local content digests must be recorded before any generation in the gitignored manifest:

`experiments/07_safe_selection/_local_manifest.md`

Neither checkpoint may be chosen by comparing newly generated GSM8K calibration or test results across alternatives. Only one generator and one verifier are permitted. Substitution requires a pre-data steering decision and a tracked amendment.

The frozen class is:

- a small public math-instruct causal generator;
- a public process-reward verifier intended for mathematical reasoning and Best-of-\(N\) scoring;
- both BF16-compatible on one 24 GB GPU, either co-resident if preflight memory permits or through staged generation and scoring;
- no weight training, adaptation, LoRA, critic fitting, or learned calibration.

The verifier’s frozen documented interface inserts a step marker after each
reasoning step and produces a positive-class probability for every step. Its
exact documentation URL is recorded only in the gitignored local manifest.

## Dataset and cohort construction

Use:

- dataset: `openai/gsm8k`;
- config: `main`;
- split: official `train`;
- revision: `740312add88f781978c0658806c59bc2815b9866`;
- expected split size: 7,473;
- expected train-split fingerprint: `c6f812ae33c9159d`.

The official GSM8K test split is not used by this pilot. In particular, the
256 base-A rows, the 256 base-B rows, and all 807 rows reserved for final
evaluation by `experiments/06_uesd/PREREGISTRATION.md` remain untouched.

### Why use of the train split is admissible

“Evaluated on train” normally identifies leakage when examples used to fit a
supervised system are reused to estimate that fitted system’s generalization.
That is not the estimand here. This pilot performs no weight training,
adaptation, checkpoint comparison, or test-label-based policy fitting. It
compares six frozen selection policies on an identical bank generated by one
frozen generator and scored by one frozen verifier. Only the 256-problem
calibration cohort may set policy thresholds; the 512-problem pilot-test
cohort is inaccessible until those thresholds and the calibration artifact
digest are frozen.

Accordingly, train/test provenance does not compromise the internal paired
policy comparison. It does limit external interpretation. The result must be
described as applying to this frozen generator–verifier pair on this
revision-pinned GSM8K train-split cohort. It is not an official-test result and
does not establish performance on unseen GSM8K problems.

The generator or verifier may have encountered GSM8K training material before
this pilot. That possibility is shared across all policies and therefore does
not bias their paired comparison, but it may change task difficulty, ceiling,
or selection headroom. First-candidate accuracy, oracle coverage, response
diversity, and oracle-to-selected gap must be reported, but none may be
described as a direct test or measurement of memorization.

### Frozen prior-consumption registry

A consumed row is identified by the split-qualified key:

```text
(dataset_id, config, revision, split, dataset_index)
```

Numerically equal indices in different datasets or splits are not the same
row. Before constructing either cohort, bind the following complete
prior-consumption registry as of 2026-08-10:

| Source | Split and use | Count | Frozen binding |
|---|---|---:|---|
| `openai/gsm8k` | train demonstrations `[0,1,2,3,4]` | 5 | index SHA-256 `6484c68c0c85987f9beb3db42175c46955a8abe05170239580fcd1ff8b514452`; row-content SHA-256 `b310a622f3e0ef1ed3694cf37cc8da38362e48f194d68adc6df34155e702025e` |
| `openai/gsm8k` | base-A official-test cohort | 256 | sorted-index SHA-256 `0cc16e5a27c42ed8cab155006c25883a2695cac4131150b7e1d61334e912192b`; row-content SHA-256 `1034771b7a0d0b547819e185ebddd7a981d6c000a81aefa1c956341df7c98674` |
| `openai/gsm8k` | base-B official-test cohort | 256 | ordered-index SHA-256 `670705ea2936f75f0e90a4048d3f5b5ec3a63b42577c0d7a9df87253b77444ff`; row-content SHA-256 `37febcb5eab403fa8f479a3adc76124ddeb78183e5305f944810eb64c25cea86` |
| `ChilleD/SVAMP` | train demonstrations `[0,1,2,3,4]` | 5 | index SHA-256 `6484c68c0c85987f9beb3db42175c46955a8abe05170239580fcd1ff8b514452`; row-content SHA-256 `f7fe5da1bf49cbb4125949999f89d1ca83773e3c7a3cbb8cc203c0ccc8c686dd` |
| `ChilleD/SVAMP` | base-A fallback official-test cohort | 256 | ordered-index SHA-256 `69af24379818a75c67c3444639acd0e2d7d17091175a6720bb111b73357dd684`; row-content SHA-256 `f674ce1024e890731192f5be4c55655d4c73abc83b395bfd0f3c216b25bbfba0` |

Normalize questions using:

```python
" ".join(question.casefold().split())
```

For SVAMP, `question` is `Body` and `Question`, stripped and joined by one
space. Deduplicate the normalized prior-consumed questions, sort them
lexicographically, join them with LF and no terminal LF, and UTF-8 encode.
Require:

- unique normalized prior-consumed question count: 778;
- normalized prior-consumed-question SHA-256:
  `c4a3a04eeb926bd89474c1938ea816fa681e66f8cf03a5bdc3e1411b9e77bc62`.

The target-split index exclusion set is therefore:

```text
X_train = {0,1,2,3,4}
```

Both selected cohorts must additionally have zero normalized-question overlap
with the complete cross-dataset consumption registry above.

### Calibration cohort

Define:

```text
P_cal = {0,...,7472} \ X_train
```

Require:

- count: 7,468;
- comma-joined ASCII sorted-index SHA-256:
  `6f3b8138db046b3f0c78dd13896864bea624c063ad48706491adebdd0f8b51c4`.

Use the fixed calibration selection string:

```text
bon-safe-selection-base-c-gsm8k-train-calibration-v1-2026-08-10
```

For each `i` in `P_cal`, compute:

```text
SHA256(
  dataset_revision + "\n" +
  calibration_selection_string + "\n" +
  decimal(i)
)
```

Sort by raw 32-byte digest ascending, breaking any tie by integer dataset
index. The first 256 indices form the calibration cohort.

Require:

- calibration count and unique count: 256;
- ordered-index SHA-256:
  `9236163527dac17431ac43c9d094874ce644a211219a77de6783917ba15705c7`;
- sorted-index SHA-256:
  `c424773f8b1246e1030dac9b3b92fd40f6f5f27e49186e03a54e0e58cc356046`;
- ordered row-content SHA-256:
  `3919f0dc142cc9c61e70da902fb1233c49c07a07ce542e3e3f62e46263ec0376`.

### Test cohort

Let `C` be the frozen calibration-index set and define:

```text
P_test = P_cal \ C
```

Require:

- count: 7,212;
- comma-joined ASCII sorted-index SHA-256:
  `37b837cd3ddb055ebf59ac83fc0c024a21f9aaae1ebce409545e01268707fc91`.

Use the fixed test selection string:

```text
bon-safe-selection-base-c-gsm8k-train-test-v1-2026-08-10
```

For each `i` in `P_test`, use the same digest-ranking construction with the
test selection string. The first 512 indices form the pilot-test cohort.

Require:

- test count and unique count: 512;
- calibration/test index overlap: 0;
- ordered-index SHA-256:
  `f151f1c46c2421ed78aef43f1cd6e7bd99fc7c8ab5bb945e36adff346bad9c48`;
- sorted-index SHA-256:
  `8aa38ad17399eb30526c05174408322b53a05a874ef74805d5b73d18a787b6bf`;
- ordered row-content SHA-256:
  `0efd8a2b9c8f7341f3e2677f8da89dd76ad65ae29cab05ae33751b1810f1afac`.

The comma-joined calibration-then-test ordered-index SHA-256 must be:

`727debb0c35837935e4686ce6ba30051f5826bca8eef2963b7acbae47953fa2f`.

The 6,700 train rows allocated to neither cohort remain unused by this pilot.
Their sorted-index SHA-256 must be:

`aa28e47af683f7fe65dd2cc78eaf29030c55d7de9127fe4995af844f1d31b44c`.

Their sorted row-content SHA-256 must be:

`0c78ada0a2b578c25e4361fc592bcf4a956a401fcb0921d5769b02e578f42e98`.

### Content and exclusion assertions

Row-content hashes use UTF-8 SHA-256 over compact canonical JSON containing,
in cohort order:

```json
[
  {"answer": "<exact dataset value>", "index": 123, "question": "<exact dataset value>"}
]
```

Serialization uses `ensure_ascii=false`, lexicographically sorted object keys,
and separators `(",", ":")`.

Before any model is loaded, assert and record:

- revision, split size, and split fingerprint match;
- all supplied pool and cohort hashes match;
- calibration and test indices are unique and disjoint;
- neither cohort contains train indices `[0,1,2,3,4]`;
- neither cohort has normalized-question overlap with any of the 778
  prior-consumed questions;
- calibration contains 256 unique normalized questions;
- test contains 512 unique normalized questions;
- calibration/test normalized-question overlap is zero;
- the five demonstrations retain row-content SHA-256
  `b310a622f3e0ef1ed3694cf37cc8da38362e48f194d68adc6df34155e702025e`;
- all row-content hashes match.

Failure before generation blocks launch and requires a pre-data correction.
Failure after retained generation begins is `VOID`.

### Precommitted provenance hash slots

The canonical preregistration must contain the following named slots:

| Slot | Frozen value before generation |
|---|---|
| `dataset_train_split_fingerprint` | `c6f812ae33c9159d` |
| `normalized_prior_consumed_questions_sha256` | `c4a3a04eeb926bd89474c1938ea816fa681e66f8cf03a5bdc3e1411b9e77bc62` |
| `demonstrations_content_sha256` | `b310a622f3e0ef1ed3694cf37cc8da38362e48f194d68adc6df34155e702025e` |
| `calibration_selected_row_content_sha256` | `3919f0dc142cc9c61e70da902fb1233c49c07a07ce542e3e3f62e46263ec0376` |
| `test_selected_row_content_sha256` | `0efd8a2b9c8f7341f3e2677f8da89dd76ad65ae29cab05ae33751b1810f1afac` |
| `remaining_unallocated_row_content_sha256` | `0c78ada0a2b578c25e4361fc592bcf4a956a401fcb0921d5769b02e578f42e98` |
| `prompt_serialization_sha256` | `UNFILLED — launch blocking` |
| `parser_source_sha256` | `UNFILLED — launch blocking` |
| `runner_source_sha256` | `UNFILLED — launch blocking` |
| `local_manifest_identity_digest` | `UNFILLED — launch blocking and never expose private identifiers` |

The CPU-only cohort pass may fill the unfilled slots but may not load either
model or generate a response. Every slot must be populated and independently
reviewed before the retained calibration smoke.

Once this registration lands, the 768 selected rows are reserved to line 07
until the pilot is completed, voided, killed, or explicitly abandoned through
a pre-data amendment. No intervening checkpoint selection, threshold
selection, calibration, or evaluation may consume them.

## Frozen prompt, generation, and bank protocol

Reuse the qualified E1 GSM8K evaluation contract:

- fixed five-shot demonstrations from revision-pinned training indices `[0,1,2,3,4]`;
- the same system preamble requiring step-by-step reasoning and a final `#### <numeric answer>` line;
- serialization through `base-C`’s frozen native chat template;
- the already-qualified exact-rational numeric parser;
- maximum 256 generated tokens;
- the same response segmentation and stopping semantics;
- model-empty and parser-recognition failures are incorrect and remain in every denominator.

Freeze stochastic decoding as:

```text
do_sample = true
temperature = 0.7
top_p = 0.8
top_k = 0
repetition_penalty = 1.0
num_return_sequences = 1 per seeded call
dtype = bfloat16
batch_size = 1
```

No prompt, demonstration, parser, stopping rule, temperature, sampling parameter, token cap, batch size, or generation library version may change after calibration generation begins.

Generate exactly 16 candidates per problem. Candidate ordinals are \(1,\ldots,16\).

Use the fixed seed string:

`bon-safe-selection-base-c-generation-seeds-v1-2026-08-10`

For dataset index \(i\) and candidate ordinal \(r\), define:

```text
payload = dataset_revision + "\n" +
          seed_string + "\n" +
          decimal(i) + "\n" +
          decimal(r)

seed = uint64_big_endian(SHA256(payload)[0:8]) mod (2^63 - 1)
```

The ordered schedule contains 12,288 seeds and has SHA-256:

`8d3d567ba90da7ec7cdc0eb9c5764b545834975c0d477f0fc030888c9f2f03fd`

Generation order is frozen:

1. calibration problems in canonical cohort order;
2. within each problem, candidate ordinals 1 through 16;
3. freeze policy thresholds from calibration;
4. only then generate test problems in canonical cohort order;
5. within each test problem, candidate ordinals 1 through 16.

All six policies use the identical generated and scored bank. Test generation may not start before the selected calibration parameters and calibration artifact digest are frozen.

A repeat-determinism check reruns all 16 seeds for calibration cohort positions 0 and 1. Responses, stopping reasons, extracted answers, and verifier-input serialization must match exactly. The repeated outputs are diagnostic duplicates and do not enter any metric.

## Frozen verifier scoring

For every candidate:

1. normalize line endings to LF;
2. split reasoning steps on one or more blank lines;
3. remove empty steps;
4. if nonempty text has no blank-line boundary, treat it as one step;
5. insert the verifier’s required step marker after every step;
6. compute each step’s positive-class probability in BF16;
7. define candidate score as the minimum step probability:

\[
q_{i,n}=\min_k q_{i,n,k}.
\]

An empty response receives \(q_{i,n}=0\). A nonempty parser-recognition failure is scored normally but remains incorrect for exact-answer evaluation. NaN, infinite, missing, or out-of-range verifier scores are integrity failures.

The verifier uses batch size 1 and deterministic evaluation mode. No score transformation, temperature calibration, isotonic regression, candidate-length correction, or outcome-model mixing is permitted.

## Evaluation prefixes

Evaluate every policy at:

\[
N\in\{1,2,4,8,16\}.
\]

At each prefix, only candidates \(1,\ldots,N\) are available.

A strict inequality is used for every replacement. Score ties retain the incumbent. Candidate order is never sorted by verifier score before a sequential policy is applied.

## Six frozen selection policies

### Policy 0 — verifier argmax

Initialize \(j_{i,1}=1\). For \(n\ge2\):

\[
j_{i,n}=
\begin{cases}
n,&q_{i,n}>q_{i,j_{i,n-1}},\\
j_{i,n-1},&\text{otherwise}.
\end{cases}
\]

This equals argmax over the observed prefix with earliest-position tie-breaking.

### Policy 1 — incumbent plus fixed margin

For one calibration-frozen \(\delta\):

\[
j_{i,n}=
\begin{cases}
n,&q_{i,n}-q_{i,j_{i,n-1}}>\delta,\\
j_{i,n-1},&\text{otherwise}.
\end{cases}
\]

Freeze \(\delta\) from:

\[
\mathcal D=
\{0.00,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30\}.
\]

### Policy 2 — incumbent plus multiplicity-aware margin

This is the primary adjudicating safe selector.

For calibration-frozen \((\delta_0,\lambda)\), define:

\[
m(n)=\delta_0+\lambda\sqrt{2\log n},
\]

and apply:

\[
j_{i,n}=
\begin{cases}
n,&q_{i,n}-q_{i,j_{i,n-1}}>m(n),\\
j_{i,n-1},&\text{otherwise}.
\end{cases}
\]

Freeze:

\[
\delta_0\in\mathcal D
\]

and

\[
\lambda\in
\{0.000,0.005,0.010,0.020,0.030,0.050,0.075,0.100\}.
\]

The \(\lambda=0\) case is retained as a registered degeneracy check. If selected, report `multiplicity_term_collapsed=true`; test results may still be computed, but multiplicity-specific language is prohibited.

### Policy 3 — calibration-frozen budget stopping

For each candidate budget:

\[
B\in\{1,2,4,8,16\},
\]

run Policy 0 through \(B\), then permanently retain its incumbent. At requested prefix \(N\), the output is:

\[
j^{\text{stop}}_{i,N}=j^{\text{argmax}}_{i,\min(N,B^*)},
\]

where \(B^*\) is selected on calibration and frozen before test generation.

This is a global stopping policy, not an adaptive per-problem rule. Its actual samples consumed at \(N=16\) are exactly \(B^*\).

### Policy 4 — verifier-blind answer plurality
(fifth policy; self-consistency baseline)

For every valid extracted numeric candidate, let \(a_{i,n}\) be its canonical
exact-rational answer under the frozen parser and numeric-normalization rules.
Model-empty responses and parser-recognition failures have no answer identity,
cast no vote, remain incorrect, and remain in every accuracy denominator.

At prefix \(N\), define

\[
V_{i,N}(a)=
\sum_{n=1}^{N}\mathbf 1[a_{i,n}=a]
\]

over valid extracted answers only.

The incumbent answer is the answer with largest \(V_{i,N}(a)\). Ties retain
the previous-prefix incumbent when it is among the tied maximizers. If no
previous valid incumbent exists, choose the tied answer whose first valid
occurrence has the smallest candidate ordinal. Select the earliest candidate
having the incumbent answer.

If no candidate in the prefix has a valid extracted answer, select candidate
1. This output is incorrect and is not converted into an abstention or a
synthetic answer.

Consequently, when every valid answer appears exactly once, the policy retains
the earliest valid answer. A repeated answer displaces the incumbent only
after obtaining a strictly larger vote count; reaching a tie is insufficient.

This policy is verifier-blind. It uses no \(q_{i,n}\) value for answer
selection.

### Policy 5 — verifier-weighted answer plurality
(sixth policy; verifier–agreement hybrid)

Use the same canonical answer identities and invalid-output treatment as
Policy 4. For each valid extracted answer define

\[
W_{i,N}(a)=
\sum_{n=1}^{N}
q_{i,n}\mathbf 1[a_{i,n}=a].
\]

Select the answer with largest \(W_{i,N}(a)\). Ties retain the previous-prefix
incumbent when possible; otherwise choose the tied answer with the smallest
first-occurrence ordinal. Within the winning answer group, select the
highest-\(q\) candidate, breaking score ties toward the smaller ordinal.

If no candidate in the prefix has a valid extracted answer, use the Policy-0
selection for that prefix. The result remains incorrect.

No temperature, exponent, count bonus, score transformation, or fitted mixing
weight is permitted. Policy 5 is a frozen baseline, not an additional
calibrated policy.

## Calibration-only parameter selection

Let Policy 0 at \(N=16\) be the acquisition reference.

For policy \(a\), define beneficial acquisition over the first sample:

\[
BA_a=
\#\{i:c_{i,1}=0,\ c_{i,j^a_{i,16}}=1\}.
\]

Define:

\[
\operatorname{retention}_a=
\frac{BA_a}{BA_{\mathrm{argmax}}}.
\]

Every numerator and denominator must be reported. If \(BA_{\mathrm{argmax}}=0\), the pilot receives KILL reason `NO_BENEFICIAL_ACQUISITION_HEADROOM`; retention may not be reported as vacuous 100%.

A parameter setting is feasible only if:

\[
\operatorname{retention}_a\ge0.90.
\]

For each setting, also compute:

- calibration accuracy at \(N=16\);
- examples with at least one accepted harmful challenge divided by 256;
- accepted harmful challenges divided by offered harmful challenges;
- rejected beneficial challenges divided by offered beneficial challenges;
- total replacements divided by all \(15\times256\) challenges;
- selected-state optimism.

Select parameters lexicographically using calibration only.

For Policy 1:

1. smallest harmful-switch-example rate;
2. highest \(N=16\) accuracy;
3. larger \(\delta\).

For Policy 2:

1. smallest harmful-switch-example rate;
2. highest \(N=16\) accuracy;
3. larger terminal margin \(m(16)\);
4. larger \(\lambda\);
5. larger \(\delta_0\).

For Policy 3:

1. smallest harmful-switch-example rate;
2. highest accuracy at its stopped output;
3. smaller \(B\).

Policy 2 is the sole primary selector. Test performance may not select among Policies 1–3, form an oracle “best safe policy,” or change the adjudicating policy.

## Frozen metrics

At every prefix and for every policy, report:

- selected exact-answer accuracy;
- oracle coverage;
- oracle-to-selected gap;
- selected index distribution;
- selected-state verifier optimism;
- harmful and beneficial switch numerators and denominators;
- rejected beneficial challenges;
- correct-incumbent survival;
- regression from correct first sample to incorrect selected output;
- acquisition from incorrect first sample to correct selected output;
- total replacements;
- model-empty and parser-recognition-failure counts;
- correct-to-empty and empty-to-correct switches;
- actual samples consumed;
- generated and verifier-scored tokens;
- wall time, GPU power, and peak allocated/reserved VRAM.

For Policies 4 and 5, also report at every prefix:

- the number of distinct valid normalized answers;
- the winning answer's vote count and vote share;
- the largest wrong-answer vote count and vote share;
- the fraction of problems with no repeated valid answer;
- the fraction decided by the frozen tie rule;
- selected-answer correct-to-wrong and wrong-to-correct transitions.

Candidate-level offered-challenge metrics remain defined by the newly offered
candidate and the prior selected answer. When a new candidate changes the
winning answer group, that event is an accepted replacement even if the
reported representative response for the winning group occurred earlier.

Define oracle coverage:

\[
O(N)=
\frac{\#\{i:\exists n\le N,\ c_{i,n}=1\}}{512}.
\]

Define selected accuracy:

\[
F_a(N)=
\frac{\#\{i:c_{i,j^a_{i,N}}=1\}}{512}.
\]

Require:

\[
F_a(N)\le O(N)
\]

for every policy and prefix.

Define selected-state optimism:

\[
\operatorname{Opt}_a(N)=
\frac1{512}
\sum_i
\left(q_{i,j^a_{i,N}}-c_{i,j^a_{i,N}}\right).
\]

Because this is a calibration residual rather than a causal estimate, claims must use “verifier optimism” and not “verifier error caused the answer.”

## Statistical protocol

Use 10,000 paired bootstrap replicates over the 512 test problems. Each replicate resamples problem identities with replacement and preserves all within-problem candidates and policy outputs.

Use seed string:

`bon-safe-selection-paired-bootstrap-v1-2026-08-10`

For bootstrap replicate \(b\) and draw \(k\), derive the sampled problem position from the first eight SHA-256 bytes of:

```text
bootstrap_string + "\n" + decimal(b) + "\n" + decimal(k)
```

reduced modulo 512.

Report percentile 95% confidence intervals. For every adjudicating baseline
\(b\in\{\mathrm{Policy\,0},\mathrm{Policy\,4},\mathrm{Policy\,5}\}\), a
“positive paired-bootstrap CI” means the 2.5th percentile of the
Policy-2-minus-baseline-\(b\) accuracy difference is strictly greater than
zero.

For reduced optimism, report Policy-0 optimism minus Policy-2 optimism. For reduced harmful switching, report the paired difference in per-problem harmful-switch indicators.

## Candidate-order sensitivity

Candidate generation is never repeated. Apply 1,000 post-hoc permutations of candidate ordinals to each test problem.

Use:

`bon-safe-selection-order-permutations-v1-2026-08-10`

For permutation \(k\), sort ordinals \(1,\ldots,16\) by:

```text
SHA256(
  permutation_string + "\n" +
  decimal(k) + "\n" +
  decimal(candidate_ordinal)
)
```

with ordinal tie-breaking.

The 1,000-line comma-joined permutation schedule has SHA-256:

`de246a58e5514c5d8a058b6a2e97af73a449d213ea4fd8e1c5b96c4284b052df`

Frozen thresholds are applied unchanged. Position-dependent margins use the candidate’s permuted stream position, not its original ordinal.

Order-permutation survival requires:

For each baseline in
\(\{\mathrm{Policy\,0},\mathrm{Policy\,4},\mathrm{Policy\,5}\}\):

- the median Policy-2 accuracy advantage at \(N=16\) is at least +3 points;
- the fifth percentile of that advantage is strictly positive.

The following global conditions also apply:

- beneficial-acquisition retention is at least 90% in at least 950/1,000 permutations;
- Policy 2 has lower optimism and fewer harmful-switch examples in at least 950/1,000 permutations;
- Policy 2 performs at least one replacement in every permutation.

## CONFIRM

**CONFIRM** requires all of the following ruling conditions, verbatim:

- at least +3 test points at \(N=16\) over each of verifier argmax,
  verifier-blind answer plurality, and verifier-weighted answer plurality;
- a paired-bootstrap 95% lower confidence bound above zero for each of those
  three accuracy differences;
- at least 90% beneficial-acquisition retention;
- no collapse to “always keep sample one”;
- materially reduced score optimism and harmful switches;
- survival across candidate-order permutations.

For adjudication, these mean:

- for every baseline
  \(b\in\{\mathrm{Policy\,0},\mathrm{Policy\,4},\mathrm{Policy\,5}\}\),

  \[
  F_{\mathrm{Policy\,2}}(16)-F_b(16)\ge0.03;
  \]

- for every such baseline, the paired-bootstrap 95% lower bound for
  \(F_{\mathrm{Policy\,2}}(16)-F_b(16)\) is strictly greater than zero;
- \(BA_{\mathrm{Policy\,2}}/BA_{\mathrm{Policy\,0}}\ge0.90\);
- Policy 2 selects an index other than 1 for at least one test problem and records at least one beneficial acquisition;
- Policy-0-minus-Policy-2 optimism is at least 0.02 with paired-bootstrap lower bound \(>0\);
- Policy 2 reduces the harmful-switch-example rate by at least 20% relative to Policy 0, with paired-bootstrap lower bound \(>0\);
- every order-permutation survival condition above passes.

Every condition is conjunctive. There is no “partial confirm,” “promising,” best-policy rescue, or qualitative override.

CONFIRM supports only:

> On this frozen generator–verifier pair and frozen GSM8K train-split cohort,
> a calibration-frozen incumbent protocol achieved higher \(N=16\) accuracy
> than verifier argmax, unweighted self-consistency, and verifier-weighted
> voting while reducing harmful switching and retaining at least 90% of
> verifier argmax's beneficial acquisitions.

It does not establish arbitrary-model, arbitrary-verifier, cross-task, or theorem-level anytime safety.

## KILL

**KILL** means the run is valid but any CONFIRM condition fails.

KILL includes:

- no beneficial-acquisition headroom under verifier argmax;
- no harmful-switch headroom under verifier argmax;
- failure to retain at least 90% of beneficial acquisitions;
- accuracy improvement below +3 points;
- paired confidence interval touching or crossing zero;
- improvement obtained by never replacing the first sample;
- optimism reduction below 0.02;
- harmful-switch reduction below 20%;
- failure under the frozen order-permutation criteria;
- multiplicity-aware selection losing to or merely matching any adjudicating baseline;
- ordinary verifier argmax showing no relevant degradation or selection gap in this frozen system.

A null or negative result kills the **model-agnostic product claim** tested here. It does not harm, weaken, or reinterpret the recurrent-controller hypothesis, whose architecture-specific trajectory geometry may still matter.

## VOID

**VOID** is reserved for integrity or protocol failures:

- cohort overlap, wrong cohort hash, wrong dataset revision, or split-fingerprint mismatch;
- generator or verifier identity/revision mismatch;
- checkpoint substitution;
- prompt, parser, decoding, temperature, seed, order, or stopping drift;
- test responses or labels available before calibration parameters are frozen;
- any test-label use in threshold selection;
- incomplete candidate bank;
- missing candidate score, correctness label, seed, or stopping record;
- failed repeat determinism;
- verifier-score aggregation drift;
- NaN, infinite, or out-of-range scores;
- broken policy implementation or failed accounting identity;
- missing test problem or candidate;
- unresolved independent review finding;
- hardware failure preventing completion of the frozen bank and registered analyses;
- overwritten, regenerated, or relocated evidence.

A VOID makes no scientific statement and authorizes no fallback cohort, checkpoint substitution, parser redesign, threshold change, or partial-grid claim. Any retry requires a steering decision and, where protocol changes are needed, a pre-data amendment.

## Compute and resource budget

There is no training.

Bank generation must fit the standing approximately 2.5 GPU-hour single-run cap. If measured preflight throughput projects the frozen 256-calibration-plus-512-test, 16-candidate bank over that cap, a pre-data cohort-resize amendment is **REQUIRED before generation**. No retained generation may begin under an over-cap projection, and no partial bank or adaptive mid-run resize is authorized.

Maximum generation is:

\[
768\times16=12{,}288
\]

responses and:

\[
12{,}288\times256=3{,}145{,}728
\]

possible generated tokens, before natural stopping.

Every response is verifier-scored once. Post-hoc policy evaluation, bootstrap analysis, and permutations are CPU-only.

GPU execution uses the repository’s single 24 GB device with an approximately 95 W active power target. Report sampled power, average active power, peak power, wall time, and watt-hours; do not report provider billing or financial cost.

The default memory plan is staged:

1. load `base-C` in BF16, generate and persist the resumable local bank, then unload it;
2. load `verifier-V` in BF16 and score the frozen bank;
3. target approximately 3–5 GiB during generation and approximately 14–15 GiB during staged verification;
4. target an 8–15 GiB active-stage envelope, subject to measured preflight;
5. use co-residency only if a reviewed preflight demonstrates total peak allocated and reserved VRAM below 24 GiB with safe headroom.

Co-resident BF16 execution is not assumed to remain inside the 8–15 GiB staged envelope. If co-residency requires materially more memory, staged scoring is mandatory.

Use workspace-local model and dataset caches. The runner must checkpoint progress to an ignored resumable work area and refuse to overwrite a completed evidence artifact. Operational resume must preserve the exact seed, cohort, order, response, and score contract.

## Evidence artifact and landing contract

The canonical immutable result is:

`experiments/07_safe_selection/results/exp_bon_safe_selection.json`

It must contain:

- complete protocol and hash bindings;
- manifest identity digests without private identifiers;
- calibration-selected parameters and complete calibration tables;
- all 12,288 candidate records or cryptographically bound sidecar records;
- exact correctness and four-category extraction accounting;
- verifier step scores and aggregate score;
- policy outputs at every prefix;
- all switch ledgers with numerators and denominators;
- bootstrap and permutation schedules and results;
- compute, wall-time, power, and VRAM telemetry;
- one terminal outcome token: `CONFIRM`, `KILL`, or `VOID`.

A new result must update in the same block:

- `experiments/ledger.jsonl`;
- `experiments/EXPERIMENTS.md`;
- this canonical preregistration with only a result pointer, not rewritten rules;
- the relevant scientific synthesis;
- `STATUS.md`;
- a new read-only audit binding the artifact and claims.

## Relationship to E2 and the 0.5B program

This pilot:

- cannot alter E2;
- cannot alter the 0.5B registrations;
- cannot rescue an E2 FAIL or VOID;
- cannot veto an E2 PROCEED;
- cannot satisfy any 0.5B critic, transfer, task-band, mechanics, or compute-matching gate;
- cannot reuse E2 as confirmatory evidence for independent sampling;
- cannot reinterpret the base-A or base-B artifacts.

A KILL kills the model-agnostic safe-selection product claim without harming the recurrent-controller hypothesis.

A CONFIRM triggers, but does not decide, a program-priority reframe. Before additional full-program work proceeds by momentum, initiate a genuine 2–3-round Codex steering dialogue to decide whether:

1. safe selection becomes the product thesis and the recurrent controller becomes an efficiency backend;
2. both lines proceed with explicit independent claims; or
3. the controller retains priority because its one-decode or compute advantages dominate.

No displacement, pivot, or reprioritization occurs automatically from CONFIRM. The decision must be recorded through steering and reflected in `STATUS.md`.

## Registered prediction for a verifier-quality follow-up

This prediction does not alter the primary pilot and cannot rescue or veto its
outcome. A separate pre-data registration must freeze the verifier-degradation
operator, quality grid, calibration procedure, and randomization schedule
before the follow-up is run.

Let

\[
\Delta_\eta(N)=
F_{\mathrm{Policy\,2},\eta}(N)
-
F_{\mathrm{Policy\,0},\eta}(N),
\]

where \(\eta\) indexes a preregistered verifier-quality family.

The prediction is non-monotone:

1. At a zero-signal verifier endpoint, where scores are independent of
   correctness and candidate position under the registered iid bank,
   \(\mathbb E[\Delta_\eta(N)]=0\). Finite-sample deviations are not evidence
   of a mechanism.

2. At an oracle endpoint \(q=c\), verifier argmax reaches oracle candidate
   coverage. Policy 2 cannot improve upon it. It may equal argmax when every
   correct challenge clears the frozen margin, or perform worse when the
   margin delays or rejects a beneficial challenge.

3. A positive hysteresis benefit, if the proposed mechanism is real, must
   occur at intermediate verifier quality: wrong candidates must still enter
   the verifier's upper tail often enough to create harmful argmax
   replacements, while correct challengers must remain sufficiently separated
   that the margin rejects more harmful than beneficial replacements.

For a margin \(m\), the local mechanism is the balance between

\[
P(c_{\mathrm{inc}}=1,c_{\mathrm{chall}}=0,
  0<q_{\mathrm{chall}}-q_{\mathrm{inc}}\le m),
\]

the harmful argmax replacements prevented by hysteresis, and

\[
P(c_{\mathrm{inc}}=0,c_{\mathrm{chall}}=1,
  0<q_{\mathrm{chall}}-q_{\mathrm{inc}}\le m),
\]

the beneficial argmax replacements lost or delayed by hysteresis. Because
incumbent occupancy changes sequentially, these terms must be measured over
the realized policy trajectory rather than treated as independent static
pairs.

The expected qualitative curve is therefore zero at zero signal, positive
only in an intermediate region if the mechanism exists, and nonpositive at
oracle quality. No strict single peak or monotone dependence on AUROC is
registered.

Report at every quality level:

- the complete empirical ROC curve and AUROC;
- upper-tail false-positive rates at the score thresholds actually capable of
  replacing incumbents;
- conditional correct-to-wrong and wrong-to-correct score-gap distributions;
- first-candidate accuracy and oracle coverage;
- correct-answer vote share;
- largest wrong-answer vote share;
- unique-answer fraction and answer-frequency entropy;
- Policy-2-minus-argmax, Policy-2-minus-unweighted-vote, and
  Policy-2-minus-weighted-vote accuracy.

Answer agreement does not directly enter Policies 0–2. It determines the
strength of the voting controls and therefore the product-level phase
boundary. Diffuse wrong answers and repeated correct answers favor voting;
a concentrated systematic wrong mode weakens voting and raises the verifier
quality required for a verifier-guided selector to be useful.

ROC-n-reroll does not provide this follow-up's answer in closed form. Its ROC
geometry characterizes static Best-of-N under its assumptions. A raw-score
margin is not ROC-invariant, sequential incumbent selection is path-dependent,
and voting requires the joint distribution of answer identity, correctness,
and verifier score. Any closed-form extension would require additional
parametric assumptions or a state recursion over incumbent score,
correctness, and answer counts.

## Amendments

Any amendment must be appended before the affected data exist and must state:

- date and evidence boundary;
- exact ambiguity or defect;
- whether it changes an adjudicating rule;
- why it does not use calibration or test outcomes;
- all new hashes and review requirements.

No amendment may convert a valid KILL into CONFIRM.

## Pre-launch steps

1. **Land the new canonical line.** Create this preregistration, add its canonical mapping and line state to `STATUS.md`, append a zero-metric designed event to `experiments/ledger.jsonl`, add the chronology entry to `experiments/EXPERIMENTS.md`, and ignore `experiments/07_safe_selection/_local_manifest.md`.

2. **Populate the local manifest.** Record `base-C` and `verifier-V` identifiers, revisions, tokenizer revisions, resolved paths, licenses, file digests, local content digests, dtype, selection provenance, and library versions.

3. **Write one reusable runner.** It must support cohort-only validation, canonical retained smoke, resumable generation, staged scoring, evaluation-only replay, and refusal to overwrite immutable evidence. No separate one-off analysis scripts.

4. **Fill and review every launch-blocking hash slot.** Run the CPU-only cohort/provenance pass, bind prompt serialization, parser source, runner source, and the private-manifest identity digest, and re-derive every supplied cohort binding without loading a model.

5. **Measure preflight throughput against the compute cap.** If the complete frozen bank projects over approximately 2.5 GPU-hours, append and review a pre-data cohort-resize amendment before any retained generation. No launch is authorized by this preregistration under an over-cap projection.

6. **Perform a retained calibration smoke.** Use only the first two canonical calibration problems and their canonical 16 seeds. Those outputs must become part of the final calibration bank; they may not be discarded and regenerated.

7. **Obtain independent pre-launch pipeline review.** Review the full runner for cohort integrity, leakage, deterministic seed use, process-reward step serialization, scoring aggregation, all six policies, threshold leakage, bootstrap pairing, permutation logic, checkpoint/resume correctness, disk growth, VRAM/RAM safety, process cleanup, artifact overwrite protection, and compute-cap compliance. Resolve every blocking finding before launch.

8. **Run the calibration stage and freeze parameters.** Land or cryptographically freeze the calibration artifact and selected thresholds before any test generation.

Cohort allocation is resolved by the revision-pinned train-split construction above. The remaining launch blockers are the unfilled hash slots, local manifest identities, runner build, and independent review. This registration authorizes no GPU work by itself.

## Amendment 2026-08-10 — pre-data slot-filling pass

This amendment fills the four launch-blocking provenance slots after the
CPU-only cohort pass and corrects one supplied schedule digest. It was appended
before any selected calibration/test response, verifier score, threshold, or
outcome existed. One out-of-range synthetic engineering fixture was used only
to verify staged model loading and the documented verifier interface; it was
not retained and did not enter cohort construction, policy selection, or any
metric.

The original permutation digest was inconsistent with the already-frozen
algorithm and its stated 1,000-line serialization. Re-derivation found that
sorting ordinals exactly as registered, serializing each permutation as
comma-joined decimal ordinals, joining rows with LF, and writing no terminal LF
has SHA-256
`354b34427aa1ff857b6c281f57cd07a7b24a0ebfd4771f2645f1b3dfa1dcca14`,
not `de246a58e5514c5d8a058b6a2e97af73a449d213ea4fd8e1c5b96c4284b052df`.
Only the binding is corrected; the seed string, sort rule, ordinal tie-break,
permutation count, and every adjudicating condition remain unchanged.

The prompt digest is over compact canonical JSON records containing partition,
cohort position, dataset index, and the native-chat-template prompt string for
all 256 calibration and 512 pilot-test prompts in frozen order. The parser
digest is over the qualified E1 parser components returned by
`parser_source_text()`. The runner digest is over the runner's raw source
bytes. The private manifest identity digest is over compact canonical JSON of
all parsed manifest entries; it discloses no private entry.

- `prompt_serialization_sha256`: `53a11fe44669f4558f98addf056ec03caf23e44fc386d3638b1048073f81558d`
- `parser_source_sha256`: `4555cf412805a261cfc8de094990ac700566022f420d556bd7860bbd7ea3c3a4`
- `runner_source_sha256`: `2c36b2dacfb2b39f066c1c98c91bb8754d59f1f540bb494586b7d1f01ecc7633`
- `local_manifest_identity_digest`: `b1f285c108843ea435e3d398c790f7fd4bac13cc529daee9ff9380a55be9f625`

All supplied cohort, prior-consumption, content, and generation-seed hashes
revalidated exactly in the slot-filling pass. The runner imports the qualified
E1 numeric parser directly. Independent pre-launch review remains mandatory
and must bind this exact runner and private-manifest identity before any
canonical calibration or test run.

## Amendment 2026-08-10 — batched-generation cap resolution and verifier-viability gate

### Evidence boundary

This amendment is appended after the retained two-problem × 16-candidate
calibration smoke and before any additional calibration response, any test
response, any calibration threshold, or any adjudicating result exists.

The retained smoke comprises calibration cohort positions 0 and 1 and remains
immutable. It was generated with the originally registered batch-size-1
schedule. It will remain in the final calibration bank and will not be
discarded, overwritten, or regenerated.

This amendment changes the post-smoke generation schedule and adds a pre-test
verifier-viability gate. It does not change any CONFIRM threshold, permit use
of test outcomes during calibration, or convert a valid KILL into CONFIRM.

### Non-adjudicating aggregation diagnostic

The retained 32 responses were replayed from their immutable stored step
scores under four response-level aggregations. The results were:

- minimum: correct mean 0.1541736000, incorrect mean 0.1567845518,
  candidate-level AUROC 0.3141025641;
- product: correct mean 0.1367948360, incorrect mean 0.0541080174,
  candidate-level AUROC 0.6794871795;
- last: correct mean 0.1542501301, incorrect mean 0.1567845518,
  candidate-level AUROC 0.3269230769;
- arithmetic mean: correct mean 0.1544356712, incorrect mean 0.1600666290,
  candidate-level AUROC 0.2243589744.

These values cover only two problem clusters. Twenty-five of 32 responses
contain one serialized verifier step, so product is additionally confounded
by step count. They are diagnostic only and did not select the aggregation.

The canonical candidate score remains the originally registered minimum
positive-class probability over steps. Product, last-step, mean, or any other
aggregation is non-adjudicating and may not replace it in this attempt.

Before further retained generation, the scorer must replay the verifier's
published golden example and reproduce every published step score within an
absolute tolerance of 0.002 under the frozen BF16 implementation. Failure is
a launch-blocking implementation discrepancy, not a scientific KILL.

### Batched-generation engineering preflight

Candidate generation after calibration position 1 may use batch size 8 or 16.
Batches are formed within one problem only:

- batch size 8 generates candidate ordinals 1–8 and then 9–16;
- batch size 16 generates candidate ordinals 1–16 in one call;
- prompts are identical within a batch;
- batch row order is candidate-ordinal order;
- `num_return_sequences` remains 1 per input row;
- tokenizer padding side is left;
- attention masks and per-row stopping states are mandatory.

The original retained smoke remains a batch-size-1 prefix. Exact
reproducibility therefore means exact reproduction under this mixed frozen
schedule, not equivalence to standalone batch-size-1 sampling.

For every post-smoke batch define:

batch_payload =
    "bon-safe-selection-batched-generation-v1-2026-08-10" + "\n" +
    partition + "\n" +
    decimal(dataset_index) + "\n" +
    decimal(first_candidate_ordinal) + "\n" +
    decimal(batch_size) + "\n" +
    comma_join(decimal(candidate_identity_seed) in batch-row order)

batch_seed =
    uint64_big_endian(SHA256(batch_payload)[0:8]) mod (2^63 - 1)

Set the CPU and CUDA RNGs to `batch_seed` immediately before the corresponding
generation call. The originally registered per-candidate seeds remain frozen
candidate identity fields; for post-smoke records they are not claims of
standalone per-candidate RNG reproduction. Every record must additionally
store its batch seed, batch size, batch row, and batch-payload SHA-256.

Benchmark batch size 8 first on diagnostic duplicates of retained calibration
positions 0 and 1. Run the complete two-problem schedule twice. If batch size
8 is ineligible, benchmark batch size 16 identically. Diagnostic duplicates
do not enter the bank or any metric.

A batch size is eligible only if:

1. the two executions match exactly for response bytes, stopping reason,
   generated-token count, extracted answer, and verifier-input serialization
   for all 32 candidates;
2. peak reserved generation VRAM is at most 20 GiB;
3. every completed row remains stopped while unfinished rows continue;
4. no NaN, CUDA error, process leak, or checkpoint inconsistency occurs.

Repeat determinism establishes reproducibility only. It is not evidence that
sampled generation is invariant to batching or padding.

### Cap projection and batch selection

For an eligible batch size b, let g_b be the larger of the two measured
diagnostic generation wall times divided by 32. Let s be the retained smoke
scoring wall time divided by 32. Include measured model-load overhead.

For a bank containing M problems, define:

H_b(M) =
    retained_smoke_generation_wall_time
    + g_b × (16M - 32)
    + s × 16M
    + measured_generation_and_scoring_model_load_overhead.

Use 8,100 seconds, 90% of the approximately 2.5-hour cap, as the
launch-authorizing ceiling. The remaining 900 seconds are operational
headroom and may not be converted into planned bank size.

If the complete 768-problem bank satisfies H_b(768) <= 8,100 seconds, choose
the smallest eligible batch size that satisfies it and retain all 256
calibration plus 512 test problems.

If no eligible batch size fits the complete bank, choose the eligible batch
size with the smallest g_b and let M* be the largest integer M <= 768 for
which H_b(M) <= 8,100 seconds.

Allocate the resized cohorts as follows:

- if M* >= 640: test count = 512 and calibration count = M* - 512;
- if 512 <= M* < 640: calibration count = 128 and test count = M* - 128;
- if M* < 512: no canonical launch is authorized.

Every resized cohort is the corresponding prefix of the already frozen
ordered cohort. Candidate count remains N=16. Calibration and test remain
disjoint. All denominators and the paired-bootstrap modulus change to the
resolved test count; all percentage-point CONFIRM thresholds remain
unchanged. The candidate-order permutation schedule remains unchanged.

No mid-run cohort resize is permitted. Operational resume preserves the
resolved bank and counts cumulative retained-bank GPU time. Splitting the
bank across process invocations does not create multiple 2.5-hour allowances.

### Calibration verifier-viability gate

Before test generation, the complete resolved calibration bank must contain:

- at least 40 correct and at least 40 incorrect candidates;
- at least 20 verifier-argmax beneficial-acquisition events at N=16;
- at least 20 verifier-argmax harmful-switch examples at N=16;
- at least two distinct finite candidate scores.

If any floor is missed, stop before test generation with
`PREFLIGHT_STOP_INSUFFICIENT_CALIBRATION_HEADROOM`. This is neither CONFIRM,
KILL, nor VOID and makes no test-set claim. Further work requires steering.

### Remaining bindings

This amendment is not launch-authorizing until a subsequent pre-data
slot-filling paragraph records:

- selected batch size;
- measured duplicate timings and both determinism digests;
- resolved calibration and test counts;
- all resized ordered-index, sorted-index, row-content, combined-cohort,
  prompt-serialization, generation-schedule, and batch-seed-schedule hashes;
- amended runner-source SHA-256;
- independent-review attestation bound to that runner and schedule.

No further retained generation may occur before those bindings are appended
and independently reviewed.

## Terminal verifier-compatibility preflight record — 2026-08-11

Status: **`PREFLIGHT_STOP_IMPLEMENTATION_DISCREPANCY / CANONICAL NOT RUN`**

The revision-pinned verifier model card states `transformers>=4.40.0`. The
two predeclared documented-compatible stack attempts are exhausted. Both held
the exact frozen golden input, tokenizer revision, checkpoint revision, BF16
dtype, `use_cache=False`, expected scores, and 0.002 absolute tolerance fixed.
No calibration or test row was accessed and no retained response was generated.

| Attempt | Environment identity | Environment SHA-256 | Observed step scores | Absolute errors | Max error | Scored tokens | Scored-forward time | Invocation wall | Outcome |
|---|---|---|---|---|---:|---:|---:|---:|---|
| 1 | Python 3.13.7; Transformers 4.47.1; Torch 2.11.0+cu128; CUDA 12.8; tokenizers 0.21.0; hub 0.27.1; safetensors 0.8.0; Windows 10.0.26200.8875 | `f69237de68a7c71c4a37e905d7a1249046b56d9a9318337c3bc8ce1c36cd4557` | `1.0 / 0.154296875 / 0.97265625 / 1.0` | `0 / 0.0361328125 / 0.00390625 / 0` | 0.0361328125 | 454 | 0.2904227s | approximately 14.2s | FAIL |
| 2 | Python 3.13.7; Transformers 4.48.0; Torch 2.11.0+cu128; CUDA 12.8; tokenizers 0.21.0; hub 0.27.1; safetensors 0.8.0; Windows 10.0.26200.8875 | `771ed19944ffb63bc889cbd55a8cdfb58e47760dfffc8088c203172da4f58b2c` | `1.0 / 0.154296875 / 0.97265625 / 1.0` | `0 / 0.0361328125 / 0.00390625 / 0` | 0.0361328125 | 454 | 0.2796334s | approximately 13.9s | FAIL |

The environment hashes are SHA-256 over the UTF-8 one-line canonical JSON
identity, including its final LF, with sorted keys exactly corresponding to
the fields listed in the table. The verifier input SHA-256 was
`e14ec22c375a3dbc31596964e53fe1a59b4f7264b4c73f5dcd80a1bbb3f52741`
in both attempts. Stack preparation took approximately 38.1 seconds for
attempt 1 and 23.3 seconds for attempt 2. Total active diagnosis completed in
under 15 minutes against the registered 90-minute cap.

### Deviations and adjudication boundary

- A non-adjudicating attempt-1 default-cache compatibility probe preceded the
  exact frozen replay. It produced the same observed score vector and did not
  select a stack, threshold, or outcome; the registered `use_cache=False`
  replay controls.
- Attempt 2 first encountered an unrelated optional-image dependency during
  `AutoModel` class-map enumeration, before verifier load or scoring. The exact
  revision-bound remote `AutoModel` class was then loaded directly through
  Transformers' dynamic-module resolver. This changed no verifier source,
  tensor, tokenizer, serialization, dtype, score extraction, or threshold.
- No eager/flash backend sweep, package-version sweep beyond the two declared
  stacks, threshold relaxation, verifier substitution, calibration/test
  inspection, or retained generation occurred.

The registered branch therefore stops at
`PREFLIGHT_STOP_IMPLEMENTATION_DISCREPANCY`. This is an implementation
preflight outcome, not a scientific KILL or evidence about safe selection.
The runner fails closed for GPU/execution modes. A new compatibility attempt,
verifier swap, or scientific continuation requires round-2 steering and a
fresh pre-data registration where the protocol boundary changes.
