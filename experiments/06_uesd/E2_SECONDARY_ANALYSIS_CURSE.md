# E2 Optimizer’s-Curse Secondary Analysis Registration

**Date:** 2026-08-10  
**Status:** Pre-data, secondary, informational only  
**Scope:** E2 mechanics pilot only. This analysis cannot affect E2 adjudication.

## Question and prediction

Arm 3 selects the maximum critic score among the first \(B\) states. If critic scores contain approximately independent noise, maximization should preferentially select positive critic errors. Under the idealized homoscedastic Gaussian model, selected error grows approximately as

\[
\sigma\sqrt{2\ln B}.
\]

The registered prediction is therefore:

1. arm 3 develops increasing selected-state critic optimism as \(B\) grows;
2. harmful arm-3 replacements are disproportionately associated with critic-score errors and, under the hysteresis-remediable version of the mechanism, small winning margins;
3. arm 4 suppresses the low-margin component while retaining most of the useful correct-state acquisition.

This is a secondary mechanism analysis, not a new E2 gate.

## Available records and notation

For test example \(i\), model seed \(s\), and horizon \(t\), define:

\[
Y_{ist}=\mathbf 1(\hat y_{ist}=y_i),\qquad
q_{ist}=\text{frozen critic score},\qquad
e_{ist}=q_{ist}-Y_{ist}.
\]

The realized residual \(e\) is the critic-score minus realized-correctness calibration gap. It is observable, but it is not the latent Gaussian error around an unobserved true correctness probability.

For arm \(a\in\{3,4\}\), let \(J^a_{is}(B)\) be its selected horizon at budget \(B\). Arm 3 uses cumulative first-argmax selection; arm 4 uses its calibration-frozen \(\delta_s\). The analysis uses only the selected arm-4 threshold and will not evaluate test outcomes for unselected grid values.

Required fields already recorded per example are the gold label, raw predictions through \(T=32\), arm-3 and arm-4 selected horizons and predictions for \(B=1,\ldots,32\), critic scores through \(T=32\), model seed, example ID, and counterfactual-group metadata.

## Registered observables

### 1. Selected-state optimism

For every arm, budget, and seed, report

\[
G_{a,s}(B)=
\frac1N\sum_i
\left(q_{is,J^a_{is}(B)}-Y_{is,J^a_{is}(B)}\right).
\]

This is the direct selected-state score-versus-correctness gap.

Because critic calibration may drift with horizon, also compute a horizon-matched baseline. Let

\[
\mu_{s,t}=\frac1N\sum_i e_{ist},
\qquad
w_{a,s,t}(B)=\frac1N\sum_i\mathbf 1[J^a_{is}(B)=t].
\]

Define horizon-adjusted selection optimism as

\[
X_{a,s}(B)=G_{a,s}(B)-\sum_{t\le B}w_{a,s,t}(B)\mu_{s,t}.
\]

Thus \(X_a(B)\) measures whether the selector preferentially chooses positive realized residuals beyond what would follow merely from its horizon mix. By construction, \(X_a(1)=0\).

Report:

- \(G_3(B)\), \(G_4(B)\);
- \(X_3(B)\), \(X_4(B)\);
- paired contrasts \(G_3(B)-G_4(B)\) and \(X_3(B)-X_4(B)\);
- selected critic score, selected correctness, and selected horizon separately, so score inflation is not conflated with accuracy.

### 2. Extreme-value shape

The fixed diagnostic grid is \(B=\{2,4,8,16,32\}\). Fit the following one-parameter shapes to \(X_3(B)\), all anchored at zero at \(B=1\):

| Model | Shape |
|---|---|
| Extreme-value | \(\beta\sqrt{2\ln B}\) |
| Flat after first opportunity | \(\beta\mathbf 1(B>1)\) |
| Saturating | \(\beta(1-1/B)\) |

Use unweighted squared error on the fixed grid. Report the fitted coefficient, residual sum of squares, bootstrap distribution of pairwise fit differences, and bootstrap frequency with which each model fits best. This comparison is descriptive; no model-selection result changes E2.

As a scale check, report

\[
\hat\sigma_s=
\operatorname{SD}_{i,t}
(e_{ist}-\mu_{s,t})
\]

and \(\hat\beta/\hat\sigma\). Because binary realized correctness contributes outcome noise and trajectory errors need not be independent or Gaussian, agreement with coefficient one is only suggestive.

### 3. Switch events and winning margins

At every offered challenge \(t>1\), reconstruct each arm’s incumbent \(j=J^a(t-1)\) and define

\[
m^a_{ist}=q_{ist}-q_{isj}.
\]

Classify accepted replacements as:

- harmful: incumbent correct, challenger incorrect;
- beneficial: incumbent incorrect, challenger correct;
- correct-to-correct;
- wrong-to-wrong.

For harmful and beneficial replacements separately, report by transition and cumulatively through each \(B\):

- offered and accepted numerators and denominators;
- acceptance hazard among eligible challenges;
- number and fraction of examples experiencing at least one event;
- mean, median, interquartile range, 90th percentile, and empirical CDF of \(m\);
- for arm 3, fractions with \(0<m\le\delta_s\) and \(m>\delta_s\);
- for arm 4, rejected beneficial-challenge rate and correct-incumbent survival;
- arm-3 minus arm-4 paired differences in harmful-event incidence, beneficial acquisition, and total replacements.

If \(\delta_s=0\), arm 4 is algorithmically equivalent to arm 3 for that seed; arm-3-versus-arm-4 mechanism discrimination is then unavailable for that seed.

The Gaussian extreme-value prediction applies directly to selected error, not to per-transition harmful-switch hazard. Under an exchangeable iid record process, the probability that the next candidate sets a new maximum is approximately \(1/t\), so per-step acceptance can decline even while cumulative selected-score inflation grows. The \(\sqrt{\ln B}\) comparison will therefore be applied to \(X_3(B)\), not asserted as the necessary shape of transition hazard. Cumulative harmful incidence will be reported as a bounded companion diagnostic.

## Statistical treatment

The test set contains 4,096 distinct examples arranged in 2,048 counterfactual groups and evaluated under both model seeds. This yields 8,192 seed-example records, but not 8,192 independent examples.

All quantities will be reported:

- separately for each seed;
- pooled by concatenating numerators and denominators, never by averaging seed rates;
- with raw counts and denominators.

Uncertainty will use 10,000 cluster-bootstrap replicates with fixed seed `20260811`. The resampling unit is the counterfactual group: both group members and their records under both model seeds travel together. This preserves counterfactual pairing, repeated-budget dependence, and shared-example dependence across seeds. Report percentile 95% intervals and simultaneous bootstrap bands across \(B=1,\ldots,32\) for the principal curves.

With only two trained model seeds, between-training-seed variance is not estimable. Bootstrap intervals quantify test-example uncertainty conditional on these two fitted models. Seed agreement or disagreement will be shown directly; pooled precision must not be presented as broad seed generalization.

### What E2 can estimate

E2 can estimate, conditional on its synthetic test distribution and two fitted critics:

- whether arm 3 selects states with increasingly positive realized residuals;
- whether that increase survives horizon adjustment;
- the paired operational effect of the frozen arm-4 rule on the same state bank;
- whether harmful replacements are predominantly low- or high-margin;
- the balance between harmful-switch suppression and rejected beneficial switches.

### What remains merely suggestive

The following are suggestive only:

- preference for the \(\sqrt{2\ln B}\) curve over flat or saturating shapes;
- an inferred Gaussian noise scale;
- independence or homoscedasticity of critic errors;
- generalization across model seeds, tasks, architectures, or scales;
- attribution of all accuracy loss to optimizer’s curse rather than score bias, schedule drift, or insufficient semantic information.

The future 0.5B experiment is required to test whether any observed signature survives real-task trajectories, the larger model and controller interface, and the full registered comparison regime. E2 cannot establish that transfer.

## Interpretation table

| Observable pattern | Registered interpretation |
|---|---|
| \(X_3(B)\) rises with budget; arm-3 harmful switches are concentrated at \(0<m\le\delta_s\); arm 4 lowers \(X\), harmful incidence, and late-horizon headroom while retaining gain | **Curse suppression:** repeated low-margin replacement is an operative failure mechanism |
| Harmful switches are predominantly \(m>\delta_s\); arm 4 removes some replacements but does not materially reduce optimism, harmful incidence, or late accuracy loss | **High-margin misranking:** critic errors, bias, drift, or insufficient semantic information exceed what hysteresis can repair |
| Arm 4 improves correct-incumbent survival and reduces harmful switches, but rejects many beneficial challenges and loses acquisition gain or gain retention | **Plasticity–stability tradeoff:** stability is purchased by blocking useful updates |
| Arm 4 and arm 3 are nearly identical because \(\delta_s=0\) | No arm-4 discrimination; analyze arm-3 optimizer’s-curse signatures only |
| Patterns differ materially between the two seeds | Mechanism unstable or seed-dependent; pooled result is not a general conclusion |

## Falsification and disconfirmation

The optimizer’s-curse mechanism is disconfirmed within E2 if arm 3 shows no positive budget-dependent horizon-adjusted selection optimism: \(X_3(B)\) is flat or decreasing, the fitted extreme-value coefficient is non-positive, and arm-3 selection is no more optimistic than the horizon-matched opportunity bank. A flat raw gap accompanied by horizon drift is insufficient; the horizon-adjusted result controls.

The narrower claim that repeated low-margin replacements explain arm-3 failure is falsified if harmful switches are dominated by margins above the frozen \(\delta_s\) at all budgets, or if arm 4 suppresses low-margin replacements without reducing harmful incidence, selected optimism, or late accuracy loss.

Large-margin harmful switches do **not**, by themselves, falsify optimizer’s curse in its general form: maximization can harvest large positive critic errors. They falsify the low-margin, hysteresis-remediable interpretation and favor high-margin misranking.

Results that are imprecise, seed-conflicted, or compatible with all three curve shapes are inconclusive rather than supportive or falsifying.

## Non-adjudication constraints

Under the arm-4 and trajectory-diagnostics amendments in `experiments/06_uesd/PREREGISTRATION.md`, this analysis may not:

- change or select \(t^*\), \(\delta_s\), an endpoint, a subset, generator settings, labels, or an outcome;
- inspect or report test outcomes for unselected arm-4 threshold values;
- alter any PROCEED, FAIL, VOID, integrity, competence, or denominator calculation;
- rescue an arm-3 FAIL, veto an arm-3 PROCEED, or enter the regression-only-VOID exception;
- create a best-selector clause or promote arm 4 post hoc;
- tune thresholds, curve forms, budget subsets, or strata after seeing results;
- exclude difficult, model-empty, or otherwise unfavorable examples;
- reinterpret E2, authorize E2-CERT, or authorize the 0.5B experiment.

Any later protocol or design decision must proceed through the existing preregistration, review, and steering gates independently of this secondary analysis.

Registered pre-data 2026-08-10, before any E2 training or evaluation existed. Committed alongside the fix commits, before the pre-training review token was issued.