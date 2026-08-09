**Audit Verdict**

The evidence is promising, but the claim is overstated. D35b supports: “In the first completed prefix-sum pair, VT again shows lower measured k than FT, even when both models have near-identical eval loss and perfect TRAIN_T accuracy.” It does not yet justify: “k-suppression generalizes beyond addition,” “the strongest counterargument is refuted,” or “confidence should rise to 9/10.”

1. **Matched-loss validity**

The matched-loss test is useful, but not decisive.

At step 10K, both models have `eval_loss=0.0003` and `SeqAcc=1.0`, so the crude “VT is just further trained / lower loss” explanation is weakened. That is real evidence.

But “identical task mastery” is too strong. Hidden differences remain:

- Eval uses a finite standardized batch, apparently 1024 samples at `TRAIN_T`, not exhaustive task distribution.
- TRAIN_T accuracy is saturated, so it may not distinguish margins, calibration, robustness, or behavior at other iteration counts.
- VT has much stronger `T_MIN` behavior, so the models are not functionally equivalent across the dynamical horizon.
- Loss rounded to four decimals can hide meaningful differences at this scale.
- Same seed means this is one paired trajectory, not a population-level result.
- The variants train on different T regimes by design. Matched final loss does not equate matched training distribution, inductive bias, or optimization path.

So: the matched-loss test refutes a narrow version of the optimization-progress artifact claim, not the whole counterargument.

2. **N=1 seed and generalization**

N=1 is not sufficient for a generalization claim.

It is sufficient for: “first positive non-addition instance.”

Minimum evidence bar before saying “generalizes beyond addition”:

- Complete the planned D35b grid, or at least several seeds at L=6 and L=8.
- Show sign consistency: VT k < FT k in most or all paired seeds.
- Report paired effect size and uncertainty, not just one `dk`.
- Confirm the result is not specific to seed 42 or L=6.
- Ideally add another non-addition algorithmic task.

A defensible current wording would be: “Initial D35b result suggests k-suppression may extend to learnable prefix sum; replication pending.”

3. **Prefix-sum dk is 60% larger than addition**

This is weakly informative, not strong evidence.

The larger `dk=-0.0051` may reflect task structure, horizon mismatch, V=8 simplicity, saturation effects, measurement sensitivity, or architecture-task interaction. Prefix sum at V=8 is not merely “another task”; it has different carry/accumulation dynamics and a much worse FT `T_MIN` failure mode. That can amplify k differences without implying a broader law.

The 60% comparison is also between one prefix-sum pair and an addition D=6 estimate. Without seed variance for prefix sum, the ratio is not statistically meaningful. Treat it as an observation to explain, not evidence for stronger generality.

4. **k/rho dissociation**

This helps the **k-first** framing but hurts any clean rho-coupled narrative.

VT has lower k but higher rho on prefix sum. That clarifies that k and rho are not interchangeable, and it supports the idea that k is measuring the relevant finite-time contraction property better than rho.

But it also complicates the theory. If previous narratives implied VT generally suppresses spectral radius, this result is a counterexample. The right interpretation is:

- Good for: “k is the primary observable.”
- Bad for: “VT globally makes dynamics more spectrally contractive.”
- Requires: a sharper account of what directions k measures versus what rho measures.

5. **8.5 → 9/10 confidence**

Premature.

I would maybe move T5 k-contraction from 8.5 to 8.6 or 8.7, not 9.0. The result is directionally strong and mechanistically interesting, but it is still one seed, one length, one non-addition task, same architecture, same measurement stack.

A 9/10 claim should require replicated non-addition evidence and a tighter exclusion of measurement/task artifacts.

6. **Evidence required for 9/10**

I would require:

- Full D35b completion: 4 seeds × 2 lengths × 2 variants, with paired stats.
- Same sign of `dk` across most/all pairs.
- Confidence intervals or paired tests for k, T_99, and rho.
- Matched-loss comparisons across multiple seeds/checkpoints, not only seed 42 at 10K.
- Exact or much larger held-out evaluation, including loss precision beyond rounded `0.0003`.
- Robustness across T values: TRAIN_T, T_MIN, and longer-than-train horizons.
- At least one additional non-addition task.
- Ablation separating variable-T training from other training-distribution effects.
- Verification that k measurement itself is not biased by trajectory length, input distribution, or saturation.

Bottom line: D35b first pair is a valuable positive result. It weakens the optimization-progress objection and supports the k-first reframing. But the current claim should be downgraded from “generalized/refuted/9 out of 10” to “first replicated target pending; matched-loss artifact explanation substantially weakened in one controlled pair.”

