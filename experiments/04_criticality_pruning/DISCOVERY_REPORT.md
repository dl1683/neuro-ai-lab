# Activation-Flow / Path-Flow Pruning Discovery Report

## Executive summary

The original hypothesis for experiment 04 was "criticality-preserving pruning": preserve network dynamics near a branching ratio of `1` and outperform magnitude pruning. Real experiments did **not** validate that mechanism. Branching-ratio distance is tracked, but it does not explain the accuracy gains.

The live discovery is better and more concrete:

> **Label-free path-flow pruning preserves active input-hidden-output pathways and consistently outperforms magnitude pruning at high sparsity. In this benchmark it also matches or beats gradient saliency, despite using no labels or gradients at pruning time.**

## Core benchmark

Command: `python experiments/04_criticality_pruning/run.py`

Setup:

- Dataset: sklearn digits.
- Model: one-hidden-layer MLP trained from scratch.
- Seeds: `5`.
- Sparsities: `50%, 80%, 90%, 95%`.
- Methods: random, magnitude, gradient saliency, activation-flow, path-flow, path-coverage.
- No post-prune fine-tuning in the core benchmark.

Headline:

| Metric | Result |
|---|---:|
| Dense model mean accuracy | `95.87%` |
| Best label-free method | `path_flow` |
| High-sparsity regime | `90%, 95%` |
| Path-flow gain vs magnitude | `+17.44` points |
| Path-flow wins vs magnitude | `10 / 10` paired high-sparsity cases |
| Path-flow gain vs gradient saliency | `+0.76` points |
| Path-flow wins vs gradient saliency | `6 / 10` paired high-sparsity cases |

## Method

Path-flow scores weights by whether they sit on high-throughput input-hidden-output paths.

```text
path_importance[h] = hidden_strength[h] * hidden_balance[h] * output_strength[h]
W1_score[i, h] = |W1[i, h]| * input_activity[i] * path_importance[h]
W2_score[h, c] = |W2[h, c]| * path_importance[h]
```

Terms:

- `input_activity`: mean absolute input feature activation on an unlabeled calibration set.
- `hidden_strength`: mean hidden activation magnitude.
- `hidden_balance`: preference for hidden units that fire often enough to matter but are not always-on.
- `output_strength`: average absolute outgoing strength of the hidden unit.

The method is label-free at pruning time.

## Ablation

Command: `python experiments/04_criticality_pruning/ablate_path_flow.py`

Setup: `90%` and `95%` sparsity, five seeds.

| Variant | Mean accuracy |
|---|---:|
| `magnitude` | `61.46%` |
| `gradient_saliency` | `78.14%` |
| `path_flow_full` | `78.90%` |
| `no_input_activity` | `71.05%` |
| `no_hidden_strength` | `77.60%` |
| `no_hidden_balance` | `78.47%` |
| `no_output_strength` | `77.97%` |

Ablation drops from full path-flow:

| Removed ingredient | Accuracy drop |
|---|---:|
| Input activity | `7.85` points |
| Hidden strength | `1.30` points |
| Hidden balance | `0.43` points |
| Output strength | `0.94` points |

Interpretation: input activity is the dominant term. Full path-flow is still best, so the pathway score is not reducible to pure magnitude or pure hidden activation.

## Calibration sweep

Command: `python experiments/04_criticality_pruning/calibrate_path_flow.py`

Question: does path-flow need the whole training set to estimate activations?

Setup: `90%` and `95%` sparsity, five seeds, unlabeled calibration subsets.

| Calibration fraction | Mean examples | Mean accuracy | Drop vs full calibration |
|---:|---:|---:|---:|
| `2%` | `23` | `74.44%` | `4.47` points |
| `5%` | `58` | `75.95%` | `2.96` points |
| `10%` | `117` | `77.38%` | `1.53` points |
| `25%` | `292` | `78.14%` | `0.76` points |
| `100%` | `1168` | `78.90%` | `0.00` points |

Interpretation: path-flow is usable with small unlabeled calibration batches. `10%` calibration retains most of the full-calibration benefit.

## Post-prune fine-tuning

Command: `python experiments/04_criticality_pruning/finetune_path_flow.py`

Setup: `90%` and `95%` sparsity, five seeds, 55 masked fine-tuning epochs.

| Method | Before FT | After FT | Recovery gain | Dense retention after FT |
|---|---:|---:|---:|---:|
| `magnitude` | `61.46%` | `76.10%` | `14.64` points | `79.39%` |
| `gradient_saliency` | `78.14%` | `88.74%` | `10.60` points | `92.57%` |
| `path_flow` | `78.90%` | `89.03%` | `10.13` points | `92.87%` |

Interpretation: path-flow remains the best sparse starting point after masked fine-tuning. The edge over gradient saliency is small (`+0.29` points), but the edge over magnitude remains large (`+12.93` points).

## Extreme sparsity boundary

Command: `python experiments/04_criticality_pruning/extreme_sparsity.py`

Setup: one-shot pruning, five seeds, no fine-tuning.

| Sparsity | Magnitude | Gradient saliency | Path-flow |
|---:|---:|---:|---:|
| `97%` | `36.95%` | `58.89%` | `60.51%` |
| `98%` | `32.15%` | `47.15%` | `52.46%` |
| `99%` | `20.03%` | `31.54%` | `37.27%` |

Interpretation: path-flow remains best even into collapse. The method is not merely preserving accuracy in the comfortable `90-95%` range; it gives the strongest sparse subnetworks at `97-99%` too.

## What is actually discovered

Supported:

- Label-free path-flow is a strong high-sparsity pruning heuristic on trained sklearn-digits MLPs.
- It robustly beats magnitude pruning.
- It is competitive with, and often slightly better than, gradient saliency without needing labels or gradients at pruning time.
- Input activity is the most important component.
- Small unlabeled calibration batches are sufficient to recover most of the benefit.
- Post-prune fine-tuning preserves the path-flow advantage.

Not supported yet:

- The original branching-ratio criticality story.
- Generality to CNNs, transformers, CIFAR, or larger models.
- Superiority over more advanced pruning methods like GraSP, SynFlow, movement pruning, or iterative magnitude pruning with full training budgets.

## Next experiments

1. Reproduce on Fashion-MNIST with a Torch MLP.
2. Reproduce on CIFAR-10 with a small CNN.
3. Compare against SynFlow and SNIP more formally.
4. Test path-flow masks selected from out-of-distribution calibration data.
5. Test whether path-flow works before training or only after training.
6. Add confidence intervals and paired nonparametric tests for path-flow vs gradient saliency.

## Fashion-MNIST external-validity check

Commands:

- `python experiments/04_criticality_pruning/torch_fashion_mnist_path_flow.py`
- `python experiments/04_criticality_pruning/torch_fashion_mnist_layerwise.py`
- `python experiments/04_criticality_pruning/torch_fashion_mnist_input_signal.py`
- `python experiments/04_criticality_pruning/torch_fashion_mnist_no_balance_blend.py`

Setup:

- Dataset: Fashion-MNIST downloaded through torchvision.
- Model: Torch MLP, `784 -> 256 -> 10`.
- Training subset: `16k` examples.
- Test subset: `3k` examples.
- Seeds: `2`.
- Sparsities: `90%, 95%, 98%`.

Initial result: the sklearn path-flow formula failed badly on Fashion-MNIST. Pure magnitude beat full path-flow at all tested sparsities.

Failure analysis:

- Layerwise pruning did not fix the failure.
- Using per-pixel standard deviation instead of mean absolute normalized input helped only slightly.
- The hidden-balance term was strongly harmful on Fashion-MNIST.
- A conservative no-balance blend recovered useful signal.

Best Fashion-MNIST variant:

```text
score = |weight| * (std_input * hidden_strength * output_strength) ^ alpha
alpha = 0.25
```

Results for the best no-balance blend versus pure magnitude:

| Sparsity | Magnitude | No-balance blend | Delta |
|---:|---:|---:|---:|
| `90%` | `80.32%` | `79.27%` | `-1.05` points |
| `95%` | `68.60%` | `71.33%` | `+2.73` points |
| `98%` | `53.53%` | `60.55%` | `+7.02` points |

Interpretation:

Path-flow does **not** transfer naively from sklearn digits to Fashion-MNIST. But a corrected weak path-flow modulation does help near the compression cliff. The emerging pattern is:

- Moderate sparsity: magnitude is hard to beat.
- Severe sparsity: path information becomes useful.
- Dataset/model preprocessing matters; normalized image backgrounds can corrupt naive input-activity estimates.
- Hidden firing-rate balance is not universally valid and should be learned/tuned or removed.

This turns the finding from a single-dataset trick into a more precise hypothesis: **path-aware modulation is most valuable near the sparsity cliff, but the activation statistic must match the data modality.**

## Fashion-MNIST fine-tuning update

Command: `python experiments/04_criticality_pruning/torch_fashion_mnist_corrected_blend_finetune.py`

Setup:

- Torch Fashion-MNIST MLP.
- Corrected no-balance path/magnitude blend with `alpha=0.25`.
- Masked fine-tuning after pruning.
- Seeds: `2`.

| Sparsity | Magnitude after FT | Corrected path blend after FT | Delta |
|---:|---:|---:|---:|
| `95%` | `83.10%` | `83.40%` | `+0.30` points |
| `98%` | `79.65%` | `80.33%` | `+0.68` points |

Interpretation:

The corrected path signal gives its largest value before fine-tuning at extreme sparsity (`+7.02` points at `98%`). After masked recovery training, magnitude catches up substantially, but the corrected path blend still keeps a small edge. This suggests path-flow is primarily a **mask initialization advantage near the sparsity cliff**, not a magic replacement for post-prune training.

## CIFAR-10 MLP exploratory transfer

Command: `python experiments/04_criticality_pruning/torch_cifar10_mlp_path_flow.py`

Setup:

- Dataset: CIFAR-10 downloaded through torchvision.
- Model: Torch MLP, `3072 -> 512 -> 10`.
- Train subset: `20k` examples.
- Test subset: `5k` examples.
- Seeds: `2`.
- Sparsities: `90%, 95%, 98%`.
- Method: corrected no-balance path/magnitude blend with alpha sweep.

Result:

| Sparsity | Magnitude (`alpha=0`) | Best path blend | Delta |
|---:|---:|---:|---:|
| `90%` | `41.01%` | `41.23%` (`alpha=0.25`) | `+0.22` points |
| `95%` | `34.61%` | `34.88%` (`alpha=0.25`) | `+0.27` points |
| `98%` | `26.34%` | `28.32%` (`alpha=0.50`) | `+1.98` points |

Across all three sparsities, best average alpha was `0.50`, with `34.42%` mean accuracy versus `33.99%` for magnitude.

Interpretation:

CIFAR-10 MLP does not show the large sklearn-digits effect, but it does reproduce the qualitative Fashion-MNIST pattern: path information is most useful at the severe sparsity cliff. The effect is small and exploratory because this is only two seeds and an MLP, but it argues against the Fashion-MNIST correction being pure noise.

## Fashion-MNIST CNN boundary test

Command: `python experiments/04_criticality_pruning/torch_fashion_mnist_cnn_path_flow.py`

Setup:

- Dataset: Fashion-MNIST.
- Model: small Torch CNN with two conv layers and two FC layers.
- Seeds: `2`.
- Sparsities: `90%, 95%, 98%`.
- Alpha sweep: `0` is magnitude, positive alpha applies channel/path activation correction.

Result:

| Sparsity | Magnitude | Best path correction | Verdict |
|---:|---:|---:|---|
| `90%` | `74.29%` | `72.65%` (`alpha=0.1`) | Path hurts |
| `95%` | `60.54%` | `56.19%` (`alpha=0.1`) | Path hurts |
| `98%` | `44.19%` | `43.33%` (`alpha=0.1`) | Path hurts |

Strong path correction (`alpha >= 0.25`) collapses to near-random accuracy.

Interpretation:

The current path-flow formulation does **not** transfer to CNN pruning. The severe-sparsity correction appears valid for MLP weight matrices but not for naive conv-channel scoring. Convolutional path-flow needs a different derivation, likely channel/filter-level rather than individual-weight global scoring.

## CNN dense-hybrid recovery

Command: `python experiments/04_criticality_pruning/torch_fashion_mnist_cnn_dense_hybrid.py`

The naive CNN path-flow test failed because applying the same path statistic to convolutional filters was destructive. A hybrid test isolates the dense-path part:

- Conv layers: magnitude pruning.
- Dense layers: corrected no-balance path-flow blend.
- Dataset: Fashion-MNIST.
- Model: small CNN.
- Seeds: `2`.

Results versus pure magnitude:

| Sparsity | Magnitude | Best dense-hybrid path correction | Delta |
|---:|---:|---:|---:|
| `90%` | `74.64%` | `75.00%` (`alpha=0.05`) | `+0.36` points |
| `95%` | `60.81%` | `62.98%` (`alpha=0.20`) | `+2.16` points |
| `98%` | `44.38%` | `49.18%` (`alpha=0.20`) | `+4.80` points |

Interpretation:

This recovers the severe-sparsity pattern for CNNs, but only when path-flow is applied to dense bottleneck layers and convolutional layers remain magnitude-pruned. The path principle appears to apply to dense path matrices; convolutional filters need a separate channel/filter-level method.

## CNN dense-hybrid fine-tuning boundary

Command: `python experiments/04_criticality_pruning/torch_fashion_mnist_cnn_dense_hybrid_finetune.py`

Setup:

- Fashion-MNIST CNN.
- Conv layers: magnitude pruning.
- Dense layers: magnitude vs dense-hybrid path-flow (`alpha=0.20`).
- Seeds: `2`.
- Masked fine-tuning after pruning.

| Sparsity | Method | Before FT | After FT | Interpretation |
|---:|---|---:|---:|---|
| `95%` | Magnitude | `62.41%` | `84.69%` | Strong recovery |
| `95%` | Dense-hybrid | `63.54%` | `85.00%` | Small positive edge remains |
| `98%` | Magnitude | `43.91%` | `80.70%` | Recovers better |
| `98%` | Dense-hybrid | `49.59%` | `79.55%` | Better one-shot, worse after recovery |

Interpretation:

Dense-hybrid path-flow is useful as a one-shot severe-sparsity mask initializer in CNNs, but after enough masked recovery training, magnitude can catch up or surpass it at `98%`. This narrows the CNN claim: path-flow helps initial mask quality, especially when recovery budget is limited.

## Sparsity-cliff path correction law

Command: `python experiments/04_criticality_pruning/meta_sparsity_cliff.py`

This meta-analysis combines the image-model transfer checks:

- Fashion-MNIST MLP corrected blend.
- CIFAR-10 MLP corrected blend.
- Fashion-MNIST CNN dense-hybrid corrected blend.

Result:

| Sparsity | Mean best alpha | Mean delta vs magnitude | Positive studies / N |
|---:|---:|---:|---:|
| `90%` | `0.10` | `+0.19` points | `2 / 3` |
| `95%` | `0.28` | `+2.21` points | `3 / 3` |
| `98%` | `0.48` | `+4.62` points | `3 / 3` |

Interpretation:

The path correction is a sparsity-cliff intervention. At moderate sparsity, magnitude should dominate. As sparsity becomes severe, the optimal correction strength increases and the average gain over magnitude grows. The emerging rule is:

```text
score = |weight| * path_signal^alpha(sparsity)
alpha ~= 0 at moderate sparsity
alpha increases as the model approaches the pruning cliff
```

This is the best current generalization of the result. It explains why naive path-flow looked too aggressive on Fashion-MNIST and CNNs: the path term should be weak and sparsity-dependent, not applied as a full replacement for magnitude.

## Adaptive image rule

Command: `python experiments/04_criticality_pruning/adaptive_path_rule_eval.py`

The current practical rule was evaluated across completed image-model runs:

- Fashion-MNIST MLP.
- CIFAR-10 MLP.
- Fashion-MNIST CNN dense-hybrid.

Rule:

- Use magnitude at `90%` sparsity.
- Use weak path correction at `95%` sparsity.
- Use stronger path correction at `98%` sparsity.
- For CNNs, apply path correction only to dense layers; keep conv layers magnitude-pruned.

Aggregate result versus pure magnitude:

- Mean delta: `+1.71` points.
- Wins: `9`.
- Ties: `6`.
- Losses: `3`.
- Paired cases: `18`.

By study:

| Study | Mean delta | Wins | Ties | Losses | N |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 MLP | `+0.75` | `4` | `2` | `0` | `6` |
| Fashion-MNIST CNN dense-hybrid | `+2.32` | `3` | `2` | `1` | `6` |
| Fashion-MNIST MLP | `+2.06` | `2` | `2` | `2` | `6` |

Interpretation:

This is the most general version of the result so far. It is not huge, but it is consistent enough to matter: a sparsity-adaptive path correction beats or ties magnitude in most paired image-model cases and has no losses on CIFAR-10 MLP in the current two-seed run.

## SynFlow baseline: Fashion-MNIST MLP

Command: `python experiments/04_criticality_pruning/fashion_mnist_mlp_synflow.py`

Setup:

- Fashion-MNIST MLP.
- Seeds: `2`.
- Sparsities: `90%, 95%, 98%`.
- Methods: magnitude, gradient saliency, SynFlow, adaptive path rule.

Results:

| Sparsity | Magnitude | SynFlow | Adaptive path | Adaptive - SynFlow |
|---:|---:|---:|---:|---:|
| `90%` | `80.32%` | `79.05%` | `80.32%` | `+1.27` points |
| `95%` | `68.60%` | `68.68%` | `71.33%` | `+2.65` points |
| `98%` | `53.53%` | `44.25%` | `55.30%` | `+11.05` points |

Mean adaptive path delta:

- Versus SynFlow: `+4.99` points.
- Versus magnitude: `+1.50` points.

Interpretation:

This is a stronger label-free result. Adaptive path is not merely beating magnitude; on this Fashion-MNIST MLP benchmark it beats SynFlow, a known label-free pruning baseline, especially at the severe sparsity cliff.

## SynFlow baseline: Fashion-MNIST CNN

Command: `python experiments/04_criticality_pruning/fashion_mnist_cnn_synflow.py`

Setup:

- Fashion-MNIST CNN.
- Seeds: `2`.
- Sparsities: `90%, 95%, 98%`.
- Methods: magnitude, SynFlow, adaptive dense-hybrid path rule.

Results:

| Sparsity | Magnitude | SynFlow | Adaptive dense-hybrid | Takeaway |
|---:|---:|---:|---:|---|
| `90%` | `74.44%` | `76.50%` | `74.44%` | SynFlow best |
| `95%` | `61.23%` | `64.38%` | `63.39%` | SynFlow slightly best |
| `98%` | `43.51%` | `10.08%` | `49.46%` | SynFlow collapses; adaptive path best |

Mean deltas across these sparsities:

- Adaptive dense-hybrid vs SynFlow: `+12.11` points.
- Adaptive dense-hybrid vs magnitude: `+2.70` points.

Interpretation:

SynFlow is strong at moderate CNN sparsity, but catastrophically fails at `98%` in this setup. The adaptive dense-hybrid path rule is valuable as a severe-sparsity guardrail: it avoids the SynFlow cliff and beats magnitude at the hardest pruning point.

## Label-free baseline synthesis

Command: `python experiments/04_criticality_pruning/synthesize_label_free_baselines.py`

This combines the Fashion-MNIST MLP and CNN comparisons against SynFlow and magnitude.

Claim:

Adaptive path correction is most valuable at the severe sparsity cliff. It is not always best at `90-95%`, but at `98%` it beats SynFlow in every paired case currently tested and avoids SynFlow collapse.

Overall across Fashion-MNIST MLP/CNN, `90/95/98%`, two seeds each:

- Adaptive path vs SynFlow: mean `+8.55` points, wins `9/12`.
- Adaptive path vs magnitude: mean `+2.10` points, wins/ties/losses `4/4/4`.

At `98%` sparsity only:

- Adaptive path vs SynFlow: mean `+25.22` points, wins `4/4`.
- Adaptive path vs magnitude: mean `+3.86` points, wins `2/4`.

Interpretation:

The strongest baseline-backed claim is not simply "adaptive path beats magnitude." Magnitude remains very hard to beat. The stronger and more interesting claim is: **adaptive path correction is a label-free guardrail against severe-sparsity collapse, especially compared with SynFlow.**

## Reproducible claim card

Command: `python experiments/04_criticality_pruning/claim_card.py`

A compact claim card now records the strongest result, reproduction commands, and caveats.

Claim grade: `strong severe-sparsity guardrail evidence`.

Primary result:

- Overall adaptive path vs SynFlow mean delta: `+8.55` points, wins `9/12`.
- At `98%` sparsity, adaptive path vs SynFlow mean delta: `+25.22` points, wins `4/4`.
- At `98%` sparsity, adaptive path vs magnitude mean delta: `+3.86` points, wins `2/4`.

Audit command: `python experiments/04_criticality_pruning/audit_claim_card.py`

Audit result: all checks passed. The claim card matches `label_free_baseline_synthesis.json` and all source artifacts exist.

Artifacts:

- `experiments/04_criticality_pruning/CLAIM_CARD.md`
- `results/04_criticality_pruning/CLAIM_CARD.json`
- `results/04_criticality_pruning/claim_card_audit.json`

## Reusable implementation update

Added reusable method kernel:

- `shared/adaptive_path_pruning.py`

Added runnable API example:

- `python experiments/04_criticality_pruning/example_adaptive_path_pruning.py`

The example writes `results/04_criticality_pruning/adaptive_path_pruning_example.json` and verifies equal mask budgets for magnitude and adaptive path scoring.

## 98% SynFlow pressure test and mechanism update

A follow-up pressure test compared magnitude, SynFlow, and adaptive path masks at exactly `98%` sparsity on Fashion-MNIST MLP and CNN, before and after masked fine-tuning. The current evidence narrows the claim:

- `adaptive_path` remains much stronger than SynFlow after recovery on average: mean adaptive minus SynFlow after fine-tuning is `+35.22` points across the MLP/CNN pressure test.
- `magnitude` is still the better fine-tuning recovery baseline on average: mean adaptive minus magnitude after fine-tuning is `-1.04` points.
- The CNN mechanism is concrete: global SynFlow keeps `0.0%` of `fc1`, killing all `128/128` dense bridge units and staying at chance (`10.28%`) after fine-tuning.
- A layerwise SynFlow rescue confirms this is not only a global-allocation bug. At `98%`, layerwise SynFlow restores `2.0%` of `fc1` and reaches `50.21%` after fine-tuning, but still trails magnitude (`83.41%`) and adaptive dense-hybrid (`82.44%`).

Updated interpretation: adaptive path correction is a real severe-sparsity guardrail and one-shot preservation method. It is not yet the best post-fine-tuning initializer; magnitude is still the baseline to beat there.

Artifacts:

- `results/04_criticality_pruning/synflow_finetune_98pct_comparison.json`
- `results/04_criticality_pruning/synflow_cnn_mask_forensics_98pct.json`
- `results/04_criticality_pruning/synflow_cnn_layerwise_rescue_98pct.json`
- `experiments/04_criticality_pruning/SYNFLOW_CNN_MASK_FORENSICS_98PCT.md`
- `experiments/04_criticality_pruning/SYNFLOW_CNN_LAYERWISE_RESCUE_98PCT.md`

## Packaging update: low-alpha adaptive path rule

The CNN `98%` alpha sweep turned the result into a more usable rule. The old aggressive cliff schedule is not the default anymore. The reusable implementation now separates objectives:

- `balanced`: default low-alpha rule; on the CNN `98%` sweep, `alpha=0.08` improved one-shot accuracy from `44.55%` to `49.38%` and after-FT accuracy from `83.34%` to `83.65%`.
- `recovery`: weaker rule; `alpha=0.05` gave the best after-FT result at `83.94%`.
- `one_shot`: stronger rule; `alpha=0.15` gave the best one-shot result at `50.01%`, but after-FT dropped to `82.93%`.

This updates the practical claim: the real win is not high path weighting; it is a small path-aware correction at the sparsity cliff.

## CIFAR-10 transfer update

The low-alpha severe-sparsity rule transferred to a real CIFAR-10 CNN, but only in the very weak correction range.

CIFAR-10 setup: real images, 20k train subset, 5k test subset, two seeds, small CNN, `98%` sparsity.

| Alpha | One-shot | After masked FT | Result |
|---:|---:|---:|---|
| `0.00` | `16.65%` | `45.28%` | magnitude baseline |
| `0.03` | `18.14%` | `45.79%` | best one-shot / balanced |
| `0.05` | `16.50%` | `46.40%` | best recovery |
| `0.08` | `15.45%` | `43.35%` | too high for CIFAR |
| `0.15` | `12.50%` | `24.24%` | collapse |

This changes the packaged rule: use `alpha=0.03-0.05` near the severe sparsity cliff unless a model-specific sweep proves a stronger setting. The reusable default is now `0.05`.

See `experiments/04_criticality_pruning/LOW_ALPHA_TRANSFER_SYNTHESIS.md`.

## CIFAR cliff sweep nuance

A follow-up CIFAR sweep over `95/98/99%` sparsity showed the low-alpha result is not uniformly stable:

- `95%`: low-alpha improves one-shot by `+2.71` points and after-FT by `+0.50` points.
- `98%`: magnitude wins one-shot, but `alpha=0.05` gives a tiny after-FT edge of `+0.33` points.
- `99%`: `alpha=0.05` improves one-shot by `+1.46` points but hurts after-FT.

This means the transfer claim is promising but not decisive; it needs a larger paired GPU replicate before being framed as robust.

## Six-seed GPU replicate conclusion

The CIFAR-10 low-alpha claim is now narrowed by stronger evidence:

- `alpha=0.03` improves one-shot accuracy at `95%` by `+1.49` points (`5/6` seeds) and at `98%` by `+1.13` points (`4/6` seeds).
- Low-alpha does not improve after masked fine-tuning on CIFAR; magnitude remains the safest recovery initializer.
- At `99%`, the correction is unstable and can hurt recovery badly.

The reusable default has been tightened: `balanced` and `one_shot` use `alpha=0.03`; `recovery` uses `alpha=0.0` unless a domain-specific sweep says otherwise.

## Bridge-floor test result

A direct structural repair eliminated dead `fc1` hidden units but did not improve CIFAR accuracy. At `98%`, `bridge_floor1` was only `+0.10` after-FT points vs magnitude and `-0.18` one-shot points. At `99%`, it was only `+0.07` / `+0.07`.

This rules out a simplistic explanation: the problem is not just dead hidden units. The mask must preserve useful bridge weights, not merely make every hidden unit technically live.

## CIFAR-10 SynFlow pathology replication

The global SynFlow allocation failure replicated on CIFAR-10 CNN.

Setup: real CIFAR-10 images, 20k train subset, 5k test subset, four seeds, CUDA on RTX 5090 Laptop GPU, `98%` and `99%` global sparsity.

Key result:

| Sparsity | Method | Before FT | After FT | fc1 keep rate | Dead fc1 hidden | After-FT delta vs magnitude |
|---:|---|---:|---:|---:|---:|---:|
| `98%` | magnitude | `14.37%` | `44.08%` | `0.0081` | `81.8` | baseline |
| `98%` | global SynFlow | `9.76%` | `9.76%` | `0.0000` | `192.0` | `-34.32` pts |
| `98%` | layerwise SynFlow | `9.53%` | `23.75%` | `0.0200` | `84.3` | `-20.33` pts |
| `99%` | magnitude | `11.96%` | `33.24%` | `0.0035` | `86.3` | baseline |
| `99%` | global SynFlow | `9.76%` | `9.76%` | `0.0000` | `192.0` | `-23.48` pts |
| `99%` | layerwise SynFlow | `10.01%` | `17.58%` | `0.0100` | `89.3` | `-15.66` pts |

Interpretation: this is now the strongest finding in the repo. Global SynFlow catastrophically over-allocates away from the dense bridge in CNNs with dense tails. The failure transfers from Fashion-MNIST CNN to CIFAR-10 CNN and is not fixed enough by layerwise allocation.

Artifact: `results/04_criticality_pruning/cifar10_cnn_synflow_pathology.json`.

## Cross-dataset SynFlow synthesis

A synthesis artifact now aggregates the Fashion-MNIST CNN and CIFAR-10 CNN SynFlow pathology results:

- Cases: `3` severe-sparsity CNN settings.
- Global SynFlow zero-`fc1` cases: `3/3`.
- Mean global SynFlow after-FT delta vs magnitude: `-42.80` points.
- Mean layerwise SynFlow after-FT delta vs magnitude: `-22.21` points.

See `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md` and `results/04_criticality_pruning/synflow_pathology_synthesis.json`.

A reusable guardrail module was added at `shared/pruning_diagnostics.py` so future pruning methods can report dense-bridge collapse before fine-tuning.

## Multi-cut capacity pruning: first constructive result

A new multi-cut capacity experiment turns the SynFlow pathology into a constructive method prototype.

Setup: CIFAR-10 CNN, 20k train subset, 5k test subset, CUDA, four seeds, `98%` and `99%` sparsity.

Key result:

- At `98%`, global SynFlow stays at chance (`9.76%` after FT); multi-cut capacity reaches `40.09%`, though magnitude remains higher at `44.05%`.
- At `99%`, global SynFlow stays at chance (`9.76%` after FT); multi-cut capacity reaches `33.41%`, slightly above magnitude at `32.62%`.
- Multi-cut capacity eliminates dense bridge death (`0.0` dead `fc1` units) while preserving the same global parameter budget.

Interpretation: this is the first result supporting the constructive path-capacity thesis. The method is not mature, but preserving capacity across multiple cuts can convert an unrecoverable sparse circuit into a trainable one, and can beat magnitude at the harsher `99%` cliff in this run.

See `experiments/04_criticality_pruning/CIFAR10_CNN_MULTICUT_CAPACITY_PRUNING.md`.

## Adaptive capacity pruning update

A less hand-tuned Path-Capacity method now exists. It predicts global SynFlow cutsets from score distributions, reserves `55%` of the keep budget across critical cuts by output-unit count, and fills the rest by SynFlow saliency.

CIFAR-10 CNN result, four seeds:

- `98%`: adaptive capacity recovers SynFlow from `9.76%` to `41.70%` after FT, but trails magnitude at `44.11%`.
- `99%`: adaptive capacity recovers SynFlow from `9.76%` to `34.65%` after FT, beating magnitude at `32.83%` on mean.
- In both cases, adaptive capacity eliminates dense-bridge death: `0.0` dead `fc1` units.

This is now the strongest constructive result because it is less hand-set than the earlier multi-cut floor method and directly implements the circuit-viability thesis.

See `experiments/04_criticality_pruning/CIFAR10_CNN_ADAPTIVE_CAPACITY_PRUNING.md`.

## Risk-adaptive allocator failed to beat fixed reserve

A predictor-only allocator based on dead-output risk was tested. It prevented `fc1` death but underperformed the fixed `0.55` reserve:

- `98%`: risk-adaptive after FT `39.49%`, fixed capacity `41.55%`, magnitude `43.70%`.
- `99%`: risk-adaptive after FT `32.22%`, fixed capacity `33.17%`, magnitude `32.61%`.

This sharpens the next step: a real cut-capacity predictor must include score mass concentration and route quality, not just dead-output counts.
