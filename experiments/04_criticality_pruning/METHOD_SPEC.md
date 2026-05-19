# Adaptive Path Pruning Method Spec

This is the reusable method distilled from experiment 04.

## Claim

Adaptive path pruning is a label-free severe-sparsity guardrail. It is not a universal replacement for magnitude pruning. It is most useful near the pruning cliff, especially against SynFlow-style collapse.

## Core score

For dense path matrices `input -> hidden -> output`:

```text
hidden_path[h] = hidden_strength[h] * output_strength[h]
fc1_score[h, i] = |fc1[h, i]| * (input_signal[i] * hidden_path[h])^alpha
fc2_score[o, h] = |fc2[o, h]| * (hidden_path[h])^alpha
```

Where:

- `input_signal`: per-feature standard deviation on an unlabeled calibration batch.
- `hidden_strength`: mean absolute hidden activation on the same calibration batch.
- `output_strength`: mean absolute outgoing weight strength from the hidden unit.
- `alpha`: sparsity-dependent correction strength.

## Current alpha schedule

```text
sparsity < 92.5%      -> alpha = 0.00
92.5% <= sparsity < 97.5% -> alpha = 0.25
sparsity >= 97.5%     -> alpha = 0.50
```

This schedule is conservative and empirical. It comes from Fashion-MNIST MLP, CIFAR-10 MLP, and Fashion-MNIST CNN dense-tail checks.

## CNN rule

Do not apply the dense path score directly to convolutional filters. Current evidence says naive conv path-flow hurts.

Use:

- Convolutional layers: magnitude scores.
- Dense bottleneck/tail layers: adaptive path scores.

## Reusable implementation

See `shared/adaptive_path_pruning.py`.

Important functions:

- `adaptive_alpha(sparsity)`
- `dense_calibration_stats(flat_inputs, hidden_activations)`
- `dense_path_scores(fc1_weight, fc2_weight, input_signal, hidden_strength, config)`
- `score_dense_tail_with_magnitude_convs(...)`
- `global_topk_mask(scores, sparsity)`

## Evidence anchors

- `experiments/04_criticality_pruning/CLAIM_CARD.md`
- `experiments/04_criticality_pruning/LABEL_FREE_BASELINE_SYNTHESIS.md`
- `experiments/04_criticality_pruning/EVIDENCE_SYNTHESIS.md`
- `results/04_criticality_pruning/CLAIM_CARD.json`

## Minimal runnable example

Command:

```bash
python experiments/04_criticality_pruning/example_adaptive_path_pruning.py
```

This example does not claim an accuracy result. It only demonstrates the reusable scoring/masking API on a synthetic dense path and writes:

- `results/04_criticality_pruning/adaptive_path_pruning_example.json`

Use it to verify that the shared API is importable and that path and magnitude masks keep the same global parameter budget.

## Alpha sweep update: weak path correction is the usable rule

A focused `98%` Fashion-MNIST CNN alpha sweep found that aggressive path weighting is not the right packaged default. The practical rule is weak path modulation:

| Alpha | Before FT | After FT | Interpretation |
|---:|---:|---:|---|
| `0.00` | `44.55%` | `83.34%` | magnitude baseline |
| `0.05` | `45.58%` | `83.94%` | best fine-tuning recovery |
| `0.08` | `49.38%` | `83.65%` | best balanced setting |
| `0.15` | `50.01%` | `82.93%` | best one-shot, slight recovery cost |
| `0.50` | `27.74%` | `80.34%` | too aggressive |

The reusable kernel now exposes this distinction through `PathPruningConfig.objective`:

- `balanced`: default, low-alpha severe-sparsity guardrail.
- `recovery`: weakest path correction for fine-tuning-first workflows.
- `one_shot`: stronger path correction when no recovery training is planned.

Artifact: `results/04_criticality_pruning/cnn_98pct_adaptive_alpha_ft_sweep.json`.

## Transfer correction: default alpha is now `0.05`

CIFAR-10 CNN transfer showed that `alpha=0.08` is already too strong outside Fashion-MNIST. The reusable severe-sparsity default is now `0.05`, with the practical recommendation to sweep `0.03-0.05` when possible.

See `experiments/04_criticality_pruning/LOW_ALPHA_TRANSFER_SYNTHESIS.md`.

## Evidence-tightened default

The six-seed CIFAR GPU replicate changed the default rule:

- `balanced` / `one_shot`: `alpha=0.03` near the severe sparsity cliff.
- `recovery`: `alpha=0.0`, because CIFAR fine-tuning recovery favored magnitude.

This is encoded in `shared/adaptive_path_pruning.py`.

## SynFlow pathology transfer update

The CIFAR-10 CNN SynFlow run confirms that the global SynFlow failure is not Fashion-MNIST-specific. At `98/99%`, global SynFlow allocates zero weights to `fc1`, killing every dense bridge hidden unit and remaining at chance after masked fine-tuning. Layerwise SynFlow partially rescues bridge allocation but remains far below magnitude.

This strengthens the guardrail recommendation: always inspect per-layer keep rates and dense-bridge liveness before trusting global SynFlow or any global saliency method at severe sparsity.

## Packaged guardrail diagnostic

The reusable diagnostics module `shared/pruning_diagnostics.py` now exposes:

- `summarize_layer_mask`: per-layer keep rate and dead output units.
- `summarize_masks`: summaries for a full mask dictionary.
- `summarize_dense_bridge`: dense bridge keep rate, dead hidden units, min/max fan-in, and collapse status.
- `bridge_collapse_report`: compact guardrail report with a `flagged` boolean.

The intended use is to reject or warn on severe global pruning masks before fine-tuning if a dense classifier bridge is fully collapsed or has excessive dead hidden units.
