# Claim Card: Adaptive Path Correction

Claim grade: **strong severe-sparsity guardrail evidence**

## Claim

Against SynFlow and magnitude on Fashion-MNIST MLP/CNN, adaptive path correction is most valuable at the severe sparsity cliff. It is not always best at 90-95%, but at 98% it beats both baselines in every paired case currently tested.

## Primary result

Setting: Fashion-MNIST MLP and CNN, SynFlow/magnitude baselines, 90/95/98% sparsity, two seeds each.

| Metric | Value |
|---|---:|
| Overall adaptive path vs SynFlow mean delta | `0.0855` |
| Overall adaptive path vs SynFlow wins | `9 / 12` |
| 98% sparsity adaptive path vs SynFlow mean delta | `0.2522` |
| 98% sparsity adaptive path vs SynFlow wins | `4 / 4` |
| 98% sparsity adaptive path vs magnitude mean delta | `0.0386` |
| 98% sparsity adaptive path vs magnitude wins | `2 / 4` |

## What this is not

- Not evidence for the original branching-ratio criticality mechanism.
- Not a universal replacement for magnitude pruning.
- Not yet validated on large CNNs, transformers, or production-scale datasets.

## Reproduce

- `python experiments/04_criticality_pruning/fashion_mnist_mlp_synflow.py`
- `python experiments/04_criticality_pruning/fashion_mnist_cnn_synflow.py`
- `python experiments/04_criticality_pruning/synthesize_label_free_baselines.py`

## Source artifacts

- `results\04_criticality_pruning\fashion_mnist_mlp_synflow_comparison.json`
- `results\04_criticality_pruning\fashion_mnist_cnn_synflow_comparison.json`
- `results\04_criticality_pruning\label_free_baseline_synthesis.json`

## Post-claim pressure-test amendment

Newer pressure tests narrow and strengthen the claim:

- Global SynFlow can catastrophically starve the dense bridge in a CNN at `98%` sparsity, keeping `0.0%` of `fc1` and all `128/128` hidden bridge units dead.
- Layerwise SynFlow only partially rescues this failure: `50.21%` after masked fine-tuning, far below magnitude (`83.41%`) and adaptive dense-hybrid (`82.44%`) in that rescue run.
- A separate alpha sweep found the practical adaptive rule: weak path correction, not aggressive path weighting. `alpha=0.08` gave `49.38%` one-shot and `83.65%` after fine-tuning, beating the same-run magnitude baseline on both metrics.

The claim should be read as: adaptive path correction is a severe-sparsity guardrail and weak correction rule, not a blanket replacement for magnitude pruning in all fine-tuning regimes.
