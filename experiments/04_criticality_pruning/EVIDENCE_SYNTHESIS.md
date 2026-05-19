# Path-Flow Evidence Synthesis

This document consolidates the real experiment evidence for the pruning thread. It intentionally narrows the claim to what survived cross-dataset testing.

## Claim v3

Path-aware pruning is not universally better than magnitude. It is strongest as a label-free severe-sparsity path-preservation correction. On sklearn digits MLPs, full path-flow dominates magnitude and slightly beats gradient saliency. On Fashion-MNIST MLPs, naive path-flow fails, but a no-balance weak path/magnitude blend improves severe-sparsity masks and keeps a small edge after fine-tuning.

## Sklearn digits paired evidence

High-sparsity regime: `90%` and `95%`, five seeds, paired by seed and sparsity.

| Comparison | Mean delta | 95% bootstrap CI | Wins / N | Median delta |
|---|---:|---:|---:|---:|
| `path_flow_vs_magnitude` | `0.1744` | `[0.1339, 0.2202]` | `10 / 10` | `0.1486` |
| `path_flow_vs_gradient_saliency` | `0.0076` | `[-0.0167, 0.0334]` | `6 / 10` | `0.0095` |
| `activation_flow_vs_magnitude` | `0.1650` | `[0.1288, 0.2049]` | `10 / 10` | `0.1455` |

## Extreme sparsity evidence

Extreme regime: `97%`, `98%`, `99%`, five seeds, one-shot pruning.

| Comparison | Mean delta | 95% bootstrap CI | Wins / N | Median delta |
|---|---:|---:|---:|---:|
| `path_flow_vs_magnitude_97_99` | `0.2037` | `[0.1640, 0.2471]` | `15 / 15` | `0.1955` |
| `path_flow_vs_gradient_97_99` | `0.0422` | `[0.0056, 0.0822]` | `8 / 15` | `0.0223` |

## Ingredient ablation

Positive drops mean removing the ingredient hurt full path-flow.

| Removed ingredient | Accuracy drop |
|---|---:|
| `no_input_activity` | `0.0785` |
| `no_hidden_strength` | `0.0130` |
| `no_hidden_balance` | `0.0043` |
| `no_output_strength` | `0.0094` |

## Calibration requirement

Accuracy drop versus full unlabeled calibration on sklearn digits high-sparsity masks.

| Calibration fraction | Accuracy drop |
|---:|---:|
| `0.02` | `0.0447` |
| `0.05` | `0.0296` |
| `0.1` | `0.0153` |
| `0.25` | `0.0076` |

## Fashion-MNIST adaptive rule

use alpha=0 magnitude at 90% sparsity; use corrected no-balance path blend alpha=0.25 at 95% and 98% sparsity

Paired delta vs magnitude across two seeds and three sparsities: mean `0.0325`, 95% bootstrap CI `[0.0037, 0.0705]`, wins `3 / 6`.

Rows:

| Seed | Sparsity | Chosen alpha | Adaptive accuracy | Magnitude accuracy | Delta |
|---:|---:|---:|---:|---:|---:|
| `31` | `0.90` | `0.00` | `0.7817` | `0.7817` | `0.0000` |
| `31` | `0.95` | `0.25` | `0.7073` | `0.6520` | `0.0553` |
| `31` | `0.98` | `0.25` | `0.5947` | `0.4780` | `0.1167` |
| `32` | `0.90` | `0.00` | `0.8247` | `0.8247` | `0.0000` |
| `32` | `0.95` | `0.25` | `0.7193` | `0.7200` | `-0.0007` |
| `32` | `0.98` | `0.25` | `0.6163` | `0.5927` | `0.0237` |

## Bottom line

The exceptional part is not the original criticality story. It is the severe-sparsity path-preservation principle. The strongest evidence is on sklearn digits; Fashion-MNIST forces a narrower rule: keep magnitude dominant at moderate sparsity, add a weak no-balance path correction only near the sparsity cliff.
