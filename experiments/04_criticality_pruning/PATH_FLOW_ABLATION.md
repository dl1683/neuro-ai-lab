# Path-Flow Ablation

This ablation tests which terms in the label-free path-flow pruning score matter at `90%` and `95%` sparsity across five trained sklearn-digits MLP seeds.

## Mean accuracy by variant

| Variant | Mean accuracy | Std | Mean retention | Mean hidden coverage |
|---|---:|---:|---:|---:|
| `magnitude` | `0.6146` | `0.1214` | `0.6410` | `0.6531` |
| `gradient_saliency` | `0.7814` | `0.0804` | `0.8151` | `0.7828` |
| `path_flow_full` | `0.7890` | `0.0752` | `0.8230` | `0.5273` |
| `no_input_activity` | `0.7105` | `0.0856` | `0.7410` | `0.4266` |
| `no_hidden_strength` | `0.7760` | `0.0943` | `0.8093` | `0.6164` |
| `no_hidden_balance` | `0.7847` | `0.0891` | `0.8185` | `0.5328` |
| `no_output_strength` | `0.7797` | `0.0811` | `0.8133` | `0.6211` |

## Drops from full path-flow

| Removed ingredient | Accuracy drop |
|---|---:|
| `no_input_activity` | `0.0785` |
| `no_hidden_strength` | `0.0130` |
| `no_hidden_balance` | `0.0043` |
| `no_output_strength` | `0.0094` |

Best variant: `path_flow_full`.

Interpretation: if a removal improves accuracy, that ingredient is not helping in this benchmark. If removal hurts, it is carrying useful signal.
