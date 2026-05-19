# Path-Flow Calibration Sweep

This experiment asks whether label-free path-flow pruning needs the full training set to estimate activation statistics, or whether a tiny unlabeled calibration batch is enough.

Setup: trained sklearn-digits MLPs, `90%` and `95%` sparsity, five seeds.

| Calibration fraction | Mean examples | Mean accuracy | Std | Mean retention | Drop vs full calibration |
|---:|---:|---:|---:|---:|---:|
| `0.02` | `23.0` | `0.7444` | `0.0965` | `0.7764` | `0.0447` |
| `0.05` | `58.0` | `0.7595` | `0.0824` | `0.7921` | `0.0296` |
| `0.10` | `117.0` | `0.7738` | `0.0695` | `0.8070` | `0.0153` |
| `0.25` | `292.0` | `0.7814` | `0.0756` | `0.8150` | `0.0076` |
| `1.00` | `1168.0` | `0.7890` | `0.0752` | `0.8230` | `0.0000` |

Interpretation: small drops mean the pruning score is deployable with only a small unlabeled calibration batch. Large drops mean path-flow depends on stable activation estimates from broad data coverage.
