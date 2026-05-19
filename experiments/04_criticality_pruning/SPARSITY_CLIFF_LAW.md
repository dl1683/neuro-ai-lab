# Sparsity-Cliff Path Correction

This meta-analysis extracts the best path-correction strength (`alpha`) from the completed image-model transfer checks.

## Claim

Across the current MLP and CNN transfer checks, path correction should be treated as a sparsity-cliff intervention: keep alpha near zero at moderate sparsity and increase it only when magnitude pruning begins to collapse.

## Aggregate by sparsity

| Sparsity | Mean best alpha | Mean delta vs magnitude | Positive studies | N |
|---:|---:|---:|---:|---:|
| `0.90` | `0.100` | `0.0019` | `2` | `3` |
| `0.95` | `0.283` | `0.0221` | `3` | `3` |
| `0.98` | `0.483` | `0.0462` | `3` | `3` |

## Study-level best alpha

| Study | Sparsity | Best alpha | Best accuracy | Magnitude accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST MLP corrected blend | `0.90` | `0.00` | `0.8032` | `0.8032` | `0.0000` |
| Fashion-MNIST MLP corrected blend | `0.95` | `0.40` | `0.7280` | `0.6860` | `0.0420` |
| Fashion-MNIST MLP corrected blend | `0.98` | `0.25` | `0.6055` | `0.5353` | `0.0702` |
| CIFAR-10 MLP corrected blend | `0.90` | `0.25` | `0.4123` | `0.4101` | `0.0022` |
| CIFAR-10 MLP corrected blend | `0.95` | `0.25` | `0.3488` | `0.3461` | `0.0027` |
| CIFAR-10 MLP corrected blend | `0.98` | `1.00` | `0.2839` | `0.2634` | `0.0205` |
| Fashion-MNIST CNN dense-hybrid | `0.90` | `0.05` | `0.7500` | `0.7464` | `0.0036` |
| Fashion-MNIST CNN dense-hybrid | `0.95` | `0.20` | `0.6298` | `0.6081` | `0.0216` |
| Fashion-MNIST CNN dense-hybrid | `0.98` | `0.20` | `0.4918` | `0.4437` | `0.0480` |

## Interpretation

The useful path signal is not a replacement for magnitude at ordinary sparsity. It becomes useful as the model approaches the compression cliff, especially at `95-98%` sparsity. The practical rule is to keep magnitude dominant and introduce a weak path correction only in severe pruning regimes.
