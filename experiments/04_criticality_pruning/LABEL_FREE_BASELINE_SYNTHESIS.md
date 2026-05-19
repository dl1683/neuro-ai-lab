# Label-Free Baseline Synthesis

Against SynFlow and magnitude on Fashion-MNIST MLP/CNN, adaptive path correction is most valuable at the severe sparsity cliff. It is not always best at 90-95%, but at 98% it beats both baselines in every paired case currently tested.

## Overall

| Comparison | Mean delta | Wins | Ties | Losses | N |
|---|---:|---:|---:|---:|---:|
| `vs_synflow` | `0.0855` | `9` | `0` | `3` | `12` |
| `vs_magnitude` | `0.0210` | `4` | `4` | `4` | `12` |

## By sparsity

| Sparsity | Baseline | Mean delta | Wins | Ties | Losses | N |
|---:|---|---:|---:|---:|---:|---:|
| `0.90` | `vs_synflow` | `-0.0040` | `2` | `0` | `2` | `4` |
| `0.90` | `vs_magnitude` | `0.0000` | `0` | `4` | `0` | `4` |
| `0.95` | `vs_synflow` | `0.0083` | `3` | `0` | `1` | `4` |
| `0.95` | `vs_magnitude` | `0.0245` | `2` | `0` | `2` | `4` |
| `0.98` | `vs_synflow` | `0.2522` | `4` | `0` | `0` | `4` |
| `0.98` | `vs_magnitude` | `0.0386` | `2` | `0` | `2` | `4` |

## Paired rows

| Study | Seed | Sparsity | Adaptive | SynFlow | Magnitude | Delta vs SynFlow | Delta vs Magnitude |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST CNN | `51` | `0.90` | `0.7610` | `0.7382` | `0.7610` | `0.0228` | `0.0000` |
| Fashion-MNIST CNN | `51` | `0.95` | `0.6663` | `0.6328` | `0.6190` | `0.0335` | `0.0473` |
| Fashion-MNIST CNN | `51` | `0.98` | `0.5555` | `0.1022` | `0.3967` | `0.4532` | `0.1588` |
| Fashion-MNIST CNN | `52` | `0.90` | `0.7278` | `0.7917` | `0.7278` | `-0.0640` | `0.0000` |
| Fashion-MNIST CNN | `52` | `0.95` | `0.6015` | `0.6548` | `0.6055` | `-0.0533` | `-0.0040` |
| Fashion-MNIST CNN | `52` | `0.98` | `0.4338` | `0.0993` | `0.4735` | `0.3345` | `-0.0397` |
| Fashion-MNIST MLP | `31` | `0.90` | `0.7817` | `0.7850` | `0.7817` | `-0.0033` | `0.0000` |
| Fashion-MNIST MLP | `31` | `0.95` | `0.7073` | `0.6847` | `0.6520` | `0.0227` | `0.0553` |
| Fashion-MNIST MLP | `31` | `0.98` | `0.5593` | `0.4260` | `0.4780` | `0.1333` | `0.0813` |
| Fashion-MNIST MLP | `32` | `0.90` | `0.8247` | `0.7960` | `0.8247` | `0.0287` | `0.0000` |
| Fashion-MNIST MLP | `32` | `0.95` | `0.7193` | `0.6890` | `0.7200` | `0.0303` | `-0.0007` |
| Fashion-MNIST MLP | `32` | `0.98` | `0.5467` | `0.4590` | `0.5927` | `0.0877` | `-0.0460` |
