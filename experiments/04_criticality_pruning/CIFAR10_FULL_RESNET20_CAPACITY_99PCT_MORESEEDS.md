# CIFAR-10 Full ResNet-20 Capacity at 99%

Full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[233, 234, 235, 236]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.3757` | `0.0117` | baseline | baseline | `0.0353` | `2.2747` | `3.1685` | `198.8` |
| `global_synflow` | `0.1000` | `0.0000` | `-0.2757` | `0/4` | `0.0000` | `0.0000` | `3.8779` | `664.2` |
| `reserve_0.60` | `0.3902` | `0.0050` | `+0.0145` | `4/4` | `1.1906` | `1.1738` | `3.6628` | `0.0` |
| `tuned_40_35_25` | `0.3790` | `0.0148` | `+0.0033` | `2/4` | `0.9037` | `1.9347` | `3.7448` | `0.0` |

## Interpretation

This tests the ResNet-20-style capacity result on full CIFAR-10 train/test rather than the 20k/5k subset.
