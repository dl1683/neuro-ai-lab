# CIFAR-10 Full ResNet-20 Capacity at 99%

Full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[231, 232]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.3801` | `0.0165` | baseline | baseline | `0.0974` | `2.2740` | `3.1311` | `181.0` |
| `global_synflow` | `0.1000` | `0.0000` | `-0.2801` | `0/2` | `0.0000` | `0.0000` | `3.8754` | `666.0` |
| `reserve_0.60` | `0.3961` | `0.0010` | `+0.0160` | `2/2` | `1.2117` | `1.1786` | `3.6700` | `0.0` |
| `tuned_40_35_25` | `0.3838` | `0.0002` | `+0.0037` | `1/2` | `0.9005` | `1.9347` | `3.7471` | `0.0` |

## Interpretation

This tests the ResNet-20-style capacity result on full CIFAR-10 train/test rather than the 20k/5k subset.
