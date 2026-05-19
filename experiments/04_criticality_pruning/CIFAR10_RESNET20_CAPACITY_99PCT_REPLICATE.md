# CIFAR-10 ResNet-20 Capacity 99% Replicate

Fresh four-seed replicate for CIFAR-10 ResNet-20-style path-capacity pruning at 99% sparsity.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[227, 228, 229, 230]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2725` | `0.0124` | baseline | baseline | `0.0127` | `2.3839` | `3.1370` | `341.8` |
| `global_synflow` | `0.1003` | `0.0012` | `-0.1722` | `0/4` | `0.0000` | `0.0000` | `3.8955` | `670.5` |
| `reserve_0.60` | `0.3322` | `0.0118` | `+0.0597` | `4/4` | `1.2000` | `1.1738` | `3.7037` | `0.2` |
| `tuned_40_35_25` | `0.3060` | `0.0103` | `+0.0336` | `4/4` | `0.8973` | `1.9347` | `3.7681` | `0.2` |

## Interpretation

This is a fresh four-seed replicate of the ResNet-20-style path-capacity transfer at 99% sparsity.
