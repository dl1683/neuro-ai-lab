# CIFAR-10 ResNet-20 Capacity at 99%

CIFAR-10 ResNet-20-style transfer check for path-capacity pruning at 99% sparsity.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[225, 226]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2848` | `0.0110` | baseline | baseline | `0.0012` | `2.3484` | `3.1711` | `339.0` |
| `global_synflow` | `0.0999` | `0.0009` | `-0.1849` | `0/2` | `0.0000` | `0.0000` | `3.9169` | `666.5` |
| `reserve_0.60` | `0.3241` | `0.0207` | `+0.0393` | `2/2` | `1.1930` | `1.1738` | `3.7233` | `0.0` |
| `tuned_40_35_25` | `0.2992` | `0.0156` | `+0.0144` | `2/2` | `0.8715` | `1.9347` | `3.7785` | `0.0` |

## Interpretation

This is a CIFAR ResNet-20-style transfer check for path-capacity pruning at 99% sparsity.
