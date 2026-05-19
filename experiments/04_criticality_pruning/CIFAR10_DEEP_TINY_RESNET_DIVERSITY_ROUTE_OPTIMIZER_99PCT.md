# CIFAR-10 DeepTinyResNet Diversity Route Optimizer at 99%

CIFAR-10 DeepTinyResNet transfer test for diversity-penalized route-capacity pruning at 99% sparsity.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[219, 220]`

## Chosen splits

| Seed | Main | Projection | Readout | Pre-FT loss |
|---:|---:|---:|---:|---:|
| `219` | `0.30` | `0.50` | `0.20` | `0.2121` |
| `220` | `0.25` | `0.50` | `0.25` | `0.2412` |

## Results

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2640` | `0.0028` | baseline | baseline | `0.0054` | `2.3226` | `3.1780` | `301.5` |
| `global_synflow` | `0.0996` | `0.0014` | `-0.1644` | `0/2` | `0.0000` | `0.0000` | `3.8565` | `477.5` |
| `reserve_0.60` | `0.2954` | `0.0168` | `+0.0314` | `2/2` | `1.1344` | `1.1035` | `3.2902` | `0.5` |
| `tuned_40_35_25` | `0.2988` | `0.0068` | `+0.0348` | `2/2` | `0.8906` | `1.5712` | `3.5633` | `0.5` |
| `diversity_target_optimizer` | `0.2914` | `0.0106` | `+0.0274` | `2/2` | `0.8906` | `1.8621` | `3.2917` | `0.5` |

## Interpretation

This transfers the diversity route optimizer from the original TinyResNet to a deeper residual model with two blocks per stage.
