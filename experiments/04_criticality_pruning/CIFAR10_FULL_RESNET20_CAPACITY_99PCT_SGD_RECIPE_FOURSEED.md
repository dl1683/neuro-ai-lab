# CIFAR-10 Full ResNet-20 Capacity at 99%: SGD Recipe Four-Seed Aggregate

Four-seed aggregate of full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity using 20 dense SGD/cosine epochs and 5 masked fine-tune epochs.

Sources: `cifar10_full_resnet20_capacity_99pct_sgd_recipe.json`, `cifar10_full_resnet20_capacity_99pct_sgd_recipe_moreseeds.json`
Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[241, 242, 243, 244]`
Dense epochs: `20`; masked fine-tune epochs: `5`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4287` | `0.0087` |  |  | `0.0000` | `1.3876` | `3.7465` | `343.0` |
| `global_synflow` | `0.1000` | `0.0000` | `-0.3287` | `0/4` | `0.0000` | `0.0000` | `3.7943` | `647.2` |
| `reserve_0.60` | `0.4943` | `0.0233` | `+0.0657` | `4/4` | `1.2312` | `1.1701` | `3.3285` | `1.8` |

## Interpretation

The capacity reserve advantage becomes larger under the stronger SGD/cosine recipe than under the short speed recipe. Magnitude leaves the main route at zero capacity and hundreds of dead outputs, while reserve restores a main-path floor and wins every paired seed.
