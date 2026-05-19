# CIFAR-10 Full ResNet-20 Capacity at 99%: SGD Recipe

Full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity with 20 dense SGD/cosine epochs and 5 masked fine-tune epochs.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[243, 244]`
Dense epochs: `20`; masked fine-tune epochs: `5`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4284` | `0.0122` |  |  | `0.0000` | `1.4206` | `3.7589` | `330.0` |
| `global_synflow` | `0.1000` | `0.0000` | `-0.3284` | `0/2` | `0.0000` | `0.0000` | `3.8033` | `648.5` |
| `reserve_0.60` | `0.4749` | `0.0088` | `+0.0466` | `2/2` | `1.2370` | `1.1689` | `3.3452` | `2.5` |

## Interpretation

This is a harder full-CIFAR stress test for the six-seed speed-recipe result. It asks whether the capacity-reserve advantage survives when the dense model is trained longer with an SGD/cosine recipe before pruning.
