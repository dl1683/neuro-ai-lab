# CIFAR-10 Full ResNet-20 Capacity at 99%: SGD Recipe

Full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity with 20 dense SGD/cosine epochs and 5 masked fine-tune epochs.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[241, 242]`
Dense epochs: `20`; masked fine-tune epochs: `5`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4289` | `0.0020` |  |  | `0.0000` | `1.3545` | `3.7341` | `356.0` |
| `global_synflow` | `0.1000` | `0.0000` | `-0.3289` | `0/2` | `0.0000` | `0.0000` | `3.7853` | `646.0` |
| `reserve_0.60` | `0.5137` | `0.0159` | `+0.0848` | `2/2` | `1.2254` | `1.1713` | `3.3119` | `1.0` |

## Interpretation

This is a harder full-CIFAR stress test for the six-seed speed-recipe result. It asks whether the capacity-reserve advantage survives when the dense model is trained longer with an SGD/cosine recipe before pruning.
