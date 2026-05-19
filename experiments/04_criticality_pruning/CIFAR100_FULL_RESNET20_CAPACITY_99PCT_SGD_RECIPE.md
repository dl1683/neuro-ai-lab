# CIFAR-100 Full ResNet-20 Capacity at 99%: SGD Recipe

Full CIFAR-100 train/test ResNet-20-style path-capacity pruning at 99% sparsity with 20 dense SGD/cosine epochs and 5 masked fine-tune epochs.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[251, 252]`
Dense epochs: `20`; masked fine-tune epochs: `5`

| Method | Dense | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.5953` | `0.0658` | `0.0009` |  |  | `0.0000` | `0.5174` | `3.0645` | `549.5` |
| `global_synflow` | `0.5953` | `0.0100` | `0.0000` | `-0.0558` | `0/2` | `0.0000` | `0.0000` | `0.0657` | `700.5` |
| `reserve_0.60` | `0.5953` | `0.0764` | `0.0056` | `+0.0106` | `2/2` | `1.1880` | `1.1341` | `1.0902` | `0.5` |

## Interpretation

This is the first 100-class full-dataset residual stress test. It asks whether the homeostatic circuit-viability result survives a harder output space rather than only CIFAR-10.
