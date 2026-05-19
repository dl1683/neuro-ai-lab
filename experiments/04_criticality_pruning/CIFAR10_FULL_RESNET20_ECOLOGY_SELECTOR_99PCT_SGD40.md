# CIFAR-10 Full ResNet-20 Ecology Selector at 99%: SGD-40 Stress Test

Full CIFAR-10 ResNet-20-style 99% sparsity stress test with 40 dense SGD/cosine epochs, 8 masked fine-tune epochs, and the fixed ecology-aware selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[269, 270]`
Dense epochs: `40`; masked fine-tune epochs: `8`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4873` | `0.0027` |  |  | `0.0000` | `0.8079` | `3.7863` | `311.5` |
| `plain_reserve` | `0.5262` | `0.0022` | `+0.0390` | `2/2` | `1.2209` | `1.1834` | `3.2227` | `3.0` |
| `predicted_route_split` | `0.4769` | `0.0016` | `-0.0104` | `0/2` | `1.0173` | `1.6523` | `3.5409` | `2.5` |
| `ecology_policy` | `0.5262` | `0.0022` | `+0.0390` | `2/2` | `1.2209` | `1.1834` | `3.2227` | `3.0` |

## Decisions

- seed `269`: selected `plain_reserve` readout_ratio `0.8418` split `None`
- seed `270`: selected `plain_reserve` readout_ratio `0.8605` split `None`

## Interpretation

This stress test doubles dense training relative to the 20-epoch recipe and keeps the ecology selector threshold fixed. It is not a canonical 160-epoch CIFAR schedule, but it checks whether the selector survives a better-trained dense model without retuning.
