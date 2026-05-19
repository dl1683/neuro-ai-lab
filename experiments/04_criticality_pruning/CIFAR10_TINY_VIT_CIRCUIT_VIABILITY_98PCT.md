# CIFAR-10 TinyViT Circuit Viability at 98%

TinyViT CIFAR-10 subset transformer-analogue severe-pruning test. MLP down-projection and classifier readout rows are treated as circuit bottlenecks.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[286, 287]`

| Method | After FT | Delta vs magnitude | Wins | Dead outputs | MLP-down dead | MLP-down min | Head min |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1007` |  |  | `1956.5` | `512.0` | `0.0` | `0.0` |
| `global_synflow` | `0.1100` | `+0.0093` | `1/2` | `2792.0` | `389.5` | `0.0` | `71.0` |
| `minimal_liveness_repair` | `0.1131` | `+0.0124` | `1/2` | `62.5` | `0.0` | `1.0` | `0.5` |
| `selective_mlp_readout_repair` | `0.0959` | `-0.0048` | `0/2` | `1452.0` | `0.0` | `1.0` | `1.0` |
| `all_route_liveness_floor` | `0.0996` | `-0.0011` | `0/2` | `0.0` | `0.0` | `1.0` | `1.0` |
| `mlp_readout_reserve` | `0.0995` | `-0.0012` | `1/2` | `2181.0` | `0.0` | `15.0` | `8.0` |

## Interpretation

This is the first transformer-style analogue in the repo. The vulnerable routes are TinyViT MLP down-projections and the classifier readout rather than CNN dense bridges. The test is intentionally small, but it is real: dense TinyViT models are trained on CIFAR-10, pruned at `98%`, then fine-tuned under fixed masks.
