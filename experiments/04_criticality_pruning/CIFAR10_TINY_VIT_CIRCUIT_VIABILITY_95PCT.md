# CIFAR-10 TinyViT Circuit Viability at 95%

TinyViT CIFAR-10 subset transformer-analogue pruning test at 95% sparsity. MLP down-projection and classifier readout rows are treated as circuit bottlenecks.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[288, 289]`

| Method | After FT | Delta vs magnitude | Wins | Dead outputs | MLP-down dead | MLP-down min | Head min |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0987` |  |  | `1247.0` | `512.0` | `0.0` | `0.0` |
| `global_synflow` | `0.1078` | `+0.0091` | `2/2` | `2253.0` | `232.0` | `0.0` | `87.0` |
| `minimal_liveness_repair` | `0.1087` | `+0.0100` | `2/2` | `72.5` | `0.0` | `1.0` | `0.5` |
| `selective_mlp_readout_repair` | `0.1023` | `+0.0036` | `1/2` | `760.5` | `0.0` | `1.0` | `1.0` |
| `attn_mlp_readout_repair` | `0.1031` | `+0.0044` | `2/2` | `467.0` | `0.0` | `1.0` | `0.5` |
| `all_route_liveness_floor` | `0.1036` | `+0.0049` | `1/2` | `0.0` | `0.0` | `1.0` | `1.0` |
| `mlp_readout_reserve` | `0.1022` | `+0.0035` | `1/2` | `1578.5` | `0.0` | `38.0` | `19.0` |

## Interpretation

This lowers TinyViT sparsity from `98%` to `95%` to test whether the transformer circuit-viability interventions matter before recovery reaches the chance floor. It uses fresh seeds and the same candidate mask families as the `98%` run.
