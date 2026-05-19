# CIFAR-10 TinyViT Feature-Route Margin Selector at 95%

Fresh TinyViT CIFAR-10 subset 95% sparsity prospective feature-route margin selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[294, 295]`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0895` |  |  | `0.0005` | `511.5` | `308.0` |
| `global_synflow` | `0.1460` | `+0.0565` | `2/2` | `0.0304` | `225.0` | `133.0` |
| `minimal_liveness_repair` | `0.0929` | `+0.0034` | `1/2` | `-0.0056` | `0.0` | `23.0` |
| `attn_mlp_readout_repair` | `0.1117` | `+0.0222` | `2/2` | `-0.0001` | `0.0` | `14.5` |
| `all_route_liveness_floor` | `0.0934` | `+0.0039` | `1/2` | `-0.0003` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1460` | `+0.0565` | `2/2` | `0.0304` | `225.0` | `133.0` |

## Selector decisions

- seed `294`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0271/dead383, magnitude=0.0033/dead796, attn_mlp_readout_repair=-0.0060/dead16, all_route_liveness_floor=-0.0076/dead0, minimal_liveness_repair=-0.0138/dead29
- seed `295`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0338/dead333, all_route_liveness_floor=0.0071/dead0, attn_mlp_readout_repair=0.0059/dead13, minimal_liveness_repair=0.0026/dead17, magnitude=-0.0022/dead843

## Interpretation

This is a fresh prospective selector test. The policy first ranks candidates by pre-finetune centered CLS/residual-stream feature alignment. If the top score is within a small margin of SynFlow and has much higher transformer route death, it selects SynFlow instead.
