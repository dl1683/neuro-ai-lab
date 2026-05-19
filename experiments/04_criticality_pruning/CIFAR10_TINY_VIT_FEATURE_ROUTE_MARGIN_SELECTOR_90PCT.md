# CIFAR-10 TinyViT Feature-Route Margin Selector at 90%

Fresh TinyViT CIFAR-10 subset 90% sparsity prospective feature-route margin selector. This tests the transformer selector before recovery is fully floor-dominated.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[296, 297]`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1032` |  |  | `0.0264` | `503.0` | `5.5` |
| `global_synflow` | `0.1456` | `+0.0424` | `2/2` | `0.0442` | `133.5` | `80.5` |
| `minimal_liveness_repair` | `0.1150` | `+0.0118` | `2/2` | `0.0286` | `0.5` | `1.5` |
| `attn_mlp_readout_repair` | `0.1141` | `+0.0109` | `2/2` | `0.0275` | `0.5` | `1.5` |
| `all_route_liveness_floor` | `0.1152` | `+0.0120` | `2/2` | `0.0290` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1456` | `+0.0424` | `2/2` | `0.0442` | `133.5` | `80.5` |

## Selector decisions

- seed `296`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0435/dead251, all_route_liveness_floor=0.0387/dead0, minimal_liveness_repair=0.0384/dead2, attn_mlp_readout_repair=0.0378/dead2, magnitude=0.0369/dead508
- seed `297`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0450/dead177, all_route_liveness_floor=0.0194/dead0, minimal_liveness_repair=0.0188/dead2, attn_mlp_readout_repair=0.0172/dead2, magnitude=0.0159/dead509

## Interpretation

This is the same feature-route margin policy as the `95%` TinyViT selector, evaluated at `90%` sparsity. The purpose is to test whether representation-preserving circuit viability remains useful when the sparse transformer is not already close to chance.
