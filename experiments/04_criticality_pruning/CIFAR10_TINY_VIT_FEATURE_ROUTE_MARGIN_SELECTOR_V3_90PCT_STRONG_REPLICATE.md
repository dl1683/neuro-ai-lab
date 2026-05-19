# CIFAR-10 TinyViT Feature-Route Margin Selector V3 at 90%: Strong Replicate

Two-seed fresh replicate of the full-train TinyViT CIFAR-10 90% sparsity V3 feature-route selector. Tests whether the three-way feature/liveness/trainability rule survives new strong TinyViT seeds.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[302, 303]`
Dense accuracy mean: `0.7176`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0806` |  |  | `-0.0214` | `73.0` | `4.0` |
| `global_synflow` | `0.1505` | `+0.0698` | `2/2` | `0.0399` | `94.0` | `82.5` |
| `minimal_liveness_repair` | `0.0815` | `+0.0008` | `1/2` | `-0.0229` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.0818` | `+0.0011` | `1/2` | `-0.0236` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.0814` | `+0.0007` | `1/2` | `-0.0229` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1505` | `+0.0698` | `2/2` | `0.0399` | `94.0` | `82.5` |

## Selector decisions

- seed `302`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0510/dead177, magnitude=-0.0034/dead81, attn_mlp_readout_repair=-0.0053/dead0, minimal_liveness_repair=-0.0060/dead0, all_route_liveness_floor=-0.0060/dead0
- seed `303`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0287/dead176, magnitude=-0.0394/dead73, minimal_liveness_repair=-0.0397/dead0, all_route_liveness_floor=-0.0397/dead0, attn_mlp_readout_repair=-0.0419/dead0

## Interpretation

This replicate tests the selector boundary rather than headline benchmark strength. A win means V3 is a better rule than V2; a miss identifies which feature/liveness/trainability term is still under-modeled.
