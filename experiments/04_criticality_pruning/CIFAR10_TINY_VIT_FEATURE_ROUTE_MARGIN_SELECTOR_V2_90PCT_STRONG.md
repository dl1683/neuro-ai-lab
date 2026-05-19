# CIFAR-10 TinyViT Feature-Route Margin Selector V2 at 90%: Stronger Recipe

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of a V2 feature-route selector. V2 adds a trainable-capacity guardrail: do not choose SynFlow when its feature advantage is small and it creates much more route death than magnitude.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[299]`
Dense accuracy mean: `0.7248`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1218` |  |  | `-0.0054` | `79.0` | `3.0` |
| `global_synflow` | `0.1025` | `-0.0193` | `0/1` | `0.0027` | `101.0` | `75.0` |
| `minimal_liveness_repair` | `0.1179` | `-0.0039` | `0/1` | `-0.0061` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.1200` | `-0.0018` | `0/1` | `-0.0069` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.1193` | `-0.0025` | `0/1` | `-0.0061` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1218` | `+0.0000` | `0/1` | `-0.0054` | `79.0` | `3.0` |

## Selector decisions

- seed `299`: selected `magnitude` via `magnitude_trainable_capacity_guardrail`; global_synflow=0.0027/dead176, magnitude=-0.0054/dead82, minimal_liveness_repair=-0.0061/dead0, all_route_liveness_floor=-0.0061/dead0, attn_mlp_readout_repair=-0.0069/dead0

## Interpretation

This is a one-seed fresh validation of the V2 transformer selector after the strong-recipe V1 failure. It tests whether adding a trainable-capacity guardrail can avoid overselecting SynFlow when magnitude is already comparatively viable.
