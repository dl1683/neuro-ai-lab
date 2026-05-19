# CIFAR-10 TinyViT Feature-Route Margin Selector V3 at 90%: Stronger Recipe

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of a V3 feature-route selector. V3 adds an all-route liveness guardrail when liveness repair preserves feature alignment within margin and removes much more route death than magnitude.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[301]`
Dense accuracy mean: `0.7282`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1139` |  |  | `0.0353` | `62.0` | `4.0` |
| `global_synflow` | `0.1452` | `+0.0313` | `1/1` | `0.0561` | `101.0` | `79.0` |
| `minimal_liveness_repair` | `0.1142` | `+0.0003` | `1/1` | `0.0376` | `2.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.1122` | `-0.0017` | `0/1` | `0.0364` | `2.0` | `0.0` |
| `all_route_liveness_floor` | `0.1159` | `+0.0020` | `1/1` | `0.0376` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1452` | `+0.0313` | `1/1` | `0.0561` | `101.0` | `79.0` |

## Selector decisions

- seed `301`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0561/dead180, minimal_liveness_repair=0.0376/dead2, all_route_liveness_floor=0.0376/dead0, attn_mlp_readout_repair=0.0364/dead2, magnitude=0.0353/dead66

## Interpretation

This is a one-seed fresh validation of the V3 transformer selector after the V2 replicate failure. The selector now has three possible regimes: feature-preserving SynFlow, magnitude trainable-capacity, or all-route liveness.
