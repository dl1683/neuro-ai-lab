# CIFAR-10 TinyViT Feature-Route Margin Selector V2 at 90%: Strong Replicate

Fresh replicate of the full-train TinyViT CIFAR-10 90% sparsity V2 feature-route selector. Uses the same 20 dense epochs and 5 masked fine-tune epochs as the first strong V2 pilot.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[300]`
Dense accuracy mean: `0.7275`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1382` |  |  | `0.0188` | `74.0` | `2.0` |
| `global_synflow` | `0.1424` | `+0.0042` | `1/1` | `0.0248` | `87.0` | `72.0` |
| `minimal_liveness_repair` | `0.1459` | `+0.0077` | `1/1` | `0.0198` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.1403` | `+0.0021` | `1/1` | `0.0181` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.1484` | `+0.0102` | `1/1` | `0.0198` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1382` | `+0.0000` | `0/1` | `0.0188` | `74.0` | `2.0` |

## Selector decisions

- seed `300`: selected `magnitude` via `magnitude_trainable_capacity_guardrail`; global_synflow=0.0248/dead159, minimal_liveness_repair=0.0198/dead0, all_route_liveness_floor=0.0198/dead0, magnitude=0.0188/dead76, attn_mlp_readout_repair=0.0181/dead0

## Interpretation

This replicate checks whether the V2 trainable-capacity guardrail generalizes to another strong TinyViT seed. The claim is selector behavior, not benchmark strength.
