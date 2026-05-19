# CIFAR-10 TinyViT V5 Prospective Validation at 90%: Seed 311

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V5 selector on seed 311. The seed is evaluated prospectively with the fixed V5 rule before masked fine-tuning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[311]`
Dense accuracy mean: `0.7291`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0776` | `0.0979` |  |  | `-0.0235` | `79.0` | `3.0` |
| `global_synflow` | `0.1229` | `0.1591` | `+0.0612` | `1/1` | `0.0423` | `92.0` | `81.0` |
| `minimal_liveness_repair` | `0.0778` | `0.0978` | `-0.0001` | `0/1` | `-0.0224` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.0735` | `0.0959` | `-0.0020` | `0/1` | `-0.0230` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.0778` | `0.0997` | `+0.0018` | `1/1` | `-0.0224` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1229` | `0.1591` | `+0.0612` | `1/1` | `0.0423` | `92.0` | `81.0` |

## Selector decisions

- seed `311`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0423/before0.1229/dead173, minimal_liveness_repair=-0.0224/before0.0778/dead0, all_route_liveness_floor=-0.0224/before0.0778/dead0, attn_mlp_readout_repair=-0.0230/before0.0735/dead0, magnitude=-0.0235/before0.0776/dead82

## Interpretation

This run is a prospective fixed-rule validation. The V5 selector chooses from pre-finetune residual-stream alignment, route liveness, and masked-before trainability; after-FT recovery is used only for evaluation.
