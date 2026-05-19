# CIFAR-10 TinyViT V5 Prospective Validation at 90%: Seed 312

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V5 selector on seed 312. The seed is evaluated prospectively with the fixed V5 rule before masked fine-tuning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[312]`
Dense accuracy mean: `0.7297`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1023` | `0.1189` |  |  | `0.0130` | `60.0` | `4.0` |
| `global_synflow` | `0.0739` | `0.1068` | `-0.0121` | `0/1` | `0.0022` | `75.0` | `64.0` |
| `minimal_liveness_repair` | `0.1036` | `0.1210` | `+0.0021` | `1/1` | `0.0122` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.1026` | `0.1187` | `-0.0002` | `0/1` | `0.0140` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.1036` | `0.1205` | `+0.0016` | `1/1` | `0.0122` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1026` | `0.1187` | `-0.0002` | `0/1` | `0.0140` | `0.0` | `0.0` |

## Selector decisions

- seed `312`: selected `attn_mlp_readout_repair` via `feature_argmax`; attn_mlp_readout_repair=0.0140/before0.1026/dead0, magnitude=0.0130/before0.1023/dead64, minimal_liveness_repair=0.0122/before0.1036/dead0, all_route_liveness_floor=0.0122/before0.1036/dead0, global_synflow=0.0022/before0.0739/dead139

## Interpretation

This run is a prospective fixed-rule validation. The V5 selector chooses from pre-finetune residual-stream alignment, route liveness, and masked-before trainability; after-FT recovery is used only for evaluation.
