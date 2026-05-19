# CIFAR-10 TinyViT Feature-Route Margin Selector V4 at 90%: Seed 306

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V4 selector on seed 306, which the branch scanner flagged as a non-SynFlow feature-argmax decision.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[306]`
Dense accuracy mean: `0.7299`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1017` | `0.1099` |  |  | `0.0205` | `86.0` | `3.0` |
| `global_synflow` | `0.1040` | `0.1329` | `+0.0230` | `1/1` | `-0.0064` | `113.0` | `70.0` |
| `minimal_liveness_repair` | `0.1086` | `0.1158` | `+0.0059` | `1/1` | `0.0239` | `0.0` | `1.0` |
| `attn_mlp_readout_repair` | `0.1097` | `0.1158` | `+0.0059` | `1/1` | `0.0248` | `0.0` | `1.0` |
| `all_route_liveness_floor` | `0.1085` | `0.1153` | `+0.0054` | `1/1` | `0.0241` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1097` | `0.1158` | `+0.0059` | `1/1` | `0.0248` | `0.0` | `1.0` |

## Selector decisions

- seed `306`: selected `attn_mlp_readout_repair` via `feature_argmax`; attn_mlp_readout_repair=0.0248/before0.1097/dead1, all_route_liveness_floor=0.0241/before0.1085/dead0, minimal_liveness_repair=0.0239/before0.1086/dead1, magnitude=0.0205/before0.1017/dead89, global_synflow=-0.0064/before0.1040/dead183

## Interpretation

This validates whether V4 can choose a liveness/repair-style mask prospectively when residual-stream feature alignment favors it, rather than only selecting SynFlow in feature-dominant seeds.
