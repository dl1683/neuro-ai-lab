# CIFAR-10 TinyViT V6 Prospective Validation at 90%: Seed 315

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V6 selector on seed 315. V6 keeps the V5 SynFlow masked-recovery prior and adds a live-repair tie-breaker: when live-repair feature margins are tiny, choose the repair with higher masked-before trainability.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[315]`
Dense accuracy mean: `0.7244`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1110` | `0.1242` |  |  | `0.0351` | `77.0` | `4.0` |
| `global_synflow` | `0.0434` | `0.0955` | `-0.0287` | `0/1` | `-0.0314` | `105.0` | `80.0` |
| `minimal_liveness_repair` | `0.1129` | `0.1234` | `-0.0008` | `0/1` | `0.0351` | `1.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.1134` | `0.1239` | `-0.0003` | `0/1` | `0.0337` | `1.0` | `0.0` |
| `all_route_liveness_floor` | `0.1140` | `0.1234` | `-0.0008` | `0/1` | `0.0353` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1140` | `0.1234` | `-0.0008` | `0/1` | `0.0353` | `0.0` | `0.0` |

## Selector decisions

- seed `315`: selected `all_route_liveness_floor` via `feature_argmax`; all_route_liveness_floor=0.0353/before0.1140/dead0, minimal_liveness_repair=0.0351/before0.1129/dead1, magnitude=0.0351/before0.1110/dead81, attn_mlp_readout_repair=0.0337/before0.1134/dead1, global_synflow=-0.0314/before0.0434/dead185

## Interpretation

This run tests the V6 correction motivated by seed 312. V6 does not change the SynFlow branches. It only asks whether masked-before trainability is a better tie-breaker than a tiny feature-alignment margin inside the live-repair family.
