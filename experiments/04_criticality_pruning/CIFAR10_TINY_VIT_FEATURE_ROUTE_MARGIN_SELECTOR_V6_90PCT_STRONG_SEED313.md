# CIFAR-10 TinyViT V6 Prospective Validation at 90%: Seed 313

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V6 selector on seed 313. V6 keeps the V5 SynFlow masked-recovery prior and adds a live-repair tie-breaker: when live-repair feature margins are tiny, choose the repair with higher masked-before trainability.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[313]`
Dense accuracy mean: `0.7229`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1561` | `0.1644` |  |  | `0.0602` | `69.0` | `5.0` |
| `global_synflow` | `0.0903` | `0.1010` | `-0.0634` | `0/1` | `0.0013` | `112.0` | `105.0` |
| `minimal_liveness_repair` | `0.1540` | `0.1627` | `-0.0017` | `0/1` | `0.0599` | `1.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.1553` | `0.1622` | `-0.0022` | `0/1` | `0.0597` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.1542` | `0.1628` | `-0.0016` | `0/1` | `0.0596` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1561` | `0.1644` | `+0.0000` | `0/1` | `0.0602` | `69.0` | `5.0` |

## Selector decisions

- seed `313`: selected `magnitude` via `feature_argmax`; magnitude=0.0602/before0.1561/dead74, minimal_liveness_repair=0.0599/before0.1540/dead1, attn_mlp_readout_repair=0.0597/before0.1553/dead0, all_route_liveness_floor=0.0596/before0.1542/dead0, global_synflow=0.0013/before0.0903/dead217

## Interpretation

This run tests the V6 correction motivated by seed 312. V6 does not change the SynFlow branches. It only asks whether masked-before trainability is a better tie-breaker than a tiny feature-alignment margin inside the live-repair family.
