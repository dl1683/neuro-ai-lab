# CIFAR-10 TinyViT Feature-Route Margin Selector V5 at 90%: Stronger Recipe

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V5 selector. V5 adds a SynFlow masked-recovery prior: when SynFlow's masked-before accuracy is at least magnitude and close to the selected repair, prefer SynFlow despite lower centered CLS alignment.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[308]`
Dense accuracy mean: `0.7254`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0603` | `0.0725` |  |  | `0.0142` | `55.0` | `4.0` |
| `global_synflow` | `0.0723` | `0.1018` | `+0.0293` | `1/1` | `0.0281` | `105.0` | `91.0` |
| `minimal_liveness_repair` | `0.0603` | `0.0734` | `+0.0009` | `1/1` | `0.0187` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.0597` | `0.0724` | `-0.0001` | `0/1` | `0.0165` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.0603` | `0.0746` | `+0.0021` | `1/1` | `0.0187` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.0723` | `0.1018` | `+0.0293` | `1/1` | `0.0281` | `105.0` | `91.0` |

## Selector decisions

- seed `308`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0281/before0.0723/dead196, minimal_liveness_repair=0.0187/before0.0603/dead0, all_route_liveness_floor=0.0187/before0.0603/dead0, attn_mlp_readout_repair=0.0165/before0.0597/dead0, magnitude=0.0142/before0.0603/dead59

## Interpretation

This is the first prospective test of the V5 SynFlow recovery-prior rule. The decision is made before masked fine-tuning from feature alignment, route liveness, and masked-before accuracy.
