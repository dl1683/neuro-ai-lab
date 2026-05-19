# CIFAR-10 TinyViT Feature-Route Margin Selector V4 at 90%: Stronger Recipe

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of a V4 feature-route selector. V4 adds masked pre-finetune accuracy as a trainability guardrail for the ambiguous liveness-vs-magnitude branch.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[304]`
Dense accuracy mean: `0.7263`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0984` | `0.1022` |  |  | `-0.0156` | `66.0` | `3.0` |
| `global_synflow` | `0.1555` | `0.1709` | `+0.0687` | `1/1` | `0.0323` | `103.0` | `82.0` |
| `minimal_liveness_repair` | `0.0933` | `0.1026` | `+0.0004` | `1/1` | `-0.0144` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.0930` | `0.1026` | `+0.0004` | `1/1` | `-0.0148` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.0933` | `0.1030` | `+0.0008` | `1/1` | `-0.0144` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1555` | `0.1709` | `+0.0687` | `1/1` | `0.0323` | `103.0` | `82.0` |

## Selector decisions

- seed `304`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0323/before0.1555/dead185, minimal_liveness_repair=-0.0144/before0.0933/dead0, all_route_liveness_floor=-0.0144/before0.0933/dead0, attn_mlp_readout_repair=-0.0148/before0.0930/dead0, magnitude=-0.0156/before0.0984/dead69

## Interpretation

V4 tests whether masked pre-finetune accuracy can serve as the missing trainability term in the ambiguous liveness-vs-magnitude branch. The decision is made before masked fine-tuning; after-FT recovery is only used for evaluation.
