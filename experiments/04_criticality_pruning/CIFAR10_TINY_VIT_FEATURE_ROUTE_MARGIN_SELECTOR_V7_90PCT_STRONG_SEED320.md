# CIFAR-10 TinyViT V7 Prospective Validation at 90%: Seed 320

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V7 selector on seed 320. Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V7 selector on seed 320. V7 keeps V6 and adds a magnitude-vs-live-repair guardrail: when direct live-repair feature advantage over magnitude is tiny, keep the magnitude sparse template.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[320]`
Dense accuracy mean: `0.7228`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0866` | `0.0901` |  |  | `-0.0134` | `66.0` | `5.0` |
| `global_synflow` | `0.0896` | `0.1193` | `+0.0292` | `1/1` | `0.0246` | `74.0` | `77.0` |
| `minimal_liveness_repair` | `0.0909` | `0.0920` | `+0.0019` | `1/1` | `-0.0110` | `1.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.0900` | `0.0916` | `+0.0015` | `1/1` | `-0.0118` | `1.0` | `0.0` |
| `all_route_liveness_floor` | `0.0910` | `0.0927` | `+0.0026` | `1/1` | `-0.0110` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.0896` | `0.1193` | `+0.0292` | `1/1` | `0.0246` | `74.0` | `77.0` |

## Selector decisions

- seed `320`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0246/before0.0896/dead151, all_route_liveness_floor=-0.0110/before0.0910/dead0, minimal_liveness_repair=-0.0110/before0.0909/dead1, attn_mlp_readout_repair=-0.0118/before0.0900/dead1, magnitude=-0.0134/before0.0866/dead71

## Interpretation

This run tests the V7 correction motivated by seed 312. V7 does not change the SynFlow branches. It only asks whether masked-before trainability is a better tie-breaker than a tiny feature-alignment margin inside the live-repair family.
