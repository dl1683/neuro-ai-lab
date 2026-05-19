# CIFAR-10 TinyViT Feature-Subspace Diagnostic at 95%

TinyViT CIFAR-10 subset 95% sparsity diagnostic comparing pre-finetune CLS/residual-stream feature preservation against route-liveness metrics and masked recovery.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[290, 291]`
Centered CLS cosine vs after-FT correlation: `0.5830`

| Method | After FT | Delta vs magnitude | Wins | CLS cosine | Centered CLS cosine | Dead outputs | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1102` |  |  | `0.0529` | `0.0085` | `1260.0` | `511.5` | `314.0` |
| `global_synflow` | `0.1686` | `+0.0584` | `2/2` | `0.0872` | `0.0526` | `2269.5` | `239.0` | `133.0` |
| `minimal_liveness_repair` | `0.1031` | `-0.0071` | `1/2` | `0.0398` | `-0.0083` | `78.5` | `0.0` | `24.5` |
| `attn_mlp_readout_repair` | `0.0953` | `-0.0149` | `1/2` | `0.0444` | `-0.0030` | `484.5` | `0.0` | `18.5` |
| `all_route_liveness_floor` | `0.1010` | `-0.0092` | `0/2` | `0.0422` | `-0.0071` | `0.0` | `0.0` | `0.0` |

## Interpretation

This diagnostic tests the transformer-specific hypothesis that sparse recovery depends on preserving the dense CLS/residual-stream representation, not only keeping rows alive. The feature score is measured before fine-tuning, so it is a candidate predictor rather than a post-hoc accuracy statistic.
