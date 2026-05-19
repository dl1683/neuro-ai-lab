# CIFAR-10 TinyViT Feature-Subspace Selector at 95%

Fresh TinyViT CIFAR-10 subset 95% sparsity prospective selector. The policy selects the mask with highest pre-finetune centered CLS/residual-stream feature alignment.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[292, 293]`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | Dead outputs | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0882` |  |  | `0.0164` | `1220.0` | `512.0` | `305.5` |
| `global_synflow` | `0.1357` | `+0.0475` | `2/2` | `0.0193` | `2224.5` | `235.0` | `134.5` |
| `minimal_liveness_repair` | `0.0807` | `-0.0075` | `0/2` | `-0.0203` | `82.5` | `0.0` | `29.5` |
| `attn_mlp_readout_repair` | `0.0803` | `-0.0079` | `0/2` | `-0.0189` | `459.0` | `0.0` | `19.5` |
| `all_route_liveness_floor` | `0.0816` | `-0.0066` | `0/2` | `-0.0188` | `0.0` | `0.0` | `0.0` |
| `feature_subspace_policy` | `0.1224` | `+0.0342` | `1/2` | `0.0210` | `1711.5` | `376.5` | `217.5` |

## Selector decisions

- seed `292`: selected `global_synflow`; centered CLS scores global_synflow=0.0218, magnitude=0.0126, all_route_liveness_floor=-0.0198, attn_mlp_readout_repair=-0.0252, minimal_liveness_repair=-0.0264
- seed `293`: selected `magnitude`; centered CLS scores magnitude=0.0202, global_synflow=0.0168, attn_mlp_readout_repair=-0.0126, minimal_liveness_repair=-0.0142, all_route_liveness_floor=-0.0178

## Interpretation

This is a prospective selector test. The selected method is the candidate with the highest pre-finetune centered CLS/residual-stream feature alignment to the dense model. Post-finetune accuracy is measured only after selection.
