# TinyImageNet-200 Pretrained ResNet-18 Feature-Viability Repair at 95%: Replicate

Fresh-seed TinyImageNet-200 pretrained ResNet-18 95% sparsity replicate for magnitude-first feature-subspace preservation plus minimal liveness repair.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seed: `276`
Dense accuracy: `0.5937`

| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1460` |  | `3.6147` | `3.1168` | `5.3185` | `11.0` |
| `plain_reserve` | `0.0337` | `-0.1123` | `4.2285` | `4.1744` | `5.5314` | `0.0` |
| `feature_viability_repair` | `0.1460` | `+0.0000` | `3.6147` | `3.1168` | `5.3185` | `0.0` |

## Interpretation

This fresh seed tests whether feature-viability repair consistently preserves pretrained magnitude-level recovery while eliminating dead outputs, or whether the first 95% result was seed-specific.
