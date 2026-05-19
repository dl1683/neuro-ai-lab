# TinyImageNet-200 Pretrained ResNet-18 Feature-Viability Repair at 99%

TinyImageNet-200 pretrained ResNet-18 99% sparsity test for magnitude-first feature-subspace preservation plus minimal liveness repair.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seed: `275`
Dense accuracy: `0.5950`

| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0120` |  | `0.8541` | `1.7934` | `4.3197` | `365.0` |
| `plain_reserve` | `0.0087` | `-0.0033` | `2.6813` | `2.6771` | `2.6779` | `0.0` |
| `feature_viability_repair` | `0.0143` | `+0.0023` | `1.5496` | `1.9211` | `4.3166` | `4.0` |

## Interpretation

This repeats the feature-preserving repair at the original 99% external cliff. It tests whether minimal liveness repair can rescue extreme pretrained sparsity or only preserve magnitude-level behavior at moderate sparsity.
