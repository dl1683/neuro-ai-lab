# TinyImageNet-200 Pretrained ResNet-18 Feature-Viability Repair at 95%

TinyImageNet-200 pretrained ResNet-18 95% sparsity test for magnitude-first feature-subspace preservation plus minimal liveness repair.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seed: `274`
Dense accuracy: `0.6107`

| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1513` |  | `3.6131` | `3.1172` | `5.3209` | `11.0` |
| `plain_reserve` | `0.0323` | `-0.1190` | `4.2285` | `4.1744` | `5.5320` | `0.0` |
| `feature_viability_repair` | `0.1503` | `-0.0010` | `3.6131` | `3.1172` | `5.3209` | `0.0` |

## Interpretation

This tests the new pretrained-network hypothesis: start from feature-preserving magnitude pruning, then only repair true dead output rows while preserving the global parameter budget. If this works better than reserve, the external failure is not a rejection of circuit viability; it means pretrained systems need viability constrained by feature-subspace preservation.
