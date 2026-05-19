# TinyImageNet-200 Pretrained ResNet-18 Feature-Viability Repair at 95%: Two-Seed Aggregate

Two-seed TinyImageNet-200 pretrained ResNet-18 95% sparsity aggregate for magnitude-first feature-subspace preservation plus minimal liveness repair.

Seeds: `[274, 276]`
Sources: `tinyimagenet_resnet18_pretrained_feature_viability_95pct.json`, `tinyimagenet_resnet18_pretrained_feature_viability_95pct_replicate.json`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1487` | `0.0027` |  |  | `3.6139` | `3.1170` | `5.3197` | `11.0` |
| `plain_reserve` | `0.0330` | `0.0007` | `-0.1157` | `0/2` | `4.2285` | `4.1744` | `5.5317` | `0.0` |
| `feature_viability_repair` | `0.1482` | `0.0022` | `-0.0005` | `0/2` | `3.6139` | `3.1170` | `5.3197` | `0.0` |

## Interpretation

Across two pretrained TinyImageNet seeds, feature-viability repair preserves magnitude-level accuracy while eliminating dead outputs. Broad reserve consistently destroys pretrained performance.
