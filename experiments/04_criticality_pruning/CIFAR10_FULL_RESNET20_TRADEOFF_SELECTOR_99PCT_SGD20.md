# CIFAR-10 Full ResNet-20 Tradeoff Selector at 99%

Fresh full-CIFAR ResNet-20 99% sparsity test of a feature-preservation / liveness tradeoff selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[279, 280]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4460` | `0.0066` |  |  | `0.0000` | `1.3667` | `3.7457` | `373.0` |
| `feature_viability_repair` | `0.4997` | `0.0243` | `+0.0537` | `2/2` | `0.5827` | `1.2214` | `3.7182` | `32.0` |
| `plain_reserve` | `0.4913` | `0.0088` | `+0.0454` | `2/2` | `1.2483` | `1.1786` | `3.2587` | `1.5` |
| `predicted_route_split` | `0.4850` | `0.0179` | `+0.0391` | `2/2` | `1.0395` | `1.5641` | `3.6272` | `1.0` |
| `tradeoff_policy` | `0.4997` | `0.0243` | `+0.0537` | `2/2` | `0.5827` | `1.2214` | `3.7182` | `32.0` |

## Selector decisions

- seed `279`: selected `feature_viability_repair`; scores feature_viability_repair=0.827, predicted_route_split=0.777, magnitude=0.749, plain_reserve=0.684
- seed `280`: selected `feature_viability_repair`; scores feature_viability_repair=0.862, predicted_route_split=0.804, magnitude=0.754, plain_reserve=0.721

## Interpretation

This is a prospective test of a selector that treats viability as a tradeoff: preserve useful magnitude-like feature structure while repairing enough route liveness to keep the sparse circuit trainable.

The selector is not allowed to use post-finetune accuracy. It sees only the realized candidate masks and pre-finetune route-quality diagnostics.
