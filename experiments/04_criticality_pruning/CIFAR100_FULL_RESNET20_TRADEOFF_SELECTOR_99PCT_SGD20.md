# CIFAR-100 Full ResNet-20 Tradeoff Selector at 99%

Fresh full-CIFAR-100 ResNet-20 99% sparsity test of the feature-preservation / liveness tradeoff selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[282, 283]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0698` | `0.0003` |  |  | `0.0000` | `0.6559` | `3.0585` | `557.0` |
| `feature_viability_repair` | `0.0821` | `0.0046` | `+0.0123` | `2/2` | `0.5745` | `0.5575` | `2.8939` | `61.5` |
| `plain_reserve` | `0.0877` | `0.0055` | `+0.0180` | `2/2` | `1.1977` | `1.1190` | `1.0936` | `0.5` |
| `predicted_route_split` | `0.0887` | `0.0041` | `+0.0189` | `2/2` | `1.0059` | `1.2900` | `2.0334` | `1.0` |
| `tradeoff_policy` | `0.0821` | `0.0046` | `+0.0123` | `2/2` | `0.5745` | `0.5575` | `2.8939` | `61.5` |

## Selector decisions

- seed `282`: selected `feature_viability_repair`; scores feature_viability_repair=0.767, magnitude=0.739, predicted_route_split=0.661, plain_reserve=0.500
- seed `283`: selected `feature_viability_repair`; scores feature_viability_repair=0.789, magnitude=0.736, predicted_route_split=0.648, plain_reserve=0.504

## Interpretation

This tests whether the feature-preservation/liveness tradeoff selector handles output-diverse CIFAR-100, where previous evidence favored stronger readout preservation. A failure here is useful: it would show the score still needs task-ecology terms rather than a fixed feature-overlap weight.
