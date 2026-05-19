# CIFAR-10 Full ResNet-20 Feature Repair vs Homeostasis at 99%

Fresh full-CIFAR ResNet-20 99% sparsity test comparing magnitude-first feature repair against homeostatic/ecology repair when magnitude has a dead route floor.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[277, 278]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4591` | `0.0057` |  |  | `0.0000` | `1.4708` | `3.7702` | `364.0` |
| `feature_viability_repair` | `0.4926` | `0.0053` | `+0.0335` | `2/2` | `0.6041` | `1.3060` | `3.7490` | `35.5` |
| `plain_reserve` | `0.4778` | `0.0117` | `+0.0187` | `2/2` | `1.2301` | `1.1738` | `3.3411` | `1.0` |
| `predicted_route_split` | `0.4733` | `0.0007` | `+0.0142` | `2/2` | `1.0256` | `1.5676` | `3.6328` | `1.0` |
| `unified_policy` | `0.4778` | `0.0117` | `+0.0187` | `2/2` | `1.2301` | `1.1738` | `3.3411` | `1.0` |

## Decisions

- seed `277`: family `ecology_selector` route_floor `0.0000` method `plain_reserve`
- seed `278`: family `ecology_selector` route_floor `0.0000` method `plain_reserve`

## Interpretation

This tests the bifurcation in the unified selector. If magnitude has a dead route floor, feature-preserving liveness repair should not be enough; the selector should choose homeostatic/ecology repair instead.
