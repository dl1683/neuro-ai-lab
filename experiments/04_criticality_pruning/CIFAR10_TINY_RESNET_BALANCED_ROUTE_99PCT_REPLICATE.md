# CIFAR-10 TinyResNet Balanced Route 99% Replicate

Four fresh seed TinyResNet 99% sparsity replicate for balanced residual route-capacity pruning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[203, 204, 205, 206]`

| Method | Before FT | After FT | After std | Route min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1066` | `0.2492` | `0.0118` | `0.0009` | `1.7889` | `2.7874` | `220.8` |
| `global_synflow` | `0.0989` | `0.1013` | `0.0015` | `0.0000` | `0.0000` | `3.6335` | `298.0` |
| `reserve_0.60` | `0.1005` | `0.1939` | `0.0328` | `1.0559` | `1.0115` | `1.6283` | `3.5` |
| `balanced_route_0.60` | `0.1011` | `0.2339` | `0.0075` | `0.9438` | `0.8988` | `2.5720` | `3.0` |

## Paired deltas vs magnitude

| Method | Before delta | Before wins | After delta | After wins |
|---|---:|---:|---:|---:|
| `global_synflow` | `-0.0077` | `1/4` | `-0.1480` | `0/4` |
| `reserve_0.60` | `-0.0061` | `1/4` | `-0.0554` | `1/4` |
| `balanced_route_0.60` | `-0.0055` | `1/4` | `-0.0153` | `0/4` |

## Interpretation

This replicate tests whether the two-seed TinyResNet `99%` balanced-route gain survives fresh seeds.
