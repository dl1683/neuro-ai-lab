# CIFAR-10 TinyResNet Balanced Route Capacity

CIFAR-10 TinyResNet test of balanced residual route-capacity pruning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Protected split: `{'main': 0.5, 'projection': 0.25, 'readout': 0.25}`

| Sparsity | Method | Before FT | After FT | Route min | Projection min | FC score | Dead outputs |
|---:|---|---:|---:|---:|---:|---:|---:|
| `0.98` | `magnitude` | `0.1010` | `0.2928` | `0.5041` | `2.2884` | `3.3049` | `131.0` |
| `0.98` | `global_synflow` | `0.0998` | `0.0979` | `0.0000` | `0.0000` | `3.8928` | `276.5` |
| `0.98` | `reserve_0.60` | `0.1002` | `0.3331` | `1.4018` | `1.3545` | `3.4609` | `1.0` |
| `0.98` | `balanced_route_0.60` | `0.1010` | `0.3298` | `1.2347` | `1.2347` | `3.7086` | `0.5` |
| `0.99` | `magnitude` | `0.1007` | `0.2431` | `0.0019` | `1.8136` | `2.7081` | `219.0` |
| `0.99` | `global_synflow` | `0.0998` | `0.1010` | `0.0000` | `0.0000` | `3.6674` | `297.0` |
| `0.99` | `reserve_0.60` | `0.1007` | `0.2007` | `1.0341` | `0.9650` | `1.7307` | `3.0` |
| `0.99` | `balanced_route_0.60` | `0.1007` | `0.2568` | `0.9100` | `0.8844` | `2.5649` | `1.5` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-0.0012` | `0/2` | `-0.1949` | `0/2` |
| `0.98` | `reserve_0.60` | `-0.0008` | `0/2` | `+0.0403` | `2/2` |
| `0.98` | `balanced_route_0.60` | `+0.0000` | `1/2` | `+0.0370` | `2/2` |
| `0.99` | `global_synflow` | `-0.0009` | `0/2` | `-0.1421` | `0/2` |
| `0.99` | `reserve_0.60` | `+0.0000` | `0/2` | `-0.0424` | `0/2` |
| `0.99` | `balanced_route_0.60` | `+0.0000` | `0/2` | `+0.0137` | `1/2` |

## Interpretation

This tests whether residual recovery improves when protected capacity is explicitly balanced across main transformations, projection shortcuts, and classifier readout.
