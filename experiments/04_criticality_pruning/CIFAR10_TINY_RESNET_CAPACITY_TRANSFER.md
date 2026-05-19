# CIFAR-10 TinyResNet Capacity Transfer

CIFAR-10 TinyResNet transfer test for path-capacity reserve pruning under residual connections.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

| Sparsity | Method | Before FT | After FT | Dead outputs |
|---:|---|---:|---:|---:|
| `0.98` | `magnitude` | `0.1005` | `0.3132` | `132.0` |
| `0.98` | `global_synflow` | `0.0998` | `0.0979` | `277.0` |
| `0.98` | `reserve_0.50` | `0.0992` | `0.3174` | `0.5` |
| `0.98` | `reserve_0.60` | `0.1003` | `0.3227` | `1.0` |
| `0.99` | `magnitude` | `0.1007` | `0.2423` | `219.0` |
| `0.99` | `global_synflow` | `0.0998` | `0.1010` | `297.0` |
| `0.99` | `reserve_0.50` | `0.1008` | `0.2163` | `1.5` |
| `0.99` | `reserve_0.60` | `0.1010` | `0.2132` | `3.0` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-0.0007` | `0/2` | `-0.2153` | `0/2` |
| `0.98` | `reserve_0.50` | `-0.0013` | `1/2` | `+0.0042` | `1/2` |
| `0.98` | `reserve_0.60` | `-0.0002` | `1/2` | `+0.0095` | `1/2` |
| `0.99` | `global_synflow` | `-0.0009` | `0/2` | `-0.1413` | `0/2` |
| `0.99` | `reserve_0.50` | `+0.0001` | `1/2` | `-0.0260` | `0/2` |
| `0.99` | `reserve_0.60` | `+0.0003` | `1/2` | `-0.0291` | `0/2` |
