# CIFAR-10 TinyResNet Backbone Capacity

CIFAR-10 TinyResNet test of residual-backbone-aware path-capacity pruning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

| Sparsity | Method | Before FT | After FT | Dead outputs |
|---:|---|---:|---:|---:|
| `0.98` | `magnitude` | `0.1036` | `0.2943` | `130.0` |
| `0.98` | `global_synflow` | `0.0998` | `0.0979` | `275.0` |
| `0.98` | `reserve_0.60` | `0.0996` | `0.3146` | `1.0` |
| `0.98` | `backbone_reserve_0.60` | `0.1031` | `0.3219` | `1.0` |
| `0.99` | `magnitude` | `0.1006` | `0.2445` | `217.5` |
| `0.99` | `global_synflow` | `0.0998` | `0.1010` | `296.5` |
| `0.99` | `reserve_0.60` | `0.0988` | `0.2058` | `3.0` |
| `0.99` | `backbone_reserve_0.60` | `0.1007` | `0.1775` | `3.5` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-0.0038` | `0/2` | `-0.1964` | `0/2` |
| `0.98` | `reserve_0.60` | `-0.0040` | `0/2` | `+0.0203` | `2/2` |
| `0.98` | `backbone_reserve_0.60` | `-0.0005` | `1/2` | `+0.0276` | `2/2` |
| `0.99` | `global_synflow` | `-0.0008` | `0/2` | `-0.1435` | `0/2` |
| `0.99` | `reserve_0.60` | `-0.0018` | `0/2` | `-0.0387` | `0/2` |
| `0.99` | `backbone_reserve_0.60` | `+0.0001` | `1/2` | `-0.0670` | `0/2` |

## Interpretation

This tests a residual-specific hypothesis: projection shortcuts and classifier routes are communication backbones, so they receive extra protected capacity and use magnitude ranking inside protected budgets.
