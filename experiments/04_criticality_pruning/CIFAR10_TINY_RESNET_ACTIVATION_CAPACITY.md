# CIFAR-10 TinyResNet Activation-Supported Capacity

CIFAR-10 TinyResNet test of activation-supported path-capacity reserve pruning under residual connections.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Calibration batches: `8`

| Sparsity | Method | Before FT | After FT | Dead outputs |
|---:|---|---:|---:|---:|
| `0.98` | `magnitude` | `0.1009` | `0.2972` | `133.5` |
| `0.98` | `global_synflow` | `0.0998` | `0.1007` | `277.5` |
| `0.98` | `reserve_0.60` | `0.0981` | `0.3335` | `1.0` |
| `0.98` | `activation_reserve_0.60` | `0.1002` | `0.2959` | `1.0` |
| `0.99` | `magnitude` | `0.1007` | `0.2450` | `218.0` |
| `0.99` | `global_synflow` | `0.0998` | `0.1026` | `297.0` |
| `0.99` | `reserve_0.60` | `0.1007` | `0.2210` | `3.0` |
| `0.99` | `activation_reserve_0.60` | `0.1007` | `0.2005` | `2.5` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-0.0011` | `0/2` | `-0.1965` | `0/2` |
| `0.98` | `reserve_0.60` | `-0.0028` | `0/2` | `+0.0363` | `2/2` |
| `0.98` | `activation_reserve_0.60` | `-0.0007` | `0/2` | `-0.0013` | `1/2` |
| `0.99` | `global_synflow` | `-0.0009` | `0/2` | `-0.1424` | `0/2` |
| `0.99` | `reserve_0.60` | `+0.0000` | `0/2` | `-0.0240` | `0/2` |
| `0.99` | `activation_reserve_0.60` | `+0.0000` | `0/2` | `-0.0445` | `0/2` |

## Interpretation

This is a use-dependent stabilization test: protected capacity is ranked by saliency multiplied by presynaptic activation on calibration data.
If it beats plain reserve capacity, route quality is likely being captured by activity. If it does not, the missing residual-network variable is not simple presynaptic activity.
