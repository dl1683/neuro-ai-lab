# CIFAR-10 CNN Bridge-Floor Test

CIFAR-10 CNN structural bridge-floor test. Repairs magnitude masks so each fc1 hidden unit has at least 1 or 2 incoming weights, preserving total keep count by removing low-score unprotected weights elsewhere.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

## Means

| Sparsity | Method | Before FT | After FT | Dead fc1 hidden | fc1 keep rate |
|---:|---|---:|---:|---:|---:|
| `0.98` | `magnitude` | `0.1590` | `0.4435` | `80.5` | `0.0081` |
| `0.98` | `alpha03` | `0.1452` | `0.4431` | `83.0` | `0.0052` |
| `0.98` | `bridge_floor1` | `0.1573` | `0.4445` | `0.0` | `0.0084` |
| `0.98` | `bridge_floor2` | `0.1522` | `0.4438` | `0.0` | `0.0086` |
| `0.99` | `magnitude` | `0.1121` | `0.3239` | `87.0` | `0.0033` |
| `0.99` | `alpha03` | `0.1055` | `0.3041` | `96.8` | `0.0019` |
| `0.99` | `bridge_floor1` | `0.1128` | `0.3246` | `0.0` | `0.0036` |
| `0.99` | `bridge_floor2` | `0.1124` | `0.3151` | `0.0` | `0.0039` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `alpha03` | `-0.0139` | `1/4` | `-0.0004` | `1/4` |
| `0.98` | `bridge_floor1` | `-0.0018` | `1/4` | `+0.0010` | `2/4` |
| `0.98` | `bridge_floor2` | `-0.0068` | `1/4` | `+0.0003` | `1/4` |
| `0.99` | `alpha03` | `-0.0066` | `1/4` | `-0.0198` | `1/4` |
| `0.99` | `bridge_floor1` | `+0.0007` | `2/4` | `+0.0007` | `2/4` |
| `0.99` | `bridge_floor2` | `+0.0003` | `2/4` | `-0.0088` | `1/4` |
