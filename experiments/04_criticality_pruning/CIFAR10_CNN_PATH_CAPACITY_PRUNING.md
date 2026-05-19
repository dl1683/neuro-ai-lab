# CIFAR-10 CNN Path-Capacity Pruning

CIFAR-10 CNN first path-capacity pruning test. SynFlow saliency is constrained so fc1 keeps a minimum bridge capacity; bridge ranking uses magnitude or activation-supported hybrid scores. Same global parameter budget as ordinary global pruning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

## Means

| Sparsity | Method | Before FT | After FT | fc1 keep rate | Dead fc1 hidden | conv keep rates |
|---:|---|---:|---:|---:|---:|---|
| `0.98` | `magnitude` | `0.1575` | `0.4477` | `0.0086` | `77.8` | `0.66/0.08/0.03` |
| `0.98` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.96/0.13/0.01` |
| `0.98` | `pathcap_synflow_bridge_mag` | `0.0968` | `0.3341` | `0.0080` | `0.0` | `0.95/0.05/0.00` |
| `0.98` | `pathcap_synflow_bridge_hybrid` | `0.0963` | `0.3317` | `0.0080` | `0.0` | `0.95/0.05/0.00` |
| `0.99` | `magnitude` | `0.1205` | `0.3367` | `0.0036` | `83.2` | `0.61/0.04/0.01` |
| `0.99` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.93/0.01/0.00` |
| `0.99` | `pathcap_synflow_bridge_mag` | `0.0976` | `0.0976` | `0.0040` | `0.0` | `0.81/0.00/0.00` |
| `0.99` | `pathcap_synflow_bridge_hybrid` | `0.0976` | `0.0976` | `0.0040` | `0.0` | `0.81/0.00/0.00` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-0.0599` | `0/4` | `-0.3501` | `0/4` |
| `0.98` | `pathcap_synflow_bridge_mag` | `-0.0607` | `0/4` | `-0.1136` | `0/4` |
| `0.98` | `pathcap_synflow_bridge_hybrid` | `-0.0612` | `0/4` | `-0.1159` | `0/4` |
| `0.99` | `global_synflow` | `-0.0229` | `1/4` | `-0.2390` | `0/4` |
| `0.99` | `pathcap_synflow_bridge_mag` | `-0.0229` | `1/4` | `-0.2390` | `0/4` |
| `0.99` | `pathcap_synflow_bridge_hybrid` | `-0.0229` | `1/4` | `-0.2390` | `0/4` |
