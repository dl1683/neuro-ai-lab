# CIFAR-10 CNN SynFlow Pathology

CIFAR-10 CNN SynFlow pathology transfer test. Compares magnitude, global SynFlow, and layerwise SynFlow at 98/99% sparsity before and after masked fine-tuning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

## Means

| Sparsity | Method | Before FT | After FT | fc1 keep rate | Dead fc1 hidden | conv keep rates |
|---:|---|---:|---:|---:|---:|---|
| `0.98` | `magnitude` | `0.1436` | `0.4408` | `0.0081` | `81.8` | `0.66/0.08/0.03` |
| `0.98` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.96/0.13/0.01` |
| `0.98` | `layerwise_synflow` | `0.0953` | `0.2375` | `0.0200` | `84.2` | `0.02/0.02/0.02` |
| `0.99` | `magnitude` | `0.1196` | `0.3324` | `0.0035` | `86.2` | `0.61/0.04/0.01` |
| `0.99` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.92/0.01/0.00` |
| `0.99` | `layerwise_synflow` | `0.1001` | `0.1758` | `0.0100` | `89.2` | `0.01/0.01/0.01` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-0.0460` | `0/4` | `-0.3432` | `0/4` |
| `0.98` | `layerwise_synflow` | `-0.0484` | `0/4` | `-0.2033` | `0/4` |
| `0.99` | `global_synflow` | `-0.0220` | `0/4` | `-0.2348` | `0/4` |
| `0.99` | `layerwise_synflow` | `-0.0195` | `0/4` | `-0.1566` | `0/4` |
