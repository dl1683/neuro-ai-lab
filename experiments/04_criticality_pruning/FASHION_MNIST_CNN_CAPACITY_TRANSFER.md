# Fashion-MNIST CNN Capacity Transfer

This experiment tests whether Path-Capacity Pruning transfers back to the original Fashion-MNIST CNN SynFlow collapse setting.

## Setup

- Dataset: Fashion-MNIST.
- Model: small CNN with dense classifier tail.
- Seeds: `191, 192, 193, 194`.
- Sparsities: `98%`, `99%`.
- Reserves: `0.45`, `0.55`, `0.60`.
- Same total parameter budget for all methods.
- Device: CUDA.

## Result

| Sparsity | Method | Before FT | After FT | Dead fc1 | fc1 keep |
|---:|---|---:|---:|---:|---:|
| `0.98` | `magnitude` | `0.5553` | `0.8479` | `39.0` | `0.0095` |
| `0.98` | `global_synflow` | `0.1002` | `0.0992` | `128.0` | `0.0000` |
| `0.98` | `reserve_0.45` | `0.5794` | `0.8475` | `0.0` | `0.0072` |
| `0.98` | `reserve_0.55` | `0.5736` | `0.8498` | `0.0` | `0.0087` |
| `0.98` | `reserve_0.60` | `0.5707` | `0.8514` | `0.0` | `0.0095` |
| `0.99` | `magnitude` | `0.3622` | `0.8024` | `41.8` | `0.0028` |
| `0.99` | `global_synflow` | `0.1002` | `0.0989` | `128.0` | `0.0000` |
| `0.99` | `reserve_0.45` | `0.4098` | `0.8136` | `0.0` | `0.0037` |
| `0.99` | `reserve_0.55` | `0.3800` | `0.8139` | `0.0` | `0.0045` |
| `0.99` | `reserve_0.60` | `0.3850` | `0.8175` | `0.0` | `0.0049` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `reserve_0.45` | `+2.41` pts | `3/4` | `-0.04` pts | `3/4` |
| `0.98` | `reserve_0.55` | `+1.83` pts | `2/4` | `+0.19` pts | `3/4` |
| `0.98` | `reserve_0.60` | `+1.54` pts | `2/4` | `+0.36` pts | `3/4` |
| `0.99` | `reserve_0.45` | `+4.76` pts | `4/4` | `+1.11` pts | `3/4` |
| `0.99` | `reserve_0.55` | `+1.78` pts | `2/4` | `+1.14` pts | `2/4` |
| `0.99` | `reserve_0.60` | `+2.28` pts | `2/4` | `+1.51` pts | `3/4` |

## Interpretation

This is important transfer evidence.

Path-capacity reserve pruning now improves over magnitude on mean in both tested CNN settings:

- CIFAR-10 CNN at `99%` sparsity;
- Fashion-MNIST CNN at `98%` and `99%` sparsity.

The method also consistently prevents dense-bridge death. Global SynFlow stays near chance because it allocates zero capacity to `fc1`, while reserve masks keep `fc1` alive and recover after fine-tuning.

The effect is larger at `99%`, which supports the current thesis that path capacity matters most near the severe sparsity cliff.

Artifact: `results/04_criticality_pruning/fashion_mnist_cnn_capacity_transfer.json`.
