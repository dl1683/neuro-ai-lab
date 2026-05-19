# CIFAR-10 CNN Multi-Cut Capacity Result

This is the first constructive path-capacity pruning result after the SynFlow pathology discovery.

## Question

Can circuit-viability constraints turn global SynFlow from a structurally dead mask into a recoverable sparse network under the same global parameter budget?

The prior single-bridge capacity repair kept `fc1` alive but failed at `99%` because other cuts (`conv2`/`conv3`) collapsed. This experiment adds minimum capacities for multiple communication cuts: early/mid/late convolutional layers, the dense classifier bridge, and the output classifier.

## Setup

- Dataset: real CIFAR-10 images.
- Model: small CNN with dense classifier tail.
- Train subset: `20k` images.
- Test subset: `5k` images.
- Seeds: `141, 142, 143, 144`.
- Device: CUDA.
- Sparsities: `98%`, `99%`.
- Methods:
  - `magnitude`: ordinary global magnitude pruning.
  - `global_synflow`: ordinary global SynFlow pruning.
  - `multicut_capacity`: SynFlow saliency with minimum capacity floors for critical cuts, under the same total parameter budget.

## Result

| Sparsity | Method | Before FT | After FT | fc1 keep | Dead fc1 | conv1/conv2/conv3/fc2 keep |
|---:|---|---:|---:|---:|---:|---|
| `0.98` | `magnitude` | `0.1393` | `0.4405` | `0.0086` | `73.5` | `0.66/0.08/0.03/0.16` |
| `0.98` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.96/0.13/0.01/0.90` |
| `0.98` | `multicut_capacity` | `0.1365` | `0.4009` | `0.0084` | `0.0` | `0.92/0.03/0.02/0.81` |
| `0.99` | `magnitude` | `0.1233` | `0.3262` | `0.0036` | `79.5` | `0.62/0.04/0.01/0.08` |
| `0.99` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.92/0.01/0.00/0.82` |
| `0.99` | `multicut_capacity` | `0.1132` | `0.3341` | `0.0044` | `0.0` | `0.73/0.01/0.00/0.36` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-4.16` pts | `0/4` | `-34.29` pts | `0/4` |
| `0.98` | `multicut_capacity` | `-0.28` pts | `2/4` | `-3.96` pts | `0/4` |
| `0.99` | `global_synflow` | `-2.57` pts | `0/4` | `-22.85` pts | `0/4` |
| `0.99` | `multicut_capacity` | `-1.01` pts | `1/4` | `+0.80` pts | `2/4` |

## Interpretation

This is a real step beyond pathology detection:

- Global SynFlow dies because it creates topological cutsets.
- A single dense-bridge capacity floor is not enough at `99%`, because other layers can still become zero-capacity cuts.
- Multi-cut capacity constraints restore trainability and, at `99%`, slightly outperform magnitude after fine-tuning on this four-seed run.

This does not prove a superior general method yet. At `98%`, magnitude still wins after fine-tuning by about `3.96` points. But the result supports the north-star thesis: severe pruning should preserve circuit capacity across multiple cuts, not just rank individual synapses globally.

## Neuroscience framing

This is the first concrete implementation of the repo's stronger neuro-AI thesis:

**Biological pruning is constrained circuit remodeling, not isolated synapse deletion.**

The multi-cut mask acts like a crude homeostatic/circuit-viability constraint. It allows synapse-level saliency to prune aggressively, but prevents the system from deleting entire communication routes.

## Next step

The current capacity floors are hand-chosen. The next version should learn or predict the required cut capacities from score distributions and activation flow:

- predict dead-cut probability before pruning;
- allocate capacity by expected route viability;
- replace fixed floors with score-distribution-aware min-cut constraints;
- test against SNIP/GraSP/ERK, not only SynFlow.

Artifact: `results/04_criticality_pruning/cifar10_cnn_multicut_capacity_pruning.json`.
