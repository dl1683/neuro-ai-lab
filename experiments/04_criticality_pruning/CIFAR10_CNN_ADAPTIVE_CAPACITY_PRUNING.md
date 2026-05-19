# CIFAR-10 CNN Adaptive Capacity Pruning

This experiment tests a less hand-tuned version of Path-Capacity Pruning.

## Question

Can a pruning method predict SynFlow's impending cutsets from the score distribution, reserve capacity across critical communication cuts, and recover useful sparse networks under the same total parameter budget?

## Method

The method starts from SynFlow scores, predicts cutsets under ordinary global thresholding, then reserves a fixed fraction of the keep budget across critical layers proportional to output-unit count.

Critical layers:

- `conv2`
- `conv3`
- `fc1`
- `fc2`

Dense classifier layers use magnitude for within-layer ranking, because prior experiments showed SynFlow's dense-bridge ranking is poor. Remaining budget is filled by SynFlow saliency.

This is still not a final algorithm, but it is less manually tuned than the earlier multi-cut floor method.

## Setup

- Dataset: real CIFAR-10 images.
- Model: small CNN with dense classifier tail.
- Train subset: `20k` images.
- Test subset: `5k` images.
- Seeds: `151, 152, 153, 154`.
- Device: CUDA.
- Sparsities: `98%`, `99%`.
- Reserve fraction: `0.55` of the keep budget distributed across critical cuts by output-unit count.

## Result

| Sparsity | Method | Before FT | After FT | fc1 keep | Dead fc1 | conv2/conv3/fc2 keep |
|---:|---|---:|---:|---:|---:|---|
| `0.98` | `magnitude` | `0.1535` | `0.4411` | `0.0084` | `81.0` | `0.077/0.032/0.149` |
| `0.98` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.124/0.007/0.898` |
| `0.98` | `adaptive_capacity` | `0.1167` | `0.4170` | `0.0087` | `0.0` | `0.030/0.015/0.780` |
| `0.99` | `magnitude` | `0.1334` | `0.3283` | `0.0037` | `88.0` | `0.034/0.014/0.074` |
| `0.99` | `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.009/0.000/0.818` |
| `0.99` | `adaptive_capacity` | `0.1175` | `0.3465` | `0.0046` | `0.0` | `0.016/0.008/0.249` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `global_synflow` | `-5.59` pts | `0/4` | `-34.35` pts | `0/4` |
| `0.98` | `adaptive_capacity` | `-3.68` pts | `0/4` | `-2.41` pts | `0/4` |
| `0.99` | `global_synflow` | `-3.58` pts | `0/4` | `-23.06` pts | `0/4` |
| `0.99` | `adaptive_capacity` | `-1.59` pts | `0/4` | `+1.82` pts | `2/4` |

## Interpretation

Adaptive capacity pruning is now the strongest constructive result in the repo.

It does three things that the original pathology finding did not:

1. Predicts cutset risk from score distributions.
2. Uses circuit-viability constraints to prevent dense-bridge death.
3. Converts global SynFlow from chance-level collapse to a trainable sparse network under the same parameter budget.

At `99%`, it also beats magnitude after fine-tuning on the four-seed mean. At `98%`, it still trails magnitude, so the method is not generally dominant.

## Why this matters

This supports the stronger neuro-AI thesis:

**Pruning should preserve circuits, not just synapses.**

The method acts like a crude homeostatic constraint: it lets synapse-level saliency prune aggressively, but prevents entire communication cuts from going silent.

## Remaining limitations

- The reserve fraction is still a hyperparameter.
- The method has only been tested on a small CIFAR-10 CNN.
- It has not been compared to SNIP, GraSP, ERK, or modern pruning baselines.
- It does not yet explain the optimal capacity allocation theoretically.
- It does not beat magnitude at `98%`.

## Next step

Replace the fixed reserve fraction with a score-distribution-derived cut capacity target. The method should estimate how much capacity each cut needs from:

- layer score concentration;
- expected dead output units;
- activation-supported route usage;
- class reachability;
- post-prune min-cut capacity.

Artifact: `results/04_criticality_pruning/cifar10_cnn_adaptive_capacity_pruning.json`.
