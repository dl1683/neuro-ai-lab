# CIFAR-10 CNN Risk-Adaptive Capacity Pruning

This experiment tests whether the fixed capacity reserve can be replaced by a risk-adaptive allocator.

## Question

Can predicted dead-cut risk from the global SynFlow score distribution determine how much capacity each critical cut should receive?

## Method

The risk-adaptive method computes global SynFlow cutset predictions before pruning. It estimates each critical layer's dead-output fraction, then:

- sets reserve fraction as `0.30 + 0.40 * mean_dead_cut_risk`, clipped to `[0.30, 0.70]`;
- allocates reserve budget across critical layers by `output_units * (0.25 + dead_fraction)`;
- fills remaining budget by SynFlow saliency;
- preserves the same total parameter budget as ordinary global pruning.

It is compared against the prior fixed-capacity allocator with reserve fraction `0.55`.

## Result

| Sparsity | Method | Reserve | Before FT | After FT | fc1 keep | Dead fc1 | conv2/conv3/fc2 keep |
|---:|---|---:|---:|---:|---:|---:|---|
| `0.98` | `magnitude` |  | `0.1263` | `0.4370` | `0.0092` | `74.2` | `0.072/0.029/0.153` |
| `0.98` | `global_synflow` |  | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.125/0.007/0.896` |
| `0.98` | `fixed_capacity` | `0.550` | `0.1247` | `0.4155` | `0.0087` | `0.0` | `0.030/0.015/0.780` |
| `0.98` | `risk_adaptive_capacity` | `0.441` | `0.1154` | `0.3949` | `0.0095` | `0.0` | `0.025/0.009/0.850` |
| `0.99` | `magnitude` |  | `0.1292` | `0.3261` | `0.0041` | `78.8` | `0.033/0.012/0.081` |
| `0.99` | `global_synflow` |  | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.009/0.000/0.822` |
| `0.99` | `fixed_capacity` | `0.550` | `0.1167` | `0.3317` | `0.0045` | `0.0` | `0.016/0.008/0.248` |
| `0.99` | `risk_adaptive_capacity` | `0.535` | `0.1179` | `0.3222` | `0.0050` | `0.0` | `0.010/0.009/0.246` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `fixed_capacity` | `-0.0016` | `2/4` | `-0.0216` | `0/4` |
| `0.98` | `risk_adaptive_capacity` | `-0.0109` | `1/4` | `-0.0421` | `0/4` |
| `0.99` | `fixed_capacity` | `-0.0124` | `1/4` | `+0.0056` | `3/4` |
| `0.99` | `risk_adaptive_capacity` | `-0.0113` | `1/4` | `-0.0039` | `2/4` |

## Interpretation

This is a useful negative result.

Risk-adaptive allocation prevented dense bridge death, but it did not improve over the fixed `0.55` reserve. At `98%`, it under-reserved capacity and fell further behind magnitude. At `99%`, it lost the small fixed-capacity edge over magnitude.

The failure clarifies the theory:

- Dead-cut risk is necessary but not sufficient.
- The allocator must account for saliency mass concentration and recovery quality, not just dead output units.
- Capacity has to preserve useful route quality, not merely output liveness.

## Next hypothesis

A better predictor should combine:

1. dead-output risk;
2. score mass concentration per cut;
3. minimum fan-in/fan-out;
4. activation-supported route usage;
5. expected recovery capacity after masked fine-tuning.

Artifact: `results/04_criticality_pruning/cifar10_cnn_risk_adaptive_capacity_pruning.json`.
