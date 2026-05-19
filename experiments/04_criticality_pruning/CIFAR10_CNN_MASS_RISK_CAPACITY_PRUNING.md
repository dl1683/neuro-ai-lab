# CIFAR-10 CNN Mass-Risk Capacity Pruning

This experiment tests whether score-mass concentration improves risk-adaptive path-capacity allocation.

## Question

The previous risk-adaptive allocator used dead-output risk only and underperformed the fixed `0.55` reserve. This experiment adds score-mass deficit: if a critical cut receives less keep-rate than the global sparsity target under ordinary SynFlow thresholding, it receives more protected capacity.

## Result

| Sparsity | Method | Reserve | Before FT | After FT | fc1 keep | Dead fc1 | conv2/conv3/fc2 keep |
|---:|---|---:|---:|---:|---:|---:|---|
| `0.98` | `magnitude` |  | `0.1850` | `0.4365` | `0.0087` | `87.2` | `0.078/0.030/0.140` |
| `0.98` | `global_synflow` |  | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.125/0.007/0.898` |
| `0.98` | `fixed_capacity` | `0.550` | `0.1458` | `0.4186` | `0.0087` | `0.0` | `0.030/0.015/0.775` |
| `0.98` | `mass_risk_capacity` | `0.611` | `0.1472` | `0.3905` | `0.0125` | `0.0` | `0.012/0.014/0.609` |
| `0.99` | `magnitude` |  | `0.1306` | `0.3177` | `0.0038` | `92.5` | `0.036/0.013/0.071` |
| `0.99` | `global_synflow` |  | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.009/0.000/0.823` |
| `0.99` | `fixed_capacity` | `0.550` | `0.1338` | `0.3409` | `0.0046` | `0.0` | `0.016/0.008/0.242` |
| `0.99` | `mass_risk_capacity` | `0.656` | `0.1339` | `0.3291` | `0.0061` | `0.0` | `0.011/0.010/0.132` |

## Paired deltas vs magnitude

| Sparsity | Method | Before delta | Before wins | After delta | After wins |
|---:|---|---:|---:|---:|---:|
| `0.98` | `fixed_capacity` | `-3.92` pts | `0/4` | `-1.79` pts | `0/4` |
| `0.98` | `mass_risk_capacity` | `-3.78` pts | `0/4` | `-4.60` pts | `0/4` |
| `0.99` | `fixed_capacity` | `+0.33` pts | `2/4` | `+2.32` pts | `4/4` |
| `0.99` | `mass_risk_capacity` | `+0.33` pts | `1/4` | `+1.14` pts | `3/4` |

## Interpretation

Mass-risk allocation did not beat the fixed reserve. It over-allocated protected capacity, especially to `fc1`, and appears to steal too much budget from useful saliency-selected routes.

However, this run strengthens the constructive path-capacity result:

- fixed capacity at `99%` beats magnitude after fine-tuning by `+2.32` points;
- it wins `4/4` seeds;
- it prevents dense-bridge death under the same parameter budget;
- it converts global SynFlow from chance-level collapse into a trainable sparse circuit.

The best current algorithmic prior is therefore not risk-proportional allocation. It is a stable capacity reserve that prevents cutsets without overreacting to vulnerability.

Artifact: `results/04_criticality_pruning/cifar10_cnn_mass_risk_capacity_pruning.json`.
