# CIFAR-10 CNN Capacity Reserve Sweep at 99%

This experiment tests whether the path-capacity reserve is a brittle magic number or a broad useful regime.

## Question

Previous fixed-capacity runs found that reserving about `55%` of the keep budget for critical cuts can beat magnitude at `99%` sparsity. This sweep tests reserve fractions from `0.40` to `0.65` on fresh seeds.

## Setup

- Dataset: real CIFAR-10 images.
- Model: small CNN with dense classifier tail.
- Train subset: `20k` images.
- Test subset: `5k` images.
- Seeds: `181, 182, 183, 184`.
- Sparsity: `99%`.
- Device: CUDA.
- Same global parameter budget for all methods.

## Result

| Method | Before FT | After FT | After std | Dead fc1 | fc1 keep | conv2/conv3/fc2 keep |
|---|---:|---:|---:|---:|---:|---|
| `magnitude` | `0.1180` | `0.3232` | `0.0051` | `77.8` | `0.0039` | `0.036/0.012/0.079` |
| `global_synflow` | `0.0976` | `0.0976` | `0.0000` | `192.0` | `0.0000` | `0.009/0.000/0.824` |
| `reserve_0.40` | `0.1032` | `0.3218` | `0.0227` | `0.0` | `0.0034` | `0.012/0.006/0.404` |
| `reserve_0.45` | `0.1046` | `0.3406` | `0.0127` | `0.0` | `0.0038` | `0.013/0.007/0.353` |
| `reserve_0.50` | `0.1059` | `0.3434` | `0.0130` | `0.0` | `0.0042` | `0.014/0.007/0.301` |
| `reserve_0.55` | `0.1093` | `0.3427` | `0.0193` | `0.0` | `0.0045` | `0.016/0.008/0.251` |
| `reserve_0.60` | `0.1087` | `0.3488` | `0.0186` | `0.0` | `0.0049` | `0.017/0.009/0.203` |
| `reserve_0.65` | `0.1111` | `0.3394` | `0.0167` | `0.0` | `0.0053` | `0.018/0.009/0.158` |

## Paired deltas vs magnitude

| Method | Before delta | Before wins | After delta | After wins |
|---|---:|---:|---:|---:|
| `global_synflow` | `-2.04` pts | `0/4` | `-22.55` pts | `0/4` |
| `reserve_0.40` | `-1.48` pts | `1/4` | `-0.14` pts | `1/4` |
| `reserve_0.45` | `-1.34` pts | `0/4` | `+1.74` pts | `3/4` |
| `reserve_0.50` | `-1.21` pts | `1/4` | `+2.03` pts | `4/4` |
| `reserve_0.55` | `-0.87` pts | `1/4` | `+1.95` pts | `3/4` |
| `reserve_0.60` | `-0.93` pts | `1/4` | `+2.56` pts | `4/4` |
| `reserve_0.65` | `-0.70` pts | `1/4` | `+1.62` pts | `3/4` |

## Interpretation

This is the strongest support so far for Path-Capacity Pruning.

The result is not a single magic reserve value. A broad reserve band from `0.45` through `0.65` beats magnitude on mean after fine-tuning. The best value in this run, `0.60`, beats magnitude by `+2.56` points and wins `4/4` paired seeds.

This suggests a real severe-sparsity regime:

- too little reserved capacity (`0.40`) only prevents bridge death and does not reliably improve recovery;
- a middle reserve band (`0.45-0.65`) preserves enough circuit capacity to improve recovery;
- too much reserve starts stealing budget from useful saliency-selected routes, visible in the drop from `0.60` to `0.65`.

The next theoretical target is to predict the optimal reserve band from route-quality features, not to tune it by sweep.

Artifact: `results/04_criticality_pruning/cifar10_cnn_capacity_reserve_sweep_99pct.json`.
