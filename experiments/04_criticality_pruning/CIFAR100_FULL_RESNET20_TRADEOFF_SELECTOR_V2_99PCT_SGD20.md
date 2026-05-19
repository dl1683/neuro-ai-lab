# CIFAR-100 Full ResNet-20 Tradeoff Selector V2 at 99%

Fresh prospective full-CIFAR-100 ResNet-20 99% sparsity validation of the V2 feature-preservation / liveness tradeoff selector with task-ecology pressure.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[284, 285]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0654` | `0.0013` |  |  | `0.0000` | `0.5880` | `3.0572` | `547.0` |
| `feature_viability_repair` | `0.0887` | `0.0004` | `+0.0233` | `2/2` | `0.4843` | `0.5748` | `2.8954` | `57.0` |
| `plain_reserve` | `0.0830` | `0.0021` | `+0.0177` | `2/2` | `1.2048` | `1.1341` | `1.0750` | `0.0` |
| `predicted_route_split` | `0.0926` | `0.0013` | `+0.0272` | `2/2` | `1.0313` | `1.0451` | `2.0334` | `0.0` |
| `tradeoff_policy` | `0.0926` | `0.0013` | `+0.0272` | `2/2` | `1.0313` | `1.0451` | `2.0334` | `0.0` |

## Selector decisions

- seed `284`: selected `predicted_route_split`; scores predicted_route_split=0.751, feature_viability_repair=0.718, plain_reserve=0.669, magnitude=0.563
- seed `285`: selected `predicted_route_split`; scores predicted_route_split=0.765, feature_viability_repair=0.718, plain_reserve=0.683, magnitude=0.564

## Interpretation

This is the prospective validation that the V2 task-ecology pressure term was intended to enable. Unlike the V2 policy projection, this run retrains fresh dense models and selects the sparse method before masked fine-tuning.
