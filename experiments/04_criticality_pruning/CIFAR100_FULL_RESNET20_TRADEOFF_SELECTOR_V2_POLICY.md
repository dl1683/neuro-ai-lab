# CIFAR-100 Full ResNet-20 Tradeoff Selector V2 Policy

Policy projection over the fresh CIFAR-100 tradeoff-selector candidate run. V2 lowers feature-overlap weight under high readout/output-diversity pressure.

| Method | After FT | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0698` |  |  | `0.0000` | `0.6559` | `3.0585` | `557.0` |
| `feature_viability_repair` | `0.0821` | `+0.0123` | `2/2` | `0.5745` | `0.5575` | `2.8939` | `61.5` |
| `plain_reserve` | `0.0877` | `+0.0180` | `2/2` | `1.1977` | `1.1190` | `1.0936` | `0.5` |
| `predicted_route_split` | `0.0887` | `+0.0189` | `2/2` | `1.0059` | `1.2900` | `2.0334` | `1.0` |
| `tradeoff_policy` | `0.0821` | `+0.0123` | `2/2` | `0.5745` | `0.5575` | `2.8939` | `61.5` |
| `tradeoff_v2_policy` | `0.0887` | `+0.0189` | `2/2` | `1.0059` | `1.2900` | `2.0334` | `1.0` |

## Decisions

- seed `282`: v1 `feature_viability_repair` -> v2 `predicted_route_split`; scores predicted_route_split=0.763, feature_viability_repair=0.716, plain_reserve=0.624, magnitude=0.564
- seed `283`: v1 `feature_viability_repair` -> v2 `predicted_route_split`; scores predicted_route_split=0.753, feature_viability_repair=0.746, plain_reserve=0.635, magnitude=0.562

## Interpretation

This is a policy projection, not a new training run. The candidate masks and post-finetune outcomes come from the fresh CIFAR-100 tradeoff experiment. The V2 selector changes only the pre-finetune selection rule, reducing feature-overlap weight when the magnitude mask shows high output/readout pressure.
