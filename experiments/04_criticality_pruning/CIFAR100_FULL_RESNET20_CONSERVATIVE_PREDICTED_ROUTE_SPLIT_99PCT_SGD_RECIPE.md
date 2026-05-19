# CIFAR-100 Full ResNet-20 Conservative Predicted Route Split at 99%: SGD Recipe

Full CIFAR-100 train/test ResNet-20-style conservative predicted route split at 99% sparsity using pre-finetune route-quality deficits.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[257, 258]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0698` | `0.0070` |  |  | `0.0000` | `0.5409` | `3.0440` | `541.5` |
| `plain_reserve` | `0.0768` | `0.0056` | `+0.0071` | `2/2` | `1.1977` | `1.1543` | `1.0953` | `0.5` |
| `predicted_route_split` | `0.0878` | `0.0054` | `+0.0181` | `2/2` | `1.0058` | `1.0506` | `2.0334` | `0.5` |

## Chosen splits

- seed `257`: `{'main': 0.49999999999999994, 'projection': 0.1, 'readout': 0.4}` score `1.5780`
- seed `258`: `{'main': 0.49999999999999994, 'projection': 0.1, 'readout': 0.4}` score `1.5860`

## Interpretation

This follow-up tightens the automatic selector after the first predictor over-weighted readout on one seed. It requires a stronger main-path floor before spending capacity on the CIFAR-100 readout.
