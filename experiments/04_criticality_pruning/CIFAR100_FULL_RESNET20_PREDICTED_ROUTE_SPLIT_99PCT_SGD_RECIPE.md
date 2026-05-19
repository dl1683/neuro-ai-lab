# CIFAR-100 Full ResNet-20 Predicted Route Split at 99%: SGD Recipe

Full CIFAR-100 train/test ResNet-20-style predicted route split at 99% sparsity using pre-finetune route-quality deficits.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[255, 256]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0709` | `0.0006` |  |  | `0.0000` | `0.6473` | `3.0573` | `542.0` |
| `plain_reserve` | `0.0897` | `0.0030` | `+0.0188` | `2/2` | `1.1858` | `1.1519` | `1.0953` | `0.0` |
| `predicted_route_split` | `0.0980` | `0.0113` | `+0.0272` | `2/2` | `0.9471` | `1.4971` | `2.2300` | `1.5` |

## Chosen splits

- seed `255`: `{'main': 0.3, 'projection': 0.20000000000000004, 'readout': 0.4999999999999999}` score `1.5616`
- seed `256`: `{'main': 0.3, 'projection': 0.20000000000000004, 'readout': 0.4999999999999999}` score `1.5700`

## Interpretation

The split is selected before fine-tuning from route-quality deficits: preserve a main-path floor, avoid projection collapse, and raise readout capacity toward the magnitude readout template. This tests whether the CIFAR-100 readout result can be predicted rather than hand-picked.
