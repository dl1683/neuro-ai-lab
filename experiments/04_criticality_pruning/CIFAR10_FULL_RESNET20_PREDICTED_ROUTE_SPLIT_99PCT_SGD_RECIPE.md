# CIFAR-10 Full ResNet-20 Predicted Route Split at 99%: SGD Recipe

Full CIFAR-10 train/test ResNet-20-style conservative predicted route split at 99% sparsity using pre-finetune route-quality deficits.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[261, 262]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4209` | `0.0033` |  |  | `0.0000` | `1.3671` | `3.7406` | `352.0` |
| `plain_reserve` | `0.4788` | `0.0060` | `+0.0578` | `2/2` | `1.2302` | `1.1929` | `3.3268` | `1.0` |
| `predicted_route_split` | `0.4692` | `0.0020` | `+0.0482` | `2/2` | `1.0115` | `1.5676` | `3.6403` | `0.0` |

## Chosen splits

- seed `261`: `{'main': 0.5499999999999999, 'projection': 0.25000000000000006, 'readout': 0.2}` score `1.6240`
- seed `262`: `{'main': 0.5499999999999999, 'projection': 0.20000000000000004, 'readout': 0.25}` score `1.6048`

## Interpretation

This applies the same conservative pre-finetune route-deficit selector used for CIFAR-100 back to CIFAR-10. A true ecology-aware viability rule should not force the same split everywhere; it should choose from route deficits before recovery.
