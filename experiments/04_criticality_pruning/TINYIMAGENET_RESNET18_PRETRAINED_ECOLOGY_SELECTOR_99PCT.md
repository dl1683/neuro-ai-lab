# TinyImageNet-200 Pretrained ResNet-18 Ecology Selector at 99%

TinyImageNet-200 external-proxy subset stress test using an ImageNet-pretrained ResNet-18 adapted to 200 classes, 99% sparsity, and the fixed ecology-aware selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seed: `272`
Train subset: `12000`; validation subset: `3000`
Dense epochs: `4`; masked fine-tune epochs: `2`
Dense accuracy: `0.5983`

| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0107` |  | `0.8478` | `1.8031` | `4.3134` | `366.0` |
| `plain_reserve` | `0.0097` | `-0.0010` | `2.6809` | `2.6768` | `2.6783` | `0.0` |
| `predicted_route_split` | `0.0080` | `-0.0027` | `2.3691` | `2.9891` | `4.2285` | `0.0` |
| `ecology_policy` | `0.0080` | `-0.0027` | `2.3691` | `2.9891` | `4.2285` | `0.0` |

## Decision

Selected method: `predicted_route_split`
Readout ratio: `0.6209`
Selected split: `{'main': 0.5499999999999999, 'projection': 0.25000000000000006, 'readout': 0.2}`

## Interpretation

This upgrades the TinyImageNet proxy to an ImageNet-pretrained ResNet-18 and keeps the selector threshold fixed. It tests whether the boundary condition was caused by weak dense training rather than the viability idea itself.
