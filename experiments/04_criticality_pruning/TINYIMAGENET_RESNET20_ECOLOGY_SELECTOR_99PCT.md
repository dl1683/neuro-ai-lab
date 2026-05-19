# TinyImageNet-200 ResNet-20 Ecology Selector at 99%

TinyImageNet-200 external-proxy subset stress test using a ResNet-20-style 200-class model, 99% sparsity, and the fixed ecology-aware selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seed: `271`
Train subset: `20000`; validation subset: `5000`
Dense epochs: `12`; masked fine-tune epochs: `4`

| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0232` |  | `0.0000` | `1.1648` | `2.4574` | `622.0` |
| `plain_reserve` | `0.0308` | `+0.0076` | `1.1294` | `1.0881` | `1.1036` | `0.0` |
| `predicted_route_split` | `0.0290` | `+0.0058` | `0.9163` | `1.5110` | `1.6705` | `0.0` |
| `ecology_policy` | `0.0290` | `+0.0058` | `0.9163` | `1.5110` | `1.6705` | `0.0` |

## Decision

Selected method: `predicted_route_split`
Readout ratio: `0.4491`
Selected split: `{'main': 0.3, 'projection': 0.20000000000000004, 'readout': 0.4999999999999999}`

## Interpretation

This is a first external TinyImageNet-200 proxy, not a full publication benchmark. It tests whether the same fixed readout-ratio selector behaves coherently on a 200-class natural-image task outside CIFAR.
