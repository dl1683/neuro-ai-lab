# TinyImageNet-200 Pretrained ResNet-18 Ecology Selector at 95%

TinyImageNet-200 external-proxy subset stress test using an ImageNet-pretrained ResNet-18 adapted to 200 classes, 95% sparsity, and the fixed ecology-aware selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seed: `273`
Train subset: `12000`; validation subset: `3000`
Dense epochs: `4`; masked fine-tune epochs: `2`
Dense accuracy: `0.6077`

| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1523` |  | `3.6182` | `3.1166` | `5.3203` | `10.0` |
| `plain_reserve` | `0.0397` | `-0.1127` | `4.2285` | `4.1744` | `5.5299` | `0.0` |
| `predicted_route_split` | `0.0353` | `-0.1170` | `3.8842` | `4.1744` | `6.0488` | `0.0` |
| `ecology_policy` | `0.0397` | `-0.1127` | `4.2285` | `4.1744` | `5.5299` | `0.0` |

## Decision

Selected method: `plain_reserve`
Readout ratio: `1.0394`
Selected split: `None`

## Interpretation

This repeats the pretrained ResNet-18 TinyImageNet proxy at 95% sparsity to test whether the 99% external failure is a hard sparsity cliff rather than a selector-only failure.
