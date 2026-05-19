# TinyImageNet-200 Pretrained ResNet-18 Tradeoff Selector at 95%

TinyImageNet-200 pretrained ResNet-18 95% sparsity validation of the feature-preservation / liveness tradeoff selector.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seed: `281`
Train subset: `12000`; validation subset: `3000`
Dense epochs: `4`; masked fine-tune epochs: `2`
Dense accuracy: `0.6060`

| Method | After FT | Delta vs magnitude | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1567` |  | `3.6191` | `3.1152` | `5.3169` | `11.0` |
| `feature_viability_repair` | `0.1570` | `+0.0003` | `3.6190` | `3.1152` | `5.3169` | `0.0` |
| `plain_reserve` | `0.0380` | `-0.1187` | `4.2285` | `4.1744` | `5.5325` | `0.0` |
| `predicted_route_split` | `0.0300` | `-0.1267` | `3.8842` | `4.1744` | `6.0488` | `0.0` |
| `tradeoff_policy` | `0.1570` | `+0.0003` | `3.6190` | `3.1152` | `5.3169` | `0.0` |

## Selector decision

Selected method: `feature_viability_repair`
Scores: feature_viability_repair=0.988, magnitude=0.928, plain_reserve=0.823, predicted_route_split=0.800

## Interpretation

This is a pretrained/external validation of the feature-preservation versus liveness tradeoff selector. The selector is not allowed to use post-finetune accuracy; it ranks candidate masks from pre-finetune route-quality diagnostics and feature overlap with the magnitude template.
