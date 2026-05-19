# CIFAR-10 TinyResNet Route-Quality Audit

CIFAR-10 TinyResNet audit of route-quality metrics for residual path-capacity pruning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

## Summary by method

| Sparsity | Method | After FT | Route min | Strict route min | Route balance | Projection min | FC score | Dead outputs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `0.98` | `magnitude` | `0.2969` | `0.5495` | `0.0192` | `0.2463` | `2.2852` | `3.2994` | `131.0` |
| `0.98` | `global_synflow` | `0.1007` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `3.8959` | `275.5` |
| `0.98` | `reserve_0.60` | `0.3171` | `1.3979` | `1.3979` | `0.6778` | `1.3505` | `3.4594` | `1.0` |
| `0.98` | `activation_reserve_0.60` | `0.3298` | `1.3902` | `1.3902` | `0.6725` | `1.3424` | `3.4703` | `1.0` |
| `0.98` | `backbone_reserve_0.60` | `0.3141` | `1.0451` | `1.0395` | `0.4789` | `1.9324` | `3.4076` | `1.0` |
| `0.99` | `magnitude` | `0.2406` | `0.0019` | `0.0000` | `0.0011` | `1.8149` | `2.7081` | `219.0` |
| `0.99` | `global_synflow` | `0.1016` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `3.6648` | `296.0` |
| `0.99` | `reserve_0.60` | `0.2091` | `1.0228` | `1.0228` | `0.7757` | `0.9885` | `1.6988` | `3.0` |
| `0.99` | `activation_reserve_0.60` | `0.1918` | `1.0228` | `1.0228` | `0.7566` | `0.9763` | `1.8616` | `3.0` |
| `0.99` | `backbone_reserve_0.60` | `0.1730` | `0.9160` | `0.8910` | `0.7163` | `1.3824` | `1.1473` | `3.5` |

## Metric correlations with after-FT accuracy

| Scope | Route min | Strict route min | Route balance | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| `0.98` | `+0.852` | `+0.690` | `+0.839` | `+0.830` | `-0.911` | `-0.927` |
| `0.99` | `+0.215` | `+0.220` | `+0.213` | `+0.842` | `-0.433` | `-0.370` |
| `all` | `+0.652` | `+0.539` | `+0.405` | `+0.807` | `+0.148` | `-0.611` |

## Interpretation

This audit tests whether route-quality features explain recoverability better than total dead-output count. It treats each residual block as a composed route with a main path and, when present, a projection shortcut.

A useful next pruning method should optimize the route-quality metric that tracks after-FT accuracy, not merely keep every output unit alive.
