# CIFAR-10 TinyResNet Route Split Sweep at 99%

Four-seed targeted TinyResNet 99% projection/readout split sweep for balanced route-capacity pruning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[203, 204, 205, 206]`

| Method | After FT | After std | Delta vs magnitude | Wins | Route min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2489` | `0.0129` | baseline | baseline | `0.0009` | `1.7857` | `2.7890` | `221.0` |
| `reserve_0.60` | `0.2053` | `0.0188` | `-0.0435` | `0/4` | `1.0423` | `1.0143` | `1.6156` | `3.8` |
| `balanced_50_25_25` | `0.2330` | `0.0127` | `-0.0158` | `1/4` | `0.9378` | `0.9005` | `2.5700` | `2.8` |
| `proj_heavy_45_35_20` | `0.2422` | `0.0060` | `-0.0067` | `1/4` | `0.9646` | `1.0437` | `2.3946` | `3.2` |
| `proj_readout_40_35_25` | `0.2563` | `0.0097` | `+0.0074` | `3/4` | `0.9646` | `1.0437` | `2.5376` | `4.0` |

## Interpretation

This is a targeted projection/readout tradeoff test. The previous balanced split improved readout but still trailed magnitude, so this sweep increases projection share while preserving readout capacity.
