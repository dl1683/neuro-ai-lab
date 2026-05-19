# CIFAR-10 Full ResNet-20 Capacity at 99%: Six-Seed Aggregate

Full CIFAR-10 train/test ResNet-20-style path-capacity pruning at 99% sparsity, aggregating the original two-seed run with four fresh independent seeds.

Sources: `cifar10_full_resnet20_capacity_99pct.json`, `cifar10_full_resnet20_capacity_99pct_moreseeds.json`
Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[231, 232, 233, 234, 235, 236]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.3772` | `0.0137` |  |  | `0.0560` | `2.2744` | `3.1560` | `192.8` |
| `global_synflow` | `0.1000` | `0.0000` | `-0.2772` | `0/6` | `0.0000` | `0.0000` | `3.8770` | `664.8` |
| `reserve_0.60` | `0.3921` | `0.0050` | `+0.0150` | `6/6` | `1.1976` | `1.1754` | `3.6652` | `0.0` |
| `tuned_40_35_25` | `0.3806` | `0.0123` | `+0.0034` | `3/6` | `0.9026` | `1.9347` | `3.7456` | `0.0` |

## Interpretation

Across six full CIFAR-10 ResNet-20-style seeds at 99% sparsity, the broad capacity reserve remains positive and wins every paired seed against magnitude. The tuned route split is positive on average but less robust, indicating that the current strongest public claim is homeostatic circuit viability rather than hand-tuned route allocation.
