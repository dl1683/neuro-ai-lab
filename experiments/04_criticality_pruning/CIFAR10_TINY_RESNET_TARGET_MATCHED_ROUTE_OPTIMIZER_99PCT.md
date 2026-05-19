# CIFAR-10 TinyResNet Target-Matched Route Optimizer at 99%

Fresh four-seed TinyResNet 99% target-matched route-family split optimizer.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[211, 212, 213, 214]`

## Chosen splits

| Seed | Main | Projection | Readout | Pre-FT target loss |
|---:|---:|---:|---:|---:|
| `211` | `0.20` | `0.55` | `0.25` | `0.1112` |
| `212` | `0.20` | `0.55` | `0.25` | `0.1067` |
| `213` | `0.20` | `0.55` | `0.25` | `0.1160` |
| `214` | `0.20` | `0.55` | `0.25` | `0.1232` |

## Results

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2579` | `0.0056` | baseline | baseline | `0.0000` | `1.8181` | `2.7815` | `219.2` |
| `reserve_0.60` | `0.1874` | `0.0240` | `-0.0705` | `0/4` | `1.0112` | `1.0029` | `1.7184` | `2.0` |
| `tuned_40_35_25` | `0.2443` | `0.0039` | `-0.0136` | `0/4` | `0.9131` | `1.0270` | `2.5357` | `2.0` |
| `fixed_deficit_predictor` | `0.2359` | `0.0080` | `-0.0221` | `0/4` | `0.9131` | `1.0485` | `2.4839` | `2.0` |
| `target_matched_optimizer` | `0.2452` | `0.0203` | `-0.0127` | `1/4` | `0.9131` | `1.3017` | `2.5337` | `8.0` |

## Interpretation

This is the first optimizer-style residual route allocator. It searches route-family splits before fine-tuning and selects the mask that best matches projection/readout targets from magnitude while preserving the plain-reserve main-path floor.
