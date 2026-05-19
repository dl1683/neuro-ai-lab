# CIFAR-10 TinyResNet Diversity Route Optimizer at 99%

Fresh four-seed TinyResNet 99% diversity-penalized target-matched route optimizer.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[215, 216, 217, 218]`

## Chosen splits

| Seed | Main | Projection | Readout | Pre-FT loss |
|---:|---:|---:|---:|---:|
| `215` | `0.25` | `0.50` | `0.25` | `0.3023` |
| `216` | `0.25` | `0.50` | `0.25` | `0.2650` |
| `217` | `0.25` | `0.50` | `0.25` | `0.2968` |
| `218` | `0.25` | `0.50` | `0.25` | `0.2847` |

## Results

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2480` | `0.0055` | baseline | baseline | `0.0000` | `1.8143` | `2.7704` | `220.5` |
| `reserve_0.60` | `0.2013` | `0.0260` | `-0.0466` | `0/4` | `1.0257` | `0.9797` | `1.8210` | `2.8` |
| `tuned_40_35_25` | `0.2604` | `0.0172` | `+0.0125` | `3/4` | `0.9100` | `1.0312` | `2.5509` | `3.0` |
| `diversity_target_optimizer` | `0.2585` | `0.0023` | `+0.0105` | `4/4` | `0.9100` | `1.2358` | `2.5337` | `7.2` |

## Interpretation

This optimizer adds route-family concentration and projection-overuse penalties to the target-matching objective. It tests whether a degeneracy-inspired diversity constraint prevents projection overprotection.
