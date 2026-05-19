# CIFAR-10 DeepTinyResNet Diversity Route Optimizer 99% Replicate

Fresh four-seed DeepTinyResNet 99% replicate for diversity-penalized route-capacity pruning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[221, 222, 223, 224]`

## Chosen splits

| Seed | Main | Projection | Readout | Pre-FT loss |
|---:|---:|---:|---:|---:|
| `221` | `0.30` | `0.50` | `0.20` | `0.2210` |
| `222` | `0.25` | `0.50` | `0.25` | `0.2307` |
| `223` | `0.30` | `0.50` | `0.20` | `0.2146` |
| `224` | `0.25` | `0.50` | `0.25` | `0.2146` |

## Results

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2848` | `0.0043` | baseline | baseline | `0.0035` | `2.3009` | `3.1814` | `298.2` |
| `global_synflow` | `0.0998` | `0.0009` | `-0.1850` | `0/4` | `0.0000` | `0.0000` | `3.8569` | `479.5` |
| `reserve_0.60` | `0.3020` | `0.0310` | `+0.0172` | `3/4` | `1.1139` | `1.1242` | `3.3118` | `0.2` |
| `tuned_40_35_25` | `0.2989` | `0.0182` | `+0.0141` | `3/4` | `0.8826` | `1.5712` | `3.5798` | `0.2` |
| `diversity_target_optimizer` | `0.2883` | `0.0268` | `+0.0035` | `1/4` | `0.8826` | `1.8621` | `3.3326` | `0.2` |

## Interpretation

This is a fresh four-seed replicate for the deeper residual transfer result.
