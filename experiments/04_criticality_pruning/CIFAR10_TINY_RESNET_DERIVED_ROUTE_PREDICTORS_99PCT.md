# CIFAR-10 TinyResNet Derived Route Predictors at 99%

Fresh four-seed TinyResNet 99% test of derived route-family split predictors.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[207, 208, 209, 210]`

## Predicted splits

| Seed | Predictor | Main | Projection | Readout |
|---:|---|---:|---:|---:|
| `207` | `tuned_40_35_25` | `0.400` | `0.350` | `0.250` |
| `207` | `fixed_deficit_predictor` | `0.400` | `0.378` | `0.222` |
| `207` | `relative_deficit_predictor` | `0.264` | `0.381` | `0.355` |
| `207` | `sqrt_width_deficit_predictor` | `0.449` | `0.424` | `0.127` |
| `208` | `tuned_40_35_25` | `0.400` | `0.350` | `0.250` |
| `208` | `fixed_deficit_predictor` | `0.400` | `0.336` | `0.264` |
| `208` | `relative_deficit_predictor` | `0.257` | `0.370` | `0.372` |
| `208` | `sqrt_width_deficit_predictor` | `0.445` | `0.419` | `0.136` |
| `209` | `tuned_40_35_25` | `0.400` | `0.350` | `0.250` |
| `209` | `fixed_deficit_predictor` | `0.400` | `0.387` | `0.213` |
| `209` | `relative_deficit_predictor` | `0.268` | `0.384` | `0.348` |
| `209` | `sqrt_width_deficit_predictor` | `0.452` | `0.424` | `0.124` |
| `210` | `tuned_40_35_25` | `0.400` | `0.350` | `0.250` |
| `210` | `fixed_deficit_predictor` | `0.400` | `0.351` | `0.249` |
| `210` | `relative_deficit_predictor` | `0.260` | `0.375` | `0.365` |
| `210` | `sqrt_width_deficit_predictor` | `0.446` | `0.421` | `0.133` |

## Results

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2498` | `0.0052` | baseline | baseline | `0.0000` | `1.7922` | `2.7803` | `220.0` |
| `reserve_0.60` | `0.2070` | `0.0060` | `-0.0428` | `0/4` | `1.0255` | `1.0030` | `1.7398` | `3.0` |
| `tuned_40_35_25` | `0.2461` | `0.0252` | `-0.0037` | `2/4` | `0.9097` | `1.0437` | `2.5415` | `3.0` |
| `fixed_deficit_predictor` | `0.2459` | `0.0223` | `-0.0039` | `2/4` | `0.9097` | `1.0609` | `2.5015` | `3.0` |
| `relative_deficit_predictor` | `0.2381` | `0.0068` | `-0.0117` | `1/4` | `0.9097` | `1.0840` | `2.8747` | `6.5` |
| `sqrt_width_deficit_predictor` | `0.2275` | `0.0168` | `-0.0223` | `0/4` | `0.9097` | `1.1456` | `2.1637` | `3.0` |

## Interpretation

This experiment removes or weakens the hand-set constants in the first route-deficit predictor. `relative_deficit_predictor` uses equal route-family priors. `sqrt_width_deficit_predictor` derives the family prior from route-family output width. Both select the split before recovery fine-tuning.
