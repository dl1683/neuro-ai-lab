# CIFAR-10 TinyResNet Predicted Route Split at 99%

Four-seed TinyResNet 99% test of route-deficit-predicted capacity splits.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[203, 204, 205, 206]`

## Predicted splits

| Seed | Main | Projection | Readout | Projection deficit | Readout deficit |
|---:|---:|---:|---:|---:|---:|
| `203` | `0.400` | `0.350` | `0.250` | `0.8083` | `1.1569` |
| `204` | `0.400` | `0.346` | `0.254` | `0.7603` | `1.1177` |
| `205` | `0.400` | `0.344` | `0.256` | `0.7290` | `1.0815` |
| `206` | `0.400` | `0.340` | `0.260` | `0.7994` | `1.2188` |

## Results

| Method | After FT | After std | Delta vs magnitude | Wins | Route min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2504` | `0.0131` | baseline | baseline | `0.0009` | `1.7857` | `2.7889` | `220.8` |
| `reserve_0.60` | `0.2021` | `0.0139` | `-0.0483` | `0/4` | `1.0586` | `1.0114` | `1.6452` | `3.5` |
| `tuned_40_35_25` | `0.2487` | `0.0143` | `-0.0017` | `3/4` | `0.9767` | `1.0492` | `2.5357` | `4.0` |
| `predicted_deficit_split` | `0.2512` | `0.0056` | `+0.0008` | `2/4` | `0.9767` | `1.0409` | `2.5533` | `4.0` |

## Interpretation

This tests whether the route-family split can be derived from pre-finetune route deficits instead of swept by hand. The split compares a magnitude viability template to the plain reserve candidate, then allocates protected capacity across projection and readout deficits with a main-path floor.
