# CIFAR-100 Full ResNet-20 Route Split at 99%: SGD Recipe

Full CIFAR-100 train/test ResNet-20-style route-family split test at 99% sparsity using the 20 dense SGD/cosine plus 5 masked fine-tune recipe.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[253, 254]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0653` | `0.0001` |  |  | `0.0000` | `0.5785` | `3.0558` | `548.0` |
| `plain_reserve` | `0.0798` | `0.0010` | `+0.0145` | `2/2` | `1.2070` | `1.1394` | `1.0886` | `0.0` |
| `balanced_40_35_25` | `0.0875` | `0.0010` | `+0.0222` | `2/2` | `0.9531` | `1.9526` | `1.6400` | `0.0` |
| `readout_heavy_35_20_45` | `0.0902` | `0.0007` | `+0.0249` | `2/2` | `0.9531` | `1.4954` | `2.1365` | `0.0` |
| `readout_main_45_15_40` | `0.0912` | `0.0042` | `+0.0259` | `2/2` | `0.9828` | `1.2857` | `2.0334` | `0.0` |

## Interpretation

This tests whether CIFAR-100 needs a different route-family allocation than CIFAR-10. The hypothesis is that a 100-class output space should make readout preservation more important than the plain reserve rule captures.
