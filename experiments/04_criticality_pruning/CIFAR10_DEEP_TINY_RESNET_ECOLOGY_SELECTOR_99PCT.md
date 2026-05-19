# CIFAR-10 DeepTinyResNet Ecology-Aware Selector at 99%

CIFAR-10 subset DeepTinyResNet architecture-transfer validation of the ecology-aware pre-finetune selector at 99% sparsity.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[267, 268]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2721` | `0.0143` |  |  | `0.0002` | `2.3004` | `3.1649` | `311.0` |
| `plain_reserve` | `0.3150` | `0.0090` | `+0.0429` | `2/2` | `1.1590` | `1.1088` | `3.2669` | `1.0` |
| `predicted_route_split` | `0.3005` | `0.0015` | `+0.0284` | `2/2` | `0.9973` | `1.3197` | `3.5900` | `0.0` |
| `ecology_selected` | `0.2990` | `0.0106` | `+0.0269` | `2/2` | `1.1590` | `1.1088` | `3.2669` | `1.0` |

## Decisions

- seed `267`: selected `plain_reserve` readout_ratio `1.0088` split `None`
- seed `268`: selected `plain_reserve` readout_ratio `1.0561` split `None`

## Interpretation

This tests whether the ecology-aware selector transfers to a deeper residual architecture without changing the readout-ratio threshold. The dataset is still the existing CIFAR-10 subset harness, so this is an architecture-transfer check rather than a full benchmark.
