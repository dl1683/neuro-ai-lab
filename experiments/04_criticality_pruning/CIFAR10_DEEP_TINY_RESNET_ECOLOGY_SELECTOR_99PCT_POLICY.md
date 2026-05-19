# CIFAR-10 DeepTinyResNet Ecology Selector Policy Projection

Policy projection of the DeepTinyResNet ecology selector, using the already evaluated method chosen by the pre-finetune selector instead of duplicate stochastic fine-tuning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[267, 268]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.2721` | `0.0143` |  |  | `0.0002` | `2.3004` | `3.1649` | `311.0` |
| `plain_reserve` | `0.3150` | `0.0090` | `+0.0429` | `2/2` | `1.1590` | `1.1088` | `3.2669` | `1.0` |
| `predicted_route_split` | `0.3005` | `0.0015` | `+0.0284` | `2/2` | `0.9973` | `1.3197` | `3.5900` | `0.0` |
| `ecology_policy` | `0.3150` | `0.0090` | `+0.0429` | `2/2` | `1.1590` | `1.1088` | `3.2669` | `1.0` |

## Interpretation

The selector chooses plain reserve on both DeepTinyResNet seeds because the plain-reserve readout ratio is already above threshold. This projection removes duplicate fine-tune noise by assigning the selected-policy result to the already evaluated plain-reserve row.
