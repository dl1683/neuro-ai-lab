# CIFAR Ecology-Aware Circuit-Viability Selector at 99%

Fresh CIFAR-10 and CIFAR-100 full-dataset ResNet-20-style validation of an ecology-aware pre-finetune selector that chooses broad reserve or conservative route split from readout deficit.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

## cifar10

Seeds: `[263, 264]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.4393` | `0.0094` |  |  | `0.0000` | `1.3742` | `3.6887` | `340.5` |
| `plain_reserve` | `0.4625` | `0.0056` | `+0.0232` | `2/2` | `1.2302` | `1.1858` | `3.3212` | `0.5` |
| `predicted_route_split` | `0.4547` | `0.0009` | `+0.0154` | `2/2` | `1.0340` | `1.6523` | `3.5293` | `0.5` |
| `ecology_selected` | `0.4674` | `0.0073` | `+0.0281` | `2/2` | `1.2302` | `1.1858` | `3.3212` | `0.5` |

Selections:

- seed `263`: selected `plain_reserve` readout_ratio `0.8994` split `None`
- seed `264`: selected `plain_reserve` readout_ratio `0.9014` split `None`

## cifar100

Seeds: `[265, 266]`

| Method | After FT | After std | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0541` | `0.0001` |  |  | `0.0000` | `0.4681` | `3.0390` | `540.0` |
| `plain_reserve` | `0.0726` | `0.0042` | `+0.0185` | `2/2` | `1.2000` | `1.1344` | `1.0919` | `1.0` |
| `predicted_route_split` | `0.0915` | `0.0012` | `+0.0375` | `2/2` | `0.9973` | `1.1665` | `2.0334` | `1.0` |
| `ecology_selected` | `0.0908` | `0.0021` | `+0.0367` | `2/2` | `0.9973` | `1.1665` | `2.0334` | `1.0` |

Selections:

- seed `265`: selected `predicted_route_split` readout_ratio `0.3613` split `{'main': 0.49999999999999994, 'projection': 0.1, 'readout': 0.4}`
- seed `266`: selected `predicted_route_split` readout_ratio `0.3573` split `{'main': 0.44999999999999996, 'projection': 0.15000000000000002, 'readout': 0.4}`

## Interpretation

The selector first measures the plain-reserve readout ratio against the magnitude readout template. If plain reserve has a large readout deficit, it uses the conservative predicted route split; otherwise it keeps broad reserve. This tests whether task ecology can choose the intervention family before fine-tuning.
