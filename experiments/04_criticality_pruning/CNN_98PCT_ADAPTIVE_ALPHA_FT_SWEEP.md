# CNN 98% Adaptive Alpha Fine-Tuning Sweep

Fashion-MNIST CNN 98% adaptive dense-hybrid alpha sweep before and after 3-epoch masked fine-tuning.

| Alpha | Before FT | After FT | fc1 keep rate | Dead fc1 hidden |
|---:|---:|---:|---:|---:|
| `0.00` | `0.4455` | `0.8334` | `0.0100` | `36.0` |
| `0.03` | `0.4516` | `0.8374` | `0.0094` | `36.0` |
| `0.05` | `0.4557` | `0.8394` | `0.0089` | `36.5` |
| `0.08` | `0.4938` | `0.8365` | `0.0081` | `36.5` |
| `0.10` | `0.4930` | `0.8365` | `0.0076` | `36.5` |
| `0.15` | `0.5001` | `0.8293` | `0.0065` | `36.5` |
| `0.20` | `0.4974` | `0.8306` | `0.0055` | `38.0` |
| `0.30` | `0.4261` | `0.8274` | `0.0039` | `44.0` |
| `0.50` | `0.2774` | `0.8034` | `0.0021` | `69.0` |

Best one-shot alpha: `0.15` with `0.5001` before FT and `0.8293` after FT.
Best after-FT alpha: `0.05` with `0.8394` after FT.
Best balanced alpha: `0.08`.
