# CIFAR-10 CNN 98% Adaptive Alpha Fine-Tuning Sweep

CIFAR-10 small CNN transfer test for 98% adaptive dense-tail path correction. Conv layers use magnitude; dense tail uses low-alpha path modulation. Real CIFAR-10 images, 20k train subset, 5k test subset, 2 seeds.

Device: `cuda`

| Alpha | Before FT | After FT | Before retention | After retention | fc1 keep rate | Dead fc1 hidden |
|---:|---:|---:|---:|---:|---:|---:|
| `0.00` | `0.1665` | `0.4528` | `0.2958` | `0.8024` | `0.0110` | `81.5` |
| `0.03` | `0.1814` | `0.4579` | `0.3215` | `0.8116` | `0.0077` | `84.0` |
| `0.05` | `0.1650` | `0.4640` | `0.2918` | `0.8222` | `0.0059` | `86.5` |
| `0.08` | `0.1545` | `0.4335` | `0.2741` | `0.7678` | `0.0036` | `91.5` |
| `0.10` | `0.1584` | `0.4157` | `0.2809` | `0.7357` | `0.0025` | `101.5` |
| `0.15` | `0.1250` | `0.2424` | `0.2216` | `0.4286` | `0.0008` | `132.0` |
| `0.20` | `0.1129` | `0.1491` | `0.1999` | `0.2629` | `0.0002` | `165.0` |

Best one-shot alpha: `0.03`.
Best after-FT alpha: `0.05`.
Best balanced alpha: `0.03`.
