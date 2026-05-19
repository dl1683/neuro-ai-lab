# Torch Fashion-MNIST No-Balance Blend

Search conservative blends between magnitude and no-balance path-flow on Fashion-MNIST. alpha=0 is magnitude.

Best alpha: `0.25`

| Alpha | Mean accuracy over 90/95/98% |
|---:|---:|
| `0.00` | `0.6748` |
| `0.03` | `0.6774` |
| `0.06` | `0.6771` |
| `0.10` | `0.6863` |
| `0.15` | `0.7003` |
| `0.25` | `0.7038` |
| `0.40` | `0.6930` |
| `0.65` | `0.6532` |
| `1.00` | `0.6082` |
