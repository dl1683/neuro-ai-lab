# Unified Viability Selector Retrospective

Retrospective validation of a unified pre-finetune family selector: use ecology/homeostatic selection when magnitude has a dead route floor, use feature-preserving liveness repair when magnitude retains a route floor.

Family-selection accuracy: `1.00`

| Case | Selected family | Route floor | Selected method | Magnitude | Selected | Delta | Mag dead | Selected dead |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `full_cifar10_resnet20_sgd40` | `ecology_selector` | `0.0000` | `ecology_policy` | `0.4873` | `0.5262` | `+0.0390` | `311.5` | `3.0` |
| `deep_tinyresnet_cifar10` | `ecology_selector` | `0.0002` | `ecology_policy` | `0.2721` | `0.3150` | `+0.0429` | `311.0` | `1.0` |
| `ecology_cifar10` | `ecology_selector` | `0.0000` | `ecology_selected` | `0.4393` | `0.4674` | `+0.0281` | `340.5` | `0.5` |
| `ecology_cifar100` | `ecology_selector` | `0.0000` | `ecology_selected` | `0.0541` | `0.0908` | `+0.0367` | `540.0` | `1.0` |
| `pretrained_tinyimagenet_95` | `feature_viability_repair` | `3.6139` | `feature_viability_repair` | `0.1487` | `0.1482` | `-0.0005` | `11.0` | `0.0` |
| `pretrained_tinyimagenet_99` | `feature_viability_repair` | `0.8541` | `feature_viability_repair` | `0.0120` | `0.0143` | `+0.0023` | `365.0` | `4.0` |

## Interpretation

The unified selector does not claim one pruning mask is universally best. It first chooses the family: dead route floor implies homeostatic/ecology repair; preserved route floor implies feature-preserving liveness repair. This cleanly separates from-scratch severe-pruning collapse from pretrained feature-subspace preservation.
