# From Synaptic Saliency to Circuit Viability

## One-line thesis

Pruning should preserve circuits, not just synapses.

## Abstract

Severe neural network pruning is usually framed as a problem of ranking individual weights. This repo shows that view is incomplete. At extreme sparsity, global saliency methods can preserve many high-scoring weights while creating hidden topological cutsets that destroy communication through the model. The clearest failure is global SynFlow deleting every weight in dense classifier bridges, causing chance-level recovery after masked fine-tuning.

The constructive result is Path-Capacity Pruning: preserve minimum viable communication capacity across vulnerable route families, then spend the remaining parameter budget by saliency. Across CNN and residual experiments, capacity constraints rescue SynFlow collapse, beat magnitude in several severe-sparsity settings, and reveal a hierarchy of circuit constraints: first preserve liveness/homeostasis, then optimize route-family balance and degeneracy.

## Core claim

A sparse mask is not just a set of surviving weights. It is a circuit.

Severe pruning fails when the selected weights no longer form a viable communication graph. A pruning method should therefore optimize synaptic efficiency subject to circuit viability constraints:

- no required route family collapses;
- representation can still reach readout;
- residual/projection/main paths remain balanced enough for recovery;
- capacity is not overconcentrated into a single route family.

## Neuroscience mapping

| Neuroscience principle | Pruning analogue | Repo evidence |
|---|---|---|
| Synaptic pruning | Remove weak/redundant weights. | Magnitude, SynFlow, and saliency baselines. |
| Homeostatic plasticity | Prevent regions/routes from becoming silent. | Capacity reserve eliminates dead bridge/output collapse. |
| Use-dependent stabilization | Prefer routes that carry useful signal. | Naive activation-supported reserve was tested and failed, narrowing the mechanism. |
| Degeneracy | Preserve multiple route families, not one brittle path. | Diversity-penalized route optimizer beats magnitude in TinyResNet `99%`. |
| Communication backbones | Preserve bottleneck connectivity. | Projection/readout splits improve residual severe pruning. |

The important distinction is that biology does not merely delete isolated synapses. It remodels circuits while preserving functional viability.

## Discovery: SynFlow creates hidden cutsets

Global SynFlow can allocate zero surviving weights to dense classifier bridges.

![Global SynFlow bridge collapse](../figures/04_circuit_viability/figure_01_synflow_bridge_collapse.png)

| Case | Magnitude after FT | Global SynFlow after FT | SynFlow bridge keep | Dead bridge units |
|---|---:|---:|---:|---:|
| Fashion-MNIST CNN, `98%` | `80.86%` | `10.28%` | `0.0000` | `128/128` |
| CIFAR-10 CNN, `98%` | `44.08%` | `9.76%` | `0.0000` | `192/192` |
| CIFAR-10 CNN, `99%` | `33.24%` | `9.76%` | `0.0000` | `192/192` |

Primary artifacts:

- `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md`
- `results/04_criticality_pruning/synflow_pathology_synthesis.json`

Interpretation:

Global saliency can preserve high-scoring weights while deleting a required communication bridge. Fine-tuning cannot recover because the mask has removed the route.

## Constructive result: Path-Capacity Pruning

Path-Capacity Pruning adds circuit constraints around saliency:

1. Compute base saliency scores.
2. Identify vulnerable route families or cutsets.
3. Reserve capacity so critical routes remain viable.
4. Fill remaining budget by saliency.
5. Fine-tune under the fixed mask.

Current reusable utilities:

- `shared/path_capacity_pruning.py`
- `shared/residual_route_capacity.py`

![CNN capacity rescue](../figures/04_circuit_viability/figure_02_cnn_capacity_rescue.png)

## CNN results

### CIFAR-10 CNN reserve sweep

At CIFAR-10 CNN `99%`, capacity reserve beats magnitude across a broad reserve band.

Best result:

| Method | After FT | Delta vs magnitude | Wins |
|---|---:|---:|---:|
| magnitude | `32.31%` | baseline | baseline |
| global SynFlow | `9.76%` | `-22.56` pts | `0/4` |
| reserve `0.60` | `34.88%` | `+2.56` pts | `4/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_CNN_CAPACITY_RESERVE_SWEEP_99PCT.md`

### Fashion-MNIST transfer

Capacity reserve transfers to Fashion-MNIST CNN.

| Sparsity | Magnitude | Global SynFlow | Reserve `0.60` | Delta vs magnitude |
|---:|---:|---:|---:|---:|
| `98%` | `84.79%` | `9.92%` | `85.14%` | `+0.36` pts |
| `99%` | `80.24%` | `9.89%` | `81.75%` | `+1.51` pts |

Primary artifact:

- `experiments/04_criticality_pruning/FASHION_MNIST_CNN_CAPACITY_TRANSFER.md`

## Residual results

Residual networks expose the next layer of mechanism. Output liveness is necessary, but not always sufficient.

![Residual transfer](../figures/04_circuit_viability/figure_03_residual_transfer.png)

### TinyResNet route-quality audit

At TinyResNet `99%`, total dead outputs becomes a weaker predictor once masks are technically alive. Projection-route capacity becomes more informative.

| Scope | Route min correlation | Projection min correlation | Dead outputs correlation |
|---|---:|---:|---:|
| TinyResNet `98%` | `+0.852` | `+0.830` | `-0.927` |
| TinyResNet `99%` | `+0.215` | `+0.842` | `-0.370` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_ROUTE_QUALITY_AUDIT.md`

### Projection/readout split

A targeted split across main/projection/readout families beats magnitude in TinyResNet `99%`.

| Method | After FT | Delta vs magnitude | Wins |
|---|---:|---:|---:|
| magnitude | `24.89%` | baseline | baseline |
| plain reserve `0.60` | `20.53%` | `-4.36` pts | `0/4` |
| projection/readout `40/35/25` | `25.63%` | `+0.74` pts | `3/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_ROUTE_SPLIT_SWEEP_99PCT.md`

### Predicted route split

A route-deficit predictor selected a similar split before fine-tuning.

| Method | After FT | Delta vs magnitude | Wins |
|---|---:|---:|---:|
| magnitude | `25.04%` | baseline | baseline |
| plain reserve `0.60` | `20.21%` | `-4.83` pts | `0/4` |
| predicted deficit split | `25.12%` | `+0.08` pts | `2/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_PREDICTED_ROUTE_SPLIT_99PCT.md`

Interpretation:

The effect is tiny, but it moves from post-hoc split selection toward pre-finetuning route-deficit prediction.

### Diversity optimizer

Target matching alone overprotected projection routes and failed. Adding a degeneracy-style penalty against route-family overconcentration produced the strongest optimizer-style TinyResNet result.

| Method | After FT | Delta vs magnitude | Wins |
|---|---:|---:|---:|
| magnitude | `24.80%` | baseline | baseline |
| plain reserve `0.60` | `20.13%` | `-4.67` pts | `0/4` |
| tuned `40/35/25` | `26.04%` | `+1.25` pts | `3/4` |
| diversity optimizer | `25.85%` | `+1.05` pts | `4/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_DIVERSITY_ROUTE_OPTIMIZER_99PCT.md`

Interpretation:

Degeneracy becomes operational: do not let the optimizer preserve one route family by starving the rest.

### DeepTinyResNet transfer

A deeper residual model with two blocks per stage shows positive transfer, but the dominant mechanism changes.

Four-seed replicate:

| Method | After FT | Delta vs magnitude | Wins |
|---|---:|---:|---:|
| magnitude | `28.48%` | baseline | baseline |
| global SynFlow | `9.98%` | `-18.50` pts | `0/4` |
| plain reserve `0.60` | `30.20%` | `+1.72` pts | `3/4` |
| tuned `40/35/25` | `29.88%` | `+1.41` pts | `3/4` |
| diversity optimizer | `28.83%` | `+0.35` pts | `1/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_DEEP_TINY_RESNET_DIVERSITY_ROUTE_OPTIMIZER_99PCT_REPLICATE.md`

Interpretation:

In deeper residuals, broad liveness/homeostasis dominates. Magnitude leaves hundreds of dead outputs; capacity reserve fixes that and beats magnitude. Fine route-family diversity matters after liveness is no longer the limiting factor.

### ResNet-20-style transfer

A CIFAR ResNet-20-style model provides the first standard residual benchmark check.

| Method | After FT | Delta vs magnitude | Wins |
|---|---:|---:|---:|
| magnitude | `27.25%` | baseline | baseline |
| global SynFlow | `10.03%` | `-17.22` pts | `0/4` |
| plain reserve `0.60` | `33.22%` | `+5.97` pts | `4/4` |
| tuned `40/35/25` | `30.60%` | `+3.36` pts | `4/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_RESNET20_CAPACITY_99PCT.md`
- `experiments/04_criticality_pruning/CIFAR10_RESNET20_CAPACITY_99PCT_REPLICATE.md`

Interpretation:

The result transfers to a ResNet-20-style architecture. Again, the key mechanism is homeostasis: magnitude leaves hundreds of dead outputs, while reserve capacity eliminates dead outputs and improves recovery.

### Full CIFAR-10 ResNet-20-style transfer

The same ResNet-20-style experiment was run on full CIFAR-10 train/test and aggregated across six independent seeds.

| Method | After FT | Delta vs magnitude | Wins |
|---|---:|---:|---:|
| magnitude | `37.72%` | baseline | baseline |
| global SynFlow | `10.00%` | `-27.72` pts | `0/6` |
| plain reserve `0.60` | `39.21%` | `+1.50` pts | `6/6` |
| tuned `40/35/25` | `38.06%` | `+0.34` pts | `3/6` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_FULL_RESNET20_CAPACITY_99PCT_SIXSEED.md`

Interpretation:

The full-CIFAR result is positive but smaller than the subset result. Capacity reserve still eliminates dead outputs and beats magnitude on all six seeds, while SynFlow remains near chance. The tuned route split is positive on average but less robust, so the strongest current full-dataset claim is broad homeostatic circuit viability rather than hand-tuned route allocation.

### Stronger full-CIFAR SGD/cosine recipe

The full-CIFAR result was then stress-tested with a stronger dense training recipe: 20 SGD/cosine dense epochs followed by 5 masked fine-tune epochs.

| Method | After FT | Delta vs magnitude | Wins | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `42.87%` | baseline | baseline | `343.0` |
| global SynFlow | `10.00%` | `-32.87` pts | `0/4` | `647.2` |
| plain reserve `0.60` | `49.43%` | `+6.57` pts | `4/4` | `1.8` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_FULL_RESNET20_CAPACITY_99PCT_SGD_RECIPE_FOURSEED.md`

Interpretation:

This is the current strongest result. The capacity reserve advantage grows under the stronger recipe rather than disappearing. Mechanistically, magnitude has zero measured main-path floor and hundreds of dead outputs, while reserve restores a main-path capacity floor and nearly eliminates output death. This strengthens the neuroscience framing: the useful intervention is homeostatic circuit viability under extreme sparsity, not a marginal scoring heuristic.

### CIFAR-10 predicted route split transfer

The conservative route-deficit predictor was then applied back to full CIFAR-10 under the same strong SGD/cosine recipe.

| Method | After FT | Delta vs magnitude | Wins | Main min | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| magnitude | `42.09%` | baseline | baseline | `0.0000` | `1.3671` | `3.7406` | `352.0` |
| plain reserve | `47.88%` | `+5.78` pts | `2/2` | `1.2302` | `1.1929` | `3.3268` | `1.0` |
| predicted route split | `46.92%` | `+4.82` pts | `2/2` | `1.0115` | `1.5676` | `3.6403` | `0.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_FULL_RESNET20_PREDICTED_ROUTE_SPLIT_99PCT_SGD_RECIPE.md`

Interpretation:

The selector adapts qualitatively: on CIFAR-10 it chooses main/projection-heavy splits rather than the CIFAR-100 readout-heavy split. But it still trails plain reserve, so the current predictor is not a universal optimizer. The evidence supports a more specific claim: route-deficit prediction can adapt the direction of the constraint, while task-specific recovery still determines whether broad reserve or split allocation is best.

### Ecology-aware method selector

The next step was to select the intervention family itself. The selector measures plain-reserve readout capacity relative to the magnitude readout template before fine-tuning:

- If the ratio is high, keep broad reserve.
- If the ratio is low, use the conservative predicted route split.

Fresh cross-task validation:

| Task | Plain readout ratio | Selected method | Magnitude | Selected after FT | Delta | Wins |
|---|---:|---|---:|---:|---:|---:|
| CIFAR-10 | `~0.90` | plain reserve | `43.93%` | `46.74%` | `+2.81` pts | `2/2` |
| CIFAR-100 | `~0.36` | predicted route split | `5.41%` | `9.08%` | `+3.68` pts | `2/2` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR_ECOLOGY_SELECTOR_99PCT_SGD_RECIPE.md`

Interpretation:

This is the strongest predictor result so far because it chooses between broad homeostatic reserve and route-split allocation before recovery. It also fits the neuroscience framing cleanly: the system is not applying one fixed pruning recipe, it is measuring a circuit deficit and choosing the compensatory homeostatic constraint that matches the task ecology.

### Deeper residual architecture transfer

The same ecology selector was applied to DeepTinyResNet, a deeper residual subset benchmark used earlier in the project.

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `27.21%` | baseline | baseline | `0.0002` | `3.1649` | `311.0` |
| plain reserve | `31.50%` | `+4.29` pts | `2/2` | `1.1590` | `3.2669` | `1.0` |
| predicted route split | `30.05%` | `+2.84` pts | `2/2` | `0.9973` | `3.5900` | `0.0` |
| ecology policy | `31.50%` | `+4.29` pts | `2/2` | `1.1590` | `3.2669` | `1.0` |

Primary artifacts:

- `experiments/04_criticality_pruning/CIFAR10_DEEP_TINY_RESNET_ECOLOGY_SELECTOR_99PCT.md`
- `experiments/04_criticality_pruning/CIFAR10_DEEP_TINY_RESNET_ECOLOGY_SELECTOR_99PCT_POLICY.md`

Interpretation:

The selector transfers to a deeper residual backbone without retuning. The plain-reserve readout ratio is above threshold, so the policy keeps broad reserve. This preserves the stronger method family on the deeper CIFAR-10 residual setting while still allowing readout-split behavior on CIFAR-100.

### Longer full-CIFAR training schedule stress test

The selector was also tested under a longer full-CIFAR ResNet-20-style schedule: 40 dense SGD/cosine epochs and 8 masked fine-tune epochs. Dense accuracy reached roughly `90-91%`, higher than the earlier 20-epoch recipe.

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `48.73%` | baseline | baseline | `0.0000` | `3.7863` | `311.5` |
| plain reserve | `52.63%` | `+3.90` pts | `2/2` | `1.2209` | `3.2227` | `3.0` |
| predicted route split | `47.69%` | `-1.04` pts | `0/2` | `1.0173` | `3.5409` | `2.5` |
| ecology policy | `52.63%` | `+3.90` pts | `2/2` | `1.2209` | `3.2227` | `3.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_FULL_RESNET20_ECOLOGY_SELECTOR_99PCT_SGD40.md`

Interpretation:

The longer-schedule result strengthens the family-selection claim. The fixed selector keeps broad reserve because readout ratio remains above threshold, and the route split is actively harmful. That means the selector is not merely choosing the more complex intervention; it avoids split allocation when the measured circuit deficit does not require it.

### TinyImageNet-200 external proxy

The first non-CIFAR proxy used real TinyImageNet-200 data with a ResNet-20-style 200-class model. This was run as a subset stress test, not a publication-grade TinyImageNet benchmark.

| Method | After FT | Delta vs magnitude | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|
| magnitude | `2.32%` | baseline | `0.0000` | `2.4574` | `622.0` |
| plain reserve | `3.08%` | `+0.76` pts | `1.1294` | `1.1036` | `0.0` |
| predicted route split | `2.90%` | `+0.58` pts | `0.9163` | `1.6705` | `0.0` |
| ecology policy | `2.90%` | `+0.58` pts | `0.9163` | `1.6705` | `0.0` |

Primary artifact:

- `experiments/04_criticality_pruning/TINYIMAGENET_RESNET20_ECOLOGY_SELECTOR_99PCT.md`

Interpretation:

This result is mixed. The selector identifies a readout deficit and chooses the readout-heavy split, but plain reserve recovers slightly better on the single seed. Both viability methods beat magnitude and eliminate dead outputs, but dense accuracy is only `16.48%`, so the external proxy is mainly a boundary condition. The next step is not to overclaim; it is to improve the external training setup and retest the fixed selector.

### TinyImageNet with ImageNet-pretrained ResNet-18

To separate selector failure from weak dense training, the TinyImageNet proxy was repeated with an ImageNet-pretrained ResNet-18 adapted to 200 classes. Dense accuracy rose to roughly `60%`, making this a more meaningful external check.

At `99%` sparsity:

| Method | After FT | Delta vs magnitude | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|
| magnitude | `1.07%` | baseline | `0.8478` | `4.3134` | `366.0` |
| plain reserve | `0.97%` | `-0.10` pts | `2.6809` | `2.6783` | `0.0` |
| predicted route split | `0.80%` | `-0.27` pts | `2.3691` | `4.2285` | `0.0` |
| ecology policy | `0.80%` | `-0.27` pts | `2.3691` | `4.2285` | `0.0` |

At `95%` sparsity:

| Method | After FT | Delta vs magnitude | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|
| magnitude | `15.23%` | baseline | `3.6182` | `5.3203` | `10.0` |
| plain reserve | `3.97%` | `-11.27` pts | `4.2285` | `5.5299` | `0.0` |
| predicted route split | `3.53%` | `-11.70` pts | `3.8842` | `6.0488` | `0.0` |
| ecology policy | `3.97%` | `-11.27` pts | `4.2285` | `5.5299` | `0.0` |

Primary artifacts:

- `experiments/04_criticality_pruning/TINYIMAGENET_RESNET18_PRETRAINED_ECOLOGY_SELECTOR_99PCT.md`
- `experiments/04_criticality_pruning/TINYIMAGENET_RESNET18_PRETRAINED_ECOLOGY_SELECTOR_95PCT.md`

Interpretation:

This is the clearest negative result so far. On a pretrained external model, the current homeostatic masks preserve liveness metrics but destroy useful pretrained feature structure. Magnitude is much better at `95%`, even with a few dead outputs. The current theory therefore cannot be "preserve liveness at all costs"; it needs a second principle for pretrained systems: do not disrupt high-information feature subspaces unless there is a true topological death risk.

### Feature-preserving liveness repair

The pretrained failure motivated a new intervention: start from global magnitude, repair only truly dead output rows, and remove the weakest non-protected kept weights to preserve the global sparsity budget. This keeps the pretrained feature subspace as intact as possible while enforcing minimal circuit viability.

At `95%` sparsity on ImageNet-pretrained ResNet-18, two-seed aggregate:

| Method | After FT | Delta vs magnitude | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|
| magnitude | `14.87%` | baseline | `3.6139` | `5.3197` | `11.0` |
| plain reserve | `3.30%` | `-11.57` pts | `4.2285` | `5.5317` | `0.0` |
| feature-viability repair | `14.82%` | `-0.05` pts | `3.6139` | `5.3197` | `0.0` |

At the harder `99%` sparsity cliff:

| Method | After FT | Delta vs magnitude | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|
| magnitude | `1.20%` | baseline | `0.8541` | `4.3197` | `365.0` |
| plain reserve | `0.87%` | `-0.33` pts | `2.6813` | `2.6779` | `0.0` |
| feature-viability repair | `1.43%` | `+0.23` pts | `1.5496` | `4.3166` | `4.0` |

Primary artifact:

- `experiments/04_criticality_pruning/TINYIMAGENET_RESNET18_PRETRAINED_FEATURE_VIABILITY_95PCT.md`
- `experiments/04_criticality_pruning/TINYIMAGENET_RESNET18_PRETRAINED_FEATURE_VIABILITY_95PCT_TWOSEED.md`
- `experiments/04_criticality_pruning/TINYIMAGENET_RESNET18_PRETRAINED_FEATURE_VIABILITY_99PCT.md`

Interpretation:

At `95%`, this does not beat magnitude, but it fixes the main external failure: broad homeostatic reserve destroys pretrained performance, while minimal liveness repair eliminates dead outputs and preserves magnitude-level accuracy across two seeds. At `99%`, feature-viability repair is the only positive intervention so far, though absolute recovery remains near chance. The theory now has a sharper bifurcation: randomly trained or from-scratch sparse recovery benefits from homeostatic reserve, but pretrained systems require feature-subspace preservation with only targeted viability repair.

### Pretrained tradeoff selector validation

The corrected tradeoff selector was then tested on a fresh pretrained TinyImageNet seed at `95%`. It ranked masks before fine-tuning by feature overlap, liveness, readout preservation, and dead-output penalty.

| Method | After FT | Delta vs magnitude | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|
| magnitude | `15.67%` | baseline | `3.6191` | `5.3169` | `11.0` |
| feature-viability repair | `15.70%` | `+0.03` pts | `3.6190` | `5.3169` | `0.0` |
| plain reserve | `3.80%` | `-11.87` pts | `4.2285` | `5.5325` | `0.0` |
| predicted route split | `3.00%` | `-12.67` pts | `3.8842` | `6.0488` | `0.0` |
| tradeoff policy | `15.70%` | `+0.03` pts | `3.6190` | `5.3169` | `0.0` |

Primary artifact:

- `experiments/04_criticality_pruning/TINYIMAGENET_RESNET18_PRETRAINED_TRADEOFF_SELECTOR_95PCT.md`

Interpretation:

This is a small but important validation because it checks the selector on the main external failure mode. The policy avoided the homeostatic masks that made all route-liveness metrics look better while destroying pretrained performance. It selected feature repair, preserved magnitude-level accuracy, and eliminated dead outputs. This supports the refined principle: viability repair must be constrained by preservation of useful learned computation.

### CIFAR-100 output-diversity stress test

The next stress test moved from CIFAR-10 to full CIFAR-100 with the same ResNet-20-style body and a 100-class readout. This makes the output/readout route much harder to preserve under `99%` sparsity.

Plain reserve transfer:

| Method | After FT | Delta vs magnitude | Wins | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `6.58%` | baseline | baseline | `549.5` |
| global SynFlow | `1.00%` | `-5.58` pts | `0/2` | `700.5` |
| plain reserve `0.60` | `7.64%` | `+1.06` pts | `2/2` | `0.5` |

Route-family split follow-up:

| Method | After FT | Delta vs magnitude | Wins | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|
| magnitude | `6.53%` | baseline | baseline | `3.0558` | `548.0` |
| plain reserve | `7.98%` | `+1.45` pts | `2/2` | `1.0886` | `0.0` |
| balanced `40/35/25` | `8.75%` | `+2.22` pts | `2/2` | `1.6400` | `0.0` |
| readout-heavy `35/20/45` | `9.02%` | `+2.49` pts | `2/2` | `2.1365` | `0.0` |
| readout-main `45/15/40` | `9.12%` | `+2.59` pts | `2/2` | `2.0334` | `0.0` |

Pre-finetune predicted split:

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `6.98%` | baseline | baseline | `0.0000` | `3.0440` | `541.5` |
| plain reserve | `7.68%` | `+0.71` pts | `2/2` | `1.1977` | `1.0953` | `0.5` |
| conservative predicted split | `8.79%` | `+1.81` pts | `2/2` | `1.0058` | `2.0334` | `0.5` |

The conservative predictor selected the same `50/10/40` main/projection/readout split on both fresh seeds from route diagnostics alone.

Primary artifacts:

- `experiments/04_criticality_pruning/CIFAR100_FULL_RESNET20_CAPACITY_99PCT_SGD_RECIPE.md`
- `experiments/04_criticality_pruning/CIFAR100_FULL_RESNET20_ROUTE_SPLIT_99PCT_SGD_RECIPE.md`
- `experiments/04_criticality_pruning/CIFAR100_FULL_RESNET20_CONSERVATIVE_PREDICTED_ROUTE_SPLIT_99PCT_SGD_RECIPE.md`

Interpretation:

CIFAR-100 is not solved: absolute recovery after `99%` pruning is low. But the mechanism transfers and becomes more specific. Generic reserve prevents output death and beats magnitude, while a readout/main-biased split improves recovery further. The conservative predictor shows this can be selected before fine-tuning by balancing a main-path floor against readout restoration. This supports a stronger neuroscience connection: different task ecologies should impose different homeostatic circuit constraints, and class-rich tasks appear to demand stronger readout preservation.

![Output diversity shifts the constraint](../figures/04_circuit_viability/figure_05_output_diversity.png)

### CIFAR-100 tradeoff selector stress test

The feature-preservation/liveness tradeoff selector was then stress-tested on fresh CIFAR-100 seeds. This is the task ecology where previous evidence favored route/readout allocation.

V1 result:

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `6.98%` | baseline | baseline | `0.0000` | `3.0585` | `557.0` |
| feature repair | `8.21%` | `+1.24` pts | `2/2` | `0.5745` | `2.8939` | `61.5` |
| plain reserve | `8.78%` | `+1.80` pts | `2/2` | `1.1977` | `1.0936` | `0.5` |
| predicted route split | `8.87%` | `+1.89` pts | `2/2` | `1.0059` | `2.0334` | `1.0` |
| tradeoff policy V1 | `8.21%` | `+1.24` pts | `2/2` | `0.5745` | `2.8939` | `61.5` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR100_FULL_RESNET20_TRADEOFF_SELECTOR_99PCT_SGD20.md`

Interpretation:

This is a useful failure. The V1 selector again chose feature repair, but CIFAR-100 preferred route split. The missing term was task ecology: when the magnitude mask has low readout score and hundreds of dead outputs, the selector should reduce feature-overlap weight and prioritize route liveness.

V2 policy projection:

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| tradeoff policy V1 | `8.21%` | `+1.24` pts | `2/2` | `0.5745` | `2.8939` | `61.5` |
| tradeoff policy V2 | `8.87%` | `+1.89` pts | `2/2` | `1.0059` | `2.0334` | `1.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR100_FULL_RESNET20_TRADEOFF_SELECTOR_V2_POLICY.md`

Interpretation:

V2 is not a new training run; it is a policy projection over the already evaluated fresh candidates. It changes only the pre-finetune selection rule by adding output/readout pressure. Under that rule, both CIFAR-100 seeds switch from feature repair to predicted route split, matching the best evaluated candidate. This is the next prospective validation target.

Fresh V2 prospective validation:

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `6.54%` | baseline | baseline | `0.0000` | `3.0572` | `547.0` |
| feature repair | `8.87%` | `+2.33` pts | `2/2` | `0.4843` | `2.8954` | `57.0` |
| plain reserve | `8.31%` | `+1.77` pts | `2/2` | `1.2048` | `1.0750` | `0.0` |
| predicted route split | `9.26%` | `+2.72` pts | `2/2` | `1.0313` | `2.0334` | `0.0` |
| tradeoff policy V2 | `9.26%` | `+2.72` pts | `2/2` | `1.0313` | `2.0334` | `0.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR100_FULL_RESNET20_TRADEOFF_SELECTOR_V2_99PCT_SGD20.md`

Interpretation:

The prospective V2 run validates the correction. On both fresh seeds, the selector chose route split before fine-tuning, and route split was the best mean candidate. This is a real mechanism upgrade: the selector now behaves differently across task ecologies. CIFAR-10 and pretrained TinyImageNet favor feature preservation; CIFAR-100 favors output/readout route repair.

## First transformer analogue: TinyViT MLP route death

The first transformer-style experiment uses a small ViT on CIFAR-10. The vulnerable routes are not CNN dense bridges. They are MLP down-projection rows and the classifier readout.

Setup:

- TinyViT with patch embedding, four self-attention blocks, MLP expansion/down-projection, and classifier head.
- CIFAR-10 subset training, two seeds.
- `98%` unstructured sparsity.
- Dense training followed by masked fine-tuning under fixed masks.

Result:

| Method | After FT | Delta vs magnitude | Wins | Dead outputs | MLP-down dead | MLP-down min | Head min |
|---|---:|---:|---:|---:|---:|---:|---:|
| magnitude | `10.07%` | baseline | baseline | `1956.5` | `512.0` | `0.0` | `0.0` |
| global SynFlow | `11.00%` | `+0.93` pts | `1/2` | `2792.0` | `389.5` | `0.0` | `71.0` |
| minimal liveness repair | `11.31%` | `+1.24` pts | `1/2` | `62.5` | `0.0` | `1.0` | `0.5` |
| selective MLP/readout repair | `9.59%` | `-0.48` pts | `0/2` | `1452.0` | `0.0` | `1.0` | `1.0` |
| all-route liveness floor | `9.96%` | `-0.11` pts | `0/2` | `0.0` | `0.0` | `1.0` | `1.0` |
| MLP/readout reserve | `9.95%` | `-0.12` pts | `1/2` | `2181.0` | `0.0` | `15.0` | `8.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_CIRCUIT_VIABILITY_98PCT.md`

Interpretation:

This is not a strong transformer result yet, but it is a real analogue. Magnitude pruning kills every MLP down-projection output row in the tested TinyViT masks. Minimal liveness repair removes that MLP route death and slightly improves mean recovery. The sharper negative results are informative: MLP/readout-only repair is too narrow, and all-route liveness eliminates every measured dead output while still underperforming magnitude. The transformer lesson is therefore aligned with the rest of the repo: viability is not raw liveness. The next transformer step must preserve feature subspaces and residual-stream/attention communication while repairing the route death that matters.

At `95%`, the same TinyViT route-death signature persists on fresh seeds:

| Method | After FT | Delta vs magnitude | Wins | Dead outputs | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `9.87%` | baseline | baseline | `1247.0` | `512.0` | `314.5` |
| global SynFlow | `10.78%` | `+0.91` pts | `2/2` | `2253.0` | `232.0` | `114.5` |
| minimal liveness repair | `10.87%` | `+1.00` pts | `2/2` | `72.5` | `0.0` | `26.0` |
| selective MLP/readout repair | `10.23%` | `+0.36` pts | `1/2` | `760.5` | `0.0` | `326.5` |
| attention+MLP/readout repair | `10.31%` | `+0.44` pts | `2/2` | `467.0` | `0.0` | `17.5` |
| all-route liveness floor | `10.36%` | `+0.49` pts | `1/2` | `0.0` | `0.0` | `0.0` |
| MLP/readout reserve | `10.22%` | `+0.35` pts | `1/2` | `1578.5` | `0.0` | `508.5` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_CIRCUIT_VIABILITY_95PCT.md`

Interpretation:

The `95%` run confirms both MLP and attention-output route death, but it still does not produce a strong transformer pruning method. Minimal liveness repair is again best on mean and wins both seeds. Attention+MLP/readout repair reduces attention-output death from `314.5` to `17.5`, but it still trails minimal repair. This suggests the transformer path needs stronger dense training and richer diagnostics, especially residual-stream subspace preservation rather than row liveness alone.

### TinyViT feature-subspace diagnostic

A follow-up TinyViT diagnostic measured pre-finetune CLS/residual-stream feature preservation for each mask, then fine-tuned the same masks. This tests whether transformer recovery is better explained by preserving the dense representation than by row liveness alone.

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | Dead outputs | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| magnitude | `11.02%` | baseline | baseline | `0.0085` | `1260.0` | `511.5` | `314.0` |
| global SynFlow | `16.86%` | `+5.84` pts | `2/2` | `0.0526` | `2269.5` | `239.0` | `133.0` |
| minimal liveness repair | `10.31%` | `-0.71` pts | `1/2` | `-0.0083` | `78.5` | `0.0` | `24.5` |
| attention+MLP/readout repair | `9.53%` | `-1.49` pts | `1/2` | `-0.0030` | `484.5` | `0.0` | `18.5` |
| all-route liveness floor | `10.10%` | `-0.92` pts | `0/2` | `-0.0071` | `0.0` | `0.0` | `0.0` |

Centered CLS cosine vs after-FT recovery correlation: `0.583`.

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_SUBSPACE_DIAGNOSTIC_95PCT.md`

Interpretation:

This changes the transformer direction. In this fresh batch, global SynFlow is the best TinyViT mask even though it leaves many dead rows, while liveness repair removes route death but underperforms. The positive signal is feature-subspace preservation: centered CLS alignment is a better predictor than raw liveness in this small diagnostic. Transformer circuit viability should therefore be framed as preserving residual-stream computation under sparsity, not merely keeping MLP or attention rows alive.

### Prospective TinyViT feature-subspace selector

The next TinyViT experiment selected the mask family before fine-tuning by choosing the highest centered CLS/residual-stream feature alignment.

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `8.82%` | baseline | baseline | `0.0164` | `512.0` | `305.5` |
| global SynFlow | `13.57%` | `+4.75` pts | `2/2` | `0.0193` | `235.0` | `134.5` |
| feature-subspace policy | `12.24%` | `+3.42` pts | `1/2` | `0.0210` | `376.5` | `217.5` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_SUBSPACE_SELECTOR_95PCT.md`

Interpretation:

The prospective selector is positive but incomplete. It chose global SynFlow on one seed and magnitude on the other because magnitude had slightly higher centered CLS alignment. Global SynFlow still recovered better on both seeds. This means argmax feature alignment is not enough; the selector needs a margin rule that considers route-risk when feature scores are close.

Feature-route margin policy projection:

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| feature-subspace policy | `12.24%` | `+3.42` pts | `1/2` | `0.0210` | `376.5` | `217.5` |
| feature-route margin policy | `13.57%` | `+4.75` pts | `2/2` | `0.0193` | `235.0` | `134.5` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_POLICY_95PCT.md`

Interpretation:

This is a projection over already evaluated candidates, not a fresh training run. It shows the next transformer selector form: preserve residual-stream features, but when feature scores are close, prefer the mask with less transformer route death. The next step is a fresh prospective margin-policy validation.

Fresh prospective margin-policy validation:

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `8.95%` | baseline | baseline | `0.0005` | `511.5` | `308.0` |
| global SynFlow | `14.60%` | `+5.65` pts | `2/2` | `0.0304` | `225.0` | `133.0` |
| attention+MLP/readout repair | `11.17%` | `+2.22` pts | `2/2` | `-0.0001` | `0.0` | `14.5` |
| all-route liveness floor | `9.34%` | `+0.39` pts | `1/2` | `-0.0003` | `0.0` | `0.0` |
| feature-route margin policy | `14.60%` | `+5.65` pts | `2/2` | `0.0304` | `225.0` | `133.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_95PCT.md`

Interpretation:

The fresh selector validation supports the transformer-specific hypothesis. The policy selected SynFlow on both seeds from pre-finetune feature/route diagnostics, and that selected method was the best mean candidate. It is still not a strong benchmark because absolute accuracy is low, but it is a real direction: transformer viability is closer to residual-stream representation preservation with route-risk guardrails than to eliminating every dead row.

At `90%` sparsity, the same selector again chooses SynFlow on both fresh seeds:

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `10.32%` | baseline | baseline | `0.0264` | `503.0` | `5.5` |
| global SynFlow | `14.56%` | `+4.24` pts | `2/2` | `0.0442` | `133.5` | `80.5` |
| minimal liveness repair | `11.50%` | `+1.18` pts | `2/2` | `0.0286` | `0.5` | `1.5` |
| all-route liveness floor | `11.52%` | `+1.20` pts | `2/2` | `0.0290` | `0.0` | `0.0` |
| feature-route margin policy | `14.56%` | `+4.24` pts | `2/2` | `0.0442` | `133.5` | `80.5` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_90PCT.md`

Interpretation:

The `90%` result is less floor-dominated and preserves the same ordering: representation-preserving SynFlow beats liveness-first repairs. This strengthens the transformer-specific claim that viable sparse transformers require preserving residual-stream computation, even if some row-level route death remains.

Stronger TinyViT training changes the boundary:

| Method | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|
| magnitude | `16.70%` | baseline | `0.0358` | `73.0` | `1.0` |
| global SynFlow | `14.35%` | `-2.35` pts | `0.0440` | `92.0` | `70.0` |
| minimal liveness repair | `16.54%` | `-0.16` pts | `0.0371` | `0.0` | `0.0` |
| attention+MLP/readout repair | `16.60%` | `-0.10` pts | `0.0382` | `0.0` | `0.0` |
| feature-route margin policy | `14.35%` | `-2.35` pts | `0.0440` | `92.0` | `70.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_90PCT_STRONG.md`

Interpretation:

This is a one-seed pilot, but it is important. With full CIFAR-10 training and 20 dense epochs, dense TinyViT reaches `71.62%`. In this regime, the feature-route selector overselects SynFlow: SynFlow has the highest centered CLS alignment but lower recovery than magnitude. The earlier weak-TinyViT rule is therefore not enough. The transformer selector needs a stronger objective that distinguishes useful residual-stream preservation from masks that preserve feature direction while damaging trainable capacity.

V2 strong-selector guardrail:

| Method | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|
| magnitude | `12.18%` | baseline | `-0.0054` | `79.0` | `3.0` |
| global SynFlow | `10.25%` | `-1.93` pts | `0.0027` | `101.0` | `75.0` |
| minimal liveness repair | `11.79%` | `-0.39` pts | `-0.0061` | `0.0` | `0.0` |
| attention+MLP/readout repair | `12.00%` | `-0.18` pts | `-0.0069` | `0.0` | `0.0` |
| V2 feature-route policy | `12.18%` | `0.00` pts | `-0.0054` | `79.0` | `3.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V2_90PCT_STRONG.md`

Interpretation:

V2 adds a trainable-capacity guardrail: do not choose SynFlow when its feature-alignment advantage is small and its route-death burden is much higher than magnitude. On a fresh strong TinyViT seed, this chooses magnitude, which is the best evaluated candidate. This is not yet a robust transformer result, but it is the right kind of correction: feature preservation must be balanced against trainable sparse capacity.

Strong V2 replicate:

| Method | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|
| magnitude | `13.82%` | baseline | `0.0188` | `74.0` | `2.0` |
| global SynFlow | `14.24%` | `+0.42` pts | `0.0248` | `87.0` | `72.0` |
| minimal liveness repair | `14.59%` | `+0.77` pts | `0.0198` | `0.0` | `0.0` |
| all-route liveness floor | `14.84%` | `+1.02` pts | `0.0198` | `0.0` | `0.0` |
| V2 feature-route policy | `13.82%` | `0.00` pts | `0.0188` | `74.0` | `2.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V2_90PCT_STRONG_REPLICATE.md`

Interpretation:

The replicate falsifies the simple V2 rule. In this seed, all-route liveness is the best candidate, while V2 selects magnitude. The strong TinyViT regime therefore needs a three-way selector: preserve residual-stream features when they dominate, prefer magnitude when it is already trainable and SynFlow has too much route death, and prefer liveness repair when dead-row removal improves trainable capacity without overdisrupting the representation.

V3 strong-selector test:

| Method | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|
| magnitude | `11.39%` | baseline | `0.0353` | `62.0` | `4.0` |
| global SynFlow | `14.52%` | `+3.13` pts | `0.0561` | `101.0` | `79.0` |
| minimal liveness repair | `11.42%` | `+0.03` pts | `0.0376` | `2.0` | `0.0` |
| all-route liveness floor | `11.59%` | `+0.20` pts | `0.0376` | `0.0` | `0.0` |
| V3 feature-route policy | `14.52%` | `+3.13` pts | `0.0561` | `101.0` | `79.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V3_90PCT_STRONG.md`

Interpretation:

V3 is the first correction after the V2 replicate failure. It does not claim the strong TinyViT problem is solved. It adds the missing branch: when residual-stream feature preservation has a large margin, choose the representation-preserving mask even if it carries row-level route death. On this fresh strong seed, that selected SynFlow and beat magnitude by `+3.13` points, while the zero-dead all-route repair recovered only `+0.20` points. This sharpens the neuroscience analogy: viable sparse transformers are not defined by keeping every anatomical route alive; they need the right functional ensemble to remain trainable.

V3 strong-selector replicate:

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `8.07%` | baseline | baseline | `-0.0214` | `73.0` | `4.0` |
| global SynFlow | `15.05%` | `+6.98` pts | `2/2` | `0.0399` | `94.0` | `82.5` |
| minimal liveness repair | `8.15%` | `+0.08` pts | `1/2` | `-0.0229` | `0.0` | `0.0` |
| all-route liveness floor | `8.14%` | `+0.07` pts | `1/2` | `-0.0229` | `0.0` | `0.0` |
| V3 feature-route policy | `15.05%` | `+6.98` pts | `2/2` | `0.0399` | `94.0` | `82.5` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V3_90PCT_STRONG_REPLICATE.md`

Interpretation:

The replicate strengthens the feature-dominant branch. On two fresh full-train TinyViT seeds, V3 again selected SynFlow before fine-tuning because centered CLS/residual-stream alignment separated strongly from magnitude and the liveness repairs. Both selected masks beat magnitude. The important negative control is that all-route liveness eliminated measured dead rows but stayed near the collapsed magnitude baseline. This makes the neuroscience connection sharper: the useful unit is not isolated synapse saliency or anatomical row survival; it is a functional circuit ensemble that preserves representational dynamics under a sparse substrate.

Six-seed strong-selector boundary projection:

| Selector | Seeds | Positive vs magnitude | Matches best candidate | Mean delta vs magnitude | Mean gap to best |
|---:|---:|---:|---:|---:|
| V3 | `12` | `8/12` | `7/12` | `+2.84` pts | `0.98` pts |
| V4 | `12` | `8/12` | `9/12` | `+2.88` pts | `0.94` pts |
| V5 | `12` | `9/12` | `11/12` | `+3.80` pts | `0.02` pts |
| V6 | `15` | `11/15` | `14/15` | `+3.25` pts | `0.01` pts |
| V7 | `15` | `11/15` | `15/15` | `+3.25` pts | `0.00` pts |

Primary artifact:

- `experiments/04_criticality_pruning/TINY_VIT_STRONG_SELECTOR_BOUNDARY_SYNTHESIS.md`

Interpretation:

This synthesis projects the same V3, V4, V5, V6, and V7 rules over all completed strong TinyViT seeds. V4 fixed the two small V3 guardrail misses, but seed 306 and seed 310 expose the deeper failure: live repair masks can look safer and still remain at the magnitude floor, while SynFlow preserves a much more trainable sparse basin. V5 adds a simple SynFlow masked-recovery prior: if SynFlow's masked-before accuracy is at least magnitude and close to the selected repair, prefer SynFlow. Seed 312 is the first V5 prospective miss, showing that the live-repair branch needs a tie-breaker between attention+MLP repair and minimal liveness. V6 adds that narrow tie-breaker and fixes seed 312 in projection. Seed 315 motivates V7's magnitude-vs-live-repair guardrail. V7 reaches the evaluated oracle on the current boundary and seed 320 prospectively validates that the new guardrail does not break the SynFlow branch.

V4 strong-selector test:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `9.84%` | `10.22%` | baseline | `-0.0156` | `66.0` | `3.0` |
| global SynFlow | `15.55%` | `17.09%` | `+6.87` pts | `0.0323` | `103.0` | `82.0` |
| all-route liveness floor | `9.33%` | `10.30%` | `+0.08` pts | `-0.0144` | `0.0` | `0.0` |
| V4 feature-route policy | `15.55%` | `17.09%` | `+6.87` pts | `0.0323` | `103.0` | `82.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V4_90PCT_STRONG.md`

Interpretation:

V4 adds masked pre-finetune accuracy as a prospective trainability diagnostic for the ambiguous liveness-vs-magnitude branch. The first fresh V4 seed does not hit that ambiguous branch; it is another feature-dominant seed. That still matters: before fine-tuning, SynFlow already preserves both residual-stream alignment and masked behavior better than the liveness repairs, and after fine-tuning it beats magnitude by `+6.87` points. This further supports the idea that transformer sparse viability is functional ensemble preservation, not row-level survival alone.

V4 seed-306 non-SynFlow branch:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `10.17%` | `10.99%` | baseline | `0.0205` | `86.0` | `3.0` |
| global SynFlow | `10.40%` | `13.29%` | `+2.30` pts | `-0.0064` | `113.0` | `70.0` |
| minimal liveness repair | `10.86%` | `11.58%` | `+0.59` pts | `0.0239` | `0.0` | `1.0` |
| attention+MLP/readout repair | `10.97%` | `11.58%` | `+0.59` pts | `0.0248` | `0.0` | `1.0` |
| V4 feature-route policy | `10.97%` | `11.58%` | `+0.59` pts | `0.0248` | `0.0` | `1.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V4_90PCT_STRONG_SEED306.md`

Interpretation:

Seed 306 is the most useful V4 failure so far. The selector correctly avoids magnitude and chooses a live repair mask that improves recovery. But it misses the best candidate: SynFlow has worse centered CLS alignment and much higher row death, yet it recovers better after fine-tuning. This shows that residual-stream alignment measured before fine-tuning is not sufficient. Some SynFlow masks may preserve a trainable sparse basin that is not captured by row liveness or centered CLS cosine.

V5 fresh feature/SynFlow branch:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `6.03%` | `7.25%` | baseline | `0.0142` | `55.0` | `4.0` |
| global SynFlow | `7.23%` | `10.18%` | `+2.93` pts | `0.0281` | `105.0` | `91.0` |
| all-route liveness floor | `6.03%` | `7.46%` | `+0.21` pts | `0.0187` | `0.0` | `0.0` |
| V5 feature-route policy | `7.23%` | `10.18%` | `+2.93` pts | `0.0281` | `105.0` | `91.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V5_90PCT_STRONG.md`

Interpretation:

Seed 308 validates the feature/SynFlow branch prospectively. V5 selected SynFlow before fine-tuning, and SynFlow was the best evaluated candidate. The important negative control is again liveness: all-route liveness eliminated measured dead rows but barely moved recovery. This strengthens the claim that transformer sparse viability sometimes depends on preserving a trainable residual-stream basin rather than maximizing row survival. Seed 310 below separately tests the ambiguous SynFlow-prior branch.

V5 fresh SynFlow-prior branch:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `4.82%` | `5.86%` | baseline | `0.0047` | `96.0` | `2.0` |
| global SynFlow | `8.27%` | `15.16%` | `+9.30` pts | `0.0164` | `98.0` | `83.0` |
| minimal liveness repair | `4.96%` | `5.90%` | `+0.04` pts | `0.0053` | `1.0` | `0.0` |
| all-route liveness floor | `4.89%` | `5.86%` | `+0.00` pts | `0.0050` | `0.0` | `0.0` |
| V5 feature-route policy | `8.27%` | `15.16%` | `+9.30` pts | `0.0164` | `98.0` | `83.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V5_90PCT_STRONG_SEED310.md`

Interpretation:

Seed 310 validates the previously open V5 branch. A dense-only scanner selected the seed because the V5 rule entered `synflow_masked_recovery_prior` before any masked fine-tuning. In the full validation, the V4-style liveness choice stayed at the magnitude floor, while SynFlow recovered `+9.30` points over magnitude and matched the best evaluated candidate. The neuroscience connection is sharp: eliminating measured dead rows is not sufficient if the remaining circuit cannot recover function; the sparse substrate also needs globally coherent signal paths that keep the residual-stream computation trainable.

V5 unselected fresh seed:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `7.76%` | `9.79%` | baseline | `-0.0235` | `79.0` | `3.0` |
| global SynFlow | `12.29%` | `15.91%` | `+6.12` pts | `0.0423` | `92.0` | `81.0` |
| minimal liveness repair | `7.78%` | `9.78%` | `-0.01` pts | `-0.0224` | `0.0` | `0.0` |
| all-route liveness floor | `7.78%` | `9.97%` | `+0.18` pts | `-0.0224` | `0.0` | `0.0` |
| V5 feature-route policy | `12.29%` | `15.91%` | `+6.12` pts | `0.0423` | `92.0` | `81.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V5_90PCT_STRONG_SEED311.md`

Interpretation:

Seed 311 is an unselected fresh prospective check, not a branch-scanned seed. The fixed V5 rule again selects SynFlow before fine-tuning and matches the best evaluated candidate. This reduces the risk that the seed-310 result is only a scanner artifact, while preserving the same caveat: TinyViT is still a small transformer analogue, not evidence of general LLM pruning transfer.

V5 live-repair miss:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `10.23%` | `11.89%` | baseline | `0.0130` | `60.0` | `4.0` |
| global SynFlow | `7.39%` | `10.68%` | `-1.21` pts | `0.0022` | `75.0` | `64.0` |
| minimal liveness repair | `10.36%` | `12.10%` | `+0.21` pts | `0.0122` | `0.0` | `0.0` |
| attention+MLP/readout repair | `10.26%` | `11.87%` | `-0.02` pts | `0.0140` | `0.0` | `0.0` |
| all-route liveness floor | `10.36%` | `12.05%` | `+0.16` pts | `0.0122` | `0.0` | `0.0` |
| V5 feature-route policy | `10.26%` | `11.87%` | `-0.02` pts | `0.0140` | `0.0` | `0.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V5_90PCT_STRONG_SEED312.md`

Interpretation:

Seed 312 is the first prospective V5 miss. The selector correctly avoids SynFlow, but it overtrusts a tiny centered-CLS advantage for attention+MLP repair. Minimal liveness and all-route liveness have slightly higher masked-before accuracy and better after-FT recovery. This localizes the next algorithmic problem: the SynFlow-prior branch is useful, but the live-repair branch needs a trainability tie-breaker when feature margins are small.

V6 magnitude guardrail:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `15.61%` | `16.44%` | baseline | `0.0602` | `69.0` | `5.0` |
| global SynFlow | `9.03%` | `10.10%` | `-6.34` pts | `0.0013` | `112.0` | `105.0` |
| minimal liveness repair | `15.40%` | `16.27%` | `-0.17` pts | `0.0599` | `1.0` | `0.0` |
| attention+MLP/readout repair | `15.53%` | `16.22%` | `-0.22` pts | `0.0597` | `0.0` | `0.0` |
| all-route liveness floor | `15.42%` | `16.28%` | `-0.16` pts | `0.0596` | `0.0` | `0.0` |
| V6 feature-route policy | `15.61%` | `16.44%` | `+0.00` pts | `0.0602` | `69.0` | `5.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V6_90PCT_STRONG_SEED313.md`

Interpretation:

Seed 313 is not the live-repair branch V6 was designed to fix; it is a guardrail test. Magnitude has the best centered-CLS alignment, the best masked-before accuracy, and the best after-FT recovery. V6 correctly leaves the sparse template alone rather than forcing liveness repair. This supports the current biological interpretation: circuit remodeling is conditional, not a blanket rule to eliminate every measured dead row.

V6 live-repair miss:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `11.10%` | `12.42%` | baseline | `0.0351` | `77.0` | `4.0` |
| global SynFlow | `4.34%` | `9.55%` | `-2.87` pts | `-0.0314` | `105.0` | `80.0` |
| minimal liveness repair | `11.29%` | `12.34%` | `-0.08` pts | `0.0351` | `1.0` | `0.0` |
| attention+MLP/readout repair | `11.34%` | `12.39%` | `-0.03` pts | `0.0337` | `1.0` | `0.0` |
| all-route liveness floor | `11.40%` | `12.34%` | `-0.08` pts | `0.0353` | `0.0` | `0.0` |
| V6 feature-route policy | `11.40%` | `12.34%` | `-0.08` pts | `0.0353` | `0.0` | `0.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V6_90PCT_STRONG_SEED315.md`

Interpretation:

Seed 315 is the live-repair branch failure V6 does not yet solve. The pre-finetune metrics mildly favor all-route liveness, and the mask eliminates measured dead rows, but magnitude still recovers slightly better. This is not a collapse; it is a small but important boundary. The next selector must ask whether the live-repair advantage is large enough to justify disrupting the magnitude template.

V7 SynFlow no-regression:

| Method | Before FT | After FT | Delta vs magnitude | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `8.66%` | `9.01%` | baseline | `-0.0134` | `66.0` | `5.0` |
| global SynFlow | `8.96%` | `11.93%` | `+2.92` pts | `0.0246` | `74.0` | `77.0` |
| minimal liveness repair | `9.09%` | `9.20%` | `+0.19` pts | `-0.0110` | `1.0` | `0.0` |
| attention+MLP/readout repair | `9.00%` | `9.16%` | `+0.15` pts | `-0.0118` | `1.0` | `0.0` |
| all-route liveness floor | `9.10%` | `9.27%` | `+0.26` pts | `-0.0110` | `0.0` | `0.0` |
| V7 feature-route policy | `8.96%` | `11.93%` | `+2.92` pts | `0.0246` | `74.0` | `77.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_FEATURE_ROUTE_MARGIN_SELECTOR_V7_90PCT_STRONG_SEED320.md`

Interpretation:

Seed 320 is not the exact V7 magnitude-vs-live-repair guardrail branch. It is a no-regression test. The new guardrail leaves the SynFlow feature branch untouched, and SynFlow is again the best evaluated candidate. The missing prospective test remains a fresh seed where V7 actually invokes `magnitude_live_repair_tiny_feature_guardrail`.

V7 guardrail branch scan:

- Scanner artifact: `results/04_criticality_pruning/find_tiny_v7_magnitude_live_guardrail_seed.json`
- Seeds scanned: `321-326`
- Found `magnitude_live_repair_tiny_feature_guardrail`: `false`
- Observed branches: SynFlow feature, SynFlow masked-recovery-prior, and magnitude feature.

Interpretation:

The exact V7 guardrail branch is not frequent in the first fresh scan. That matters for claim discipline: V7 has a clean boundary projection and a SynFlow no-regression seed, but not yet a prospective full-run validation of the new guardrail itself.

## Mechanism hierarchy

The current evidence supports a hierarchy:

![Circuit-viability hierarchy](../figures/04_circuit_viability/figure_04_mechanism_hierarchy.png)

1. **Liveness/homeostasis:** prevent dead layers, dead outputs, and zero-capacity bridges.
2. **Route targets:** preserve projection/readout/main-path capacity where simple liveness is insufficient.
3. **Degeneracy:** avoid overconcentrating capacity into one route family.
4. **Prediction:** choose constraints from pre-finetuning route deficits rather than post-hoc sweeps.

This hierarchy is the main scientific contribution so far.

## Unified selector

The latest synthesis is a two-branch selector over intervention families:

1. If magnitude has a dead route floor, use homeostatic/ecology-aware circuit repair.
2. If magnitude retains a live route floor, preserve pretrained feature structure and only repair minimal liveness failures.

Retrospective validation over six real artifacts:

| Case | Route floor | Selected family | Selected method | Delta vs magnitude | Selected dead outputs |
|---|---:|---|---|---:|---:|
| CIFAR-10 ResNet-20 SGD-40 | `0.0000` | ecology selector | ecology policy | `+3.90` pts | `3.0` |
| DeepTinyResNet CIFAR-10 | `0.0002` | ecology selector | ecology policy | `+4.29` pts | `1.0` |
| CIFAR-10 ecology run | `0.0000` | ecology selector | ecology policy | `+2.81` pts | `0.5` |
| CIFAR-100 ecology run | `0.0000` | ecology selector | ecology policy | `+3.68` pts | `1.0` |
| pretrained TinyImageNet `95%` | `3.6139` | feature repair | feature-viability repair | `-0.05` pts | `0.0` |
| pretrained TinyImageNet `99%` | `0.8541` | feature repair | feature-viability repair | `+0.23` pts | `4.0` |

Primary artifact:

- `experiments/04_criticality_pruning/UNIFIED_VIABILITY_SELECTOR_RETROSPECTIVE.md`

Interpretation:

This is the current best compact theory. The project no longer says "always reserve capacity." It says severe pruning should preserve functionally viable circuits, and the kind of viability depends on whether the existing sparse template has already lost its route floor. From-scratch collapse needs homeostatic repair. Pretrained networks need feature-subspace preservation with targeted liveness repair.

### Selector failure: feature repair can also help from scratch

A fresh full-CIFAR ResNet-20-style comparison tested feature-preserving liveness repair against homeostatic reserve in a from-scratch setting where magnitude has a dead route floor.

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `45.91%` | baseline | baseline | `0.0000` | `3.7702` | `364.0` |
| feature-viability repair | `49.26%` | `+3.35` pts | `2/2` | `0.6041` | `3.7490` | `35.5` |
| plain reserve | `47.78%` | `+1.87` pts | `2/2` | `1.2301` | `3.3411` | `1.0` |
| predicted route split | `47.33%` | `+1.42` pts | `2/2` | `1.0256` | `3.6328` | `1.0` |
| unified policy | `47.78%` | `+1.87` pts | `2/2` | `1.2301` | `3.3411` | `1.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_FULL_RESNET20_FEATURE_VS_HOMEOSTASIS_99PCT_SGD20.md`

Interpretation:

This is an important correction. The simple route-floor family selector is not enough. Even when magnitude has a dead route floor, feature repair can preserve enough useful structure to outperform full homeostatic reserve, despite leaving more dead outputs. The next selector must optimize a tradeoff: how much liveness repair is necessary before feature preservation becomes more valuable than eliminating every dead output.

### Tradeoff selector

The corrected selector now ranks realized candidate masks by a pre-finetune tradeoff score:

- feature overlap with the magnitude template;
- main/projection route liveness;
- readout preservation;
- dead-output penalty.

It does not use post-finetune accuracy to choose the method.

Fresh full-CIFAR ResNet-20-style validation:

| Method | After FT | Delta vs magnitude | Wins | Main min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `44.60%` | baseline | baseline | `0.0000` | `3.7457` | `373.0` |
| feature-viability repair | `49.97%` | `+5.37` pts | `2/2` | `0.5827` | `3.7182` | `32.0` |
| plain reserve | `49.13%` | `+4.54` pts | `2/2` | `1.2483` | `3.2587` | `1.5` |
| predicted route split | `48.51%` | `+3.91` pts | `2/2` | `1.0395` | `3.6272` | `1.0` |
| tradeoff policy | `49.97%` | `+5.37` pts | `2/2` | `0.5827` | `3.7182` | `32.0` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_FULL_RESNET20_TRADEOFF_SELECTOR_99PCT_SGD20.md`

Interpretation:

This is the first prospective correction to the selector failure. The policy selected feature repair on both fresh seeds before fine-tuning, and that selected method beat both magnitude and broad reserve on mean. The result sharpens the neuroscience analogy: circuit viability is not just eliminating silence. It is preserving enough communication for recovery while avoiding unnecessary destruction of already useful computation.

## What is not solved

This is not yet a general pruning solution.

Current limitations:

- Most evidence is CIFAR/Fashion-MNIST scale.
- Residual evidence uses custom TinyResNet variants, not standard ResNet-18/34.
- The diversity optimizer still has hand-set penalty weights.
- The new tradeoff selector is also hand-weighted and has only a two-seed prospective validation so far.
- Transformer/LLM route analogues are not tested.
- The theory predicts useful constraints qualitatively, but not yet from first principles.

## Strong public claim boundary

Safe claim:

> Severe pruning can create hidden circuit cutsets. Capacity constraints inspired by circuit viability prevent these failures and can improve extreme-sparsity recovery across CNN and residual settings. The evidence supports a hierarchy: preserve liveness first, then route-family balance and degeneracy.

Unsafe claim:

> Path-Capacity Pruning solves pruning generally.

## Next experiments

1. Standard ResNet-style benchmark.
2. More seeds for the diversity optimizer.
3. Derive penalty weights from route-family sensitivity.
4. Replicate the tradeoff selector across pretrained TinyImageNet and CIFAR-100.
5. Activation-informed route quality beyond naive presynaptic weighting.
6. Transformer route-family analogues.

## Working title candidates

- **Pruning Should Preserve Circuits, Not Just Synapses**
- **From Synaptic Saliency to Circuit Viability**
- **Path-Capacity Constraints for Severe Neural Network Pruning**
- **Circuit Viability as a Constraint for Neural Network Pruning**


## Appendix A: Neuroscience framing

Merged from the former `docs/NEUROSCIENCE_FRAMING.md` (2026-08-09 consolidation); this report is now the canonical home of the biology-to-algorithm mapping.

This repo's central idea is not simply that biological brains prune synapses. The stronger idea is that biological pruning is constrained by circuit viability.

### Core translation

Biological development removes many synapses, but it does not optimize isolated synapse deletion while ignoring whether the organism can still perceive, act, and learn. Useful circuits must remain viable while redundant or weak connections are removed.

The machine-learning analogue is severe neural-network pruning:

- a weight is a synapse;
- a channel, hidden unit, attention head, or residual block is a circuit element;
- a layer-to-layer path family is a communication route;
- a global saliency threshold is unconstrained synapse deletion;
- a zero-capacity bridge is a circuit lesion;
- path capacity is the minimum viable communication substrate left after pruning.

### Biological principles and algorithmic counterparts

| Neuroscience principle | Biological meaning | Pruning counterpart | Current repo evidence |
|---|---|---|---|
| Synaptic pruning | Remove weak or redundant synapses. | Use saliency or magnitude to remove individual weights. | Magnitude and SynFlow baselines. |
| Circuit viability | Preserve functional routes through a circuit. | Maintain nonzero capacity across required communication cuts. | SynFlow bridge collapse and capacity rescue experiments. |
| Homeostatic plasticity | Prevent circuits from becoming silent or unstable. | Prevent dead outputs, dead bridges, and zero-capacity cutsets. | Dense bridge and TinyResNet dead-output diagnostics. |
| Use-dependent stabilization | Active pathways are more likely to survive. | Use activation-supported path ranking. | TinyResNet activation reserve is a negative first test. |
| Degeneracy | Multiple different routes can support similar function. | Preserve route diversity rather than a single surviving wire. | Diversity route optimization and strong TinyViT branch selection. |
| Communication backbones | Long-range/bottleneck routes support system-level function. | Estimate min-cut or path capacity between representation stages. | Current path-capacity constraints approximate this in CNNs. |

### What the current experiments show

The SynFlow pathology is a machine analogue of a circuit lesion.

Global SynFlow preserves some high-scoring synapses but deletes the dense bridge that carries representation into the classifier. Fine-tuning cannot recover because the mask has removed the route. The problem is not only that there are too few weights. The problem is that the surviving weights no longer form a viable circuit.

Path-capacity pruning adds a viability constraint. It still uses saliency, but saliency is no longer allowed to delete every route through a required communication cut.

### What the current experiments do not show

The current method is not yet a full biological theory of pruning.

Limitations:

- The best evidence is still mostly CNN severe-sparsity work.
- Residual networks reveal that output liveness alone is not route quality.
- Naive presynaptic activation support did not solve the residual case.
- The repo does not yet model richer biological mechanisms such as local competition, neuromodulation, dendritic compartment constraints, or multi-timescale plasticity.
- The transformer analogue is currently TinyViT-scale; it has branch evidence, not broad transformer or LLM transfer.

### Important lesson from the negative activation result

A simple use-dependent rule is not enough:

> saliency multiplied by presynaptic activation did not fix TinyResNet 99% and underperformed plain reserve capacity.

That is useful. It means the right neuroscience translation is not merely “active inputs survive.” The stronger translation is probably:

- active routes must connect through complete downstream paths;
- residual additions create alternate communication geometry;
- route diversity matters more than local activation;
- useful capacity should be measured after block composition, not only at individual weight tensors.

### Important lesson from residual route-quality audits

TinyResNet shows why the circuit analogy has to be graph-level. At the `99%` sparsity cliff, keeping outputs alive is not enough. The route-quality audit found that projection-route capacity correlates more strongly with recovery than total dead-output count, but the projection-backbone allocator also showed that overprotecting one route family can hurt by starving other routes.

The biological analogy is not “protect the biggest tract.” It is balanced circuit remodeling:

- preserve projection/backbone routes;
- preserve main-path transformation capacity;
- preserve classifier readout capacity;
- avoid concentrating all surviving capacity into one route family;
- maintain enough degeneracy that fine-tuning can reroute function.

That suggests the next algorithm should optimize route balance across interacting paths, not independent layer liveness.

The projection/readout split sweep is the first concrete support for that stronger interpretation. The successful TinyResNet `99%` split did not merely keep units alive. It protected projection routes and classifier readout together, while leaving enough main-path capacity for transformation. That is closer to the biological picture: functional remodeling preserves interacting route families rather than one isolated tract.

The predicted route-deficit split is the next step toward making this less post-hoc. It uses the geometry of the candidate sparse circuit before recovery training to estimate which route families are under-supported. The first predictor is crude, but it moves the method toward the biological analogy: remodeling decisions should be driven by measured circuit deficits, not by a fixed layer recipe.

The diversity-penalized route optimizer strengthens the biological connection further. Matching route targets alone overprotected projection routes, which is analogous to preserving one tract while damaging the broader circuit. Adding a degeneracy-style penalty against route-family overconcentration produced the best optimizer-style residual result so far. This makes degeneracy operational: a sparse circuit should preserve multiple interacting route families, not maximize a single bottleneck score.

The DeepTinyResNet transfer adds a second residual setting. In the deeper model, magnitude itself leaves hundreds of dead outputs at `99%` sparsity, while path-capacity methods preserve liveness and beat magnitude. The four-seed replicate narrows the interpretation: broad homeostatic capacity is the dominant gain in the deeper model, while the diversity optimizer is only weakly positive. That strengthens the homeostasis analogy: under extreme synapse loss, the first requirement is to prevent large regions of the circuit from going silent; finer route-family optimization matters after that liveness constraint is satisfied.

The ResNet-20-style transfer reinforces the same point in a more standard residual architecture. Magnitude and SynFlow both leave large numbers of dead outputs at `99%`, while capacity reserve keeps route families alive and beats magnitude. This is the clearest current machine-learning analogue of homeostatic stabilization: the sparse circuit must remain globally viable before finer-grained route optimization can matter.

The full-CIFAR ResNet-20 result makes the homeostasis claim more realistic. The advantage shrinks when dense training is stronger and the full dataset is used, but it remains positive: reserve capacity eliminates dead outputs and still beats magnitude. That is the right kind of result for a biological analogy: homeostatic viability is not magic performance; it is a guardrail that matters most when severe perturbation would otherwise silence parts of the circuit.

### Important lesson from TinyViT residual-stream branch selection

TinyViT makes the neuroscience connection sharper because it separates anatomical liveness from functional viability.

In the strong TinyViT runs, masks that eliminate measured dead rows can still remain at the magnitude floor. Seed 310 is the cleanest branch example: all-route liveness leaves zero measured MLP/attention dead rows but recovers only `5.86%`, while SynFlow keeps many dead rows by that metric and recovers `15.16%`. Seed 311 repeats the pattern without branch scanning: all-route liveness reaches `9.97%`, while SynFlow reaches `15.91%`. Seed 312 is the counterweight: SynFlow is correctly rejected, but the selector chooses the wrong live repair by overtrusting a tiny feature margin. V6 converts that failure into a rule: when live-repair feature margins are tiny, use masked-before trainability as the tie-breaker. Seed 313 then checks the other guardrail: V6 keeps magnitude when magnitude is the most trainable sparse template. Seed 315 shows the next limitation: even zero-dead all-route liveness can be slightly worse than magnitude when the measured repair advantage is too small. V7 adds that magnitude-vs-repair guardrail, and seed 320 checks that this new guardrail leaves the SynFlow feature branch intact.

The biological analogy is that keeping every local pathway anatomically present is not the same as preserving the functional ensemble. A sparse transformer circuit needs:

- residual-stream feature preservation;
- globally coherent signal paths;
- enough trainable capacity for masked recovery;
- liveness repair only when silence is the actual limiting deficit.

That is why the current theory is not simply "keep units alive." It is constrained circuit remodeling: preserve the route family that keeps the computation recoverable under the task ecology.

### Next neuroscience-grounded method hypothesis

The next method should model pruning as circuit-preserving remodeling:

1. Score synapses locally.
2. Identify vulnerable communication cuts.
3. Estimate route capacity after candidate pruning.
4. Preserve multiple viable routes through each cut.
5. Prefer routes supported by activation and downstream reachability.
6. Penalize masks that concentrate all capacity into brittle single paths.

This would map more faithfully to biological ideas:

- homeostasis: every region keeps enough viable capacity;
- use dependence: active routes get preference;
- degeneracy: multiple routes survive;
- lesion avoidance: no required cut collapses;
- efficiency: redundant synapses are still removed.

### Working thesis

Pruning should not be framed as choosing the best isolated synapses. It should be framed as constrained circuit remodeling:

> maximize synaptic efficiency subject to preserving viable, diverse, task-relevant communication routes.

That is the neuro-AI idea this repo should make real.

## Appendix B: Route-deficit predicted capacity

Merged from the former `docs/ROUTE_DEFICIT_PREDICTOR.md` (2026-08-09 consolidation); this report is now the canonical home of the residual route-deficit predictor evidence.

This note documents the current residual-network version of Path-Capacity Pruning.

### Motivation

Plain output-count capacity reserve fixes the obvious death problem in TinyResNet masks, but it still fails at the `99%` sparsity cliff. The route-quality audit showed why: once most outputs are technically alive, raw dead-output count stops explaining recovery. Residual recovery depends on route-family balance across:

- main transformation paths;
- projection shortcuts;
- classifier readout.

The goal is to select a capacity split before recovery fine-tuning, not by reading the final accuracy table.

### Predictor

Implementation:

- `shared/residual_route_capacity.py`

The first predictor does this:

1. Build a magnitude mask as a viability template.
2. Build a plain reserve mask as the candidate that prevents death but underperforms.
3. Compute route-quality diagnostics for both masks.
4. Measure projection deficit: template projection capacity minus candidate projection capacity.
5. Measure readout deficit: template classifier-readout score minus candidate classifier-readout score.
6. Keep a main-path floor.
7. Allocate remaining protected capacity between projection and readout deficits.
8. Build a route-family capacity mask and fill the rest globally by SynFlow.

The current predictor still contains two hand-set choices:

- main-path floor: `0.40`
- projection reliability weight: `2.0`

These are not final theory. They are scaffolding for converting the tuned route split into a measured-deficit rule.

### Current evidence

Four TinyResNet `99%` seeds:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `25.04%` | baseline | baseline |
| plain reserve `0.60` | `20.21%` | `-4.83` pts | `0/4` |
| tuned `40/35/25` split | `24.88%` | `-0.17` pts | `3/4` |
| predicted route-deficit split | `25.12%` | `+0.08` pts | `2/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_PREDICTED_ROUTE_SPLIT_99PCT.md`
- `results/04_criticality_pruning/cifar10_tiny_resnet_predicted_route_split_99pct.json`

### Interpretation

The effect size is tiny. The result should not be sold as a solved residual pruning method.

The important step is methodological:

- output liveness was insufficient;
- route-quality audit identified projection/readout imbalance;
- a tuned projection/readout split beat magnitude;
- a measured-deficit predictor recovered nearly the same split before fine-tuning and narrowly beat magnitude on mean.

This makes the neuroscience story more concrete. Circuit viability is not a slogan here; it becomes a pre-recovery route-deficit measurement that changes the mask.

### Next step

The predictor needs to stop relying on fixed constants.

Concrete next experiments:

1. Derive the main-path floor from parameter budget and block depth.
2. Derive projection reliability weight from route-quality correlations rather than setting it manually.
3. Validate on more TinyResNet seeds.
4. Transfer to a stronger residual backbone.
5. Add transformer route-family analogues once residual prediction is stable.

### Derived predictor follow-up

A fresh four-seed follow-up tested whether the remaining constants could be removed or weakened.

Compared methods:

- `tuned_40_35_25`: hand-selected route-family split.
- `fixed_deficit_predictor`: the first route-deficit predictor with main floor and projection reliability weight.
- `relative_deficit_predictor`: equal route-family priors, split by relative route deficits.
- `sqrt_width_deficit_predictor`: route-family prior derived from square root of output width, then adjusted by deficits.

Fresh TinyResNet `99%` seeds `[207, 208, 209, 210]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `24.98%` | baseline | baseline |
| plain reserve `0.60` | `20.70%` | `-4.28` pts | `0/4` |
| tuned `40/35/25` | `24.61%` | `-0.37` pts | `2/4` |
| fixed deficit predictor | `24.59%` | `-0.39` pts | `2/4` |
| relative deficit predictor | `23.81%` | `-1.17` pts | `1/4` |
| sqrt-width deficit predictor | `22.75%` | `-2.23` pts | `0/4` |

Interpretation:

- Removing the constants did not improve the method.
- The equal-family predictor overallocated readout and increased dead outputs.
- The width-derived predictor underallocated readout and performed worse.
- The fixed predictor and tuned split still close most of the plain-reserve gap, but they do not robustly beat magnitude on fresh seeds.

Current honest status:

**Route-deficit prediction is promising as a diagnostic and gap-closer, but not yet a robust residual pruning method.**

Next step:

Rather than guessing priors, derive the split from an optimization objective over route-quality targets, then solve for the cheapest capacity allocation that matches those targets.

### Target-matched optimizer follow-up

A first optimizer-style route allocator searched route-family splits before fine-tuning. The objective matched magnitude's projection/readout route targets while preserving the plain-reserve main-path floor.

Fresh TinyResNet `99%` seeds `[211, 212, 213, 214]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `25.79%` | baseline | baseline |
| plain reserve `0.60` | `18.74%` | `-7.05` pts | `0/4` |
| tuned `40/35/25` | `24.43%` | `-1.36` pts | `0/4` |
| fixed deficit predictor | `23.59%` | `-2.21` pts | `0/4` |
| target-matched optimizer | `24.52%` | `-1.27` pts | `1/4` |

The optimizer selected `20/55/25` main/projection/readout on every seed. That improved over plain reserve but did not beat magnitude.

Interpretation:

- Matching projection/readout targets is not enough.
- The optimizer overconcentrated protected capacity into projection routes.
- It increased dead outputs relative to tuned/fixed predictors.
- The missing term is a route-diversity or overconcentration penalty.

Next optimization objective:

**Match projection and readout targets while penalizing route-family overconcentration and preserving main-path diversity.**

### Diversity-penalized optimizer

A follow-up optimizer added route-family concentration and projection-overuse penalties. This directly tested the degeneracy hypothesis: do not allow the allocator to satisfy one route target by collapsing route-family diversity.

Fresh TinyResNet `99%` seeds `[215, 216, 217, 218]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `24.80%` | baseline | baseline |
| plain reserve `0.60` | `20.13%` | `-4.67` pts | `0/4` |
| tuned `40/35/25` | `26.04%` | `+1.25` pts | `3/4` |
| diversity target optimizer | `25.85%` | `+1.05` pts | `4/4` |

The optimizer selected `25/50/25` main/projection/readout on every seed. It still emphasizes projection, but the diversity penalty prevents the previous `20/55/25` collapse and improves seed stability.

Interpretation:

- The degeneracy constraint worked.
- The optimizer now beats magnitude on every seed in this batch.
- It is slightly below the tuned split on mean, but has stronger paired consistency.
- This is the best current evidence that route-quality optimization can become a method, not just a diagnostic.

Remaining caveat:

The penalty weights are still hand-set. The next step is to derive them from route-family sensitivity or validate them on a stronger residual backbone.

## Appendix C: Consolidated severe-sparsity results table

Moved from the former README results section (2026-08-09 consolidation). Every row is backed by a JSON artifact in `results/04_criticality_pruning/` and checked by `audit_circuit_viability_claims.py`.

The pathology becomes useful when it is turned into a circuit-viability constraint.

Current selector lesson:

1. If the sparse template is topologically dead, some liveness/homeostatic repair is needed.
2. If the sparse template preserves useful computation, broad homeostatic reserve can be too destructive.
3. The selector must score the tradeoff between feature preservation, route liveness, readout preservation, and dead-output repair.

The old route-floor rule selected the intended family on `6/6` retrospective artifacts, but a fresh full-CIFAR experiment exposed its limitation: feature-preserving repair can beat broad reserve even when magnitude has a dead main-route floor. The current tradeoff selector fixes that failure in a fresh two-seed prospective test.

Current best constructive result:

| Case | Sparsity | Magnitude after FT | Global SynFlow after FT | Capacity method after FT | Delta vs magnitude |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 CNN reserve sweep best | `99%` | `32.31%` | `9.76%` | `34.88%` | `+2.56` pts |
| Fashion-MNIST CNN transfer | `99%` | `80.24%` | `9.89%` | `81.75%` | `+1.51` pts |
| TinyResNet transfer | `98%` | `31.32%` | `9.79%` | `32.27%` | `+0.95` pts |
| TinyResNet backbone reserve | `98%` | `29.43%` | `9.79%` | `32.19%` | `+2.76` pts |
| TinyResNet transfer | `99%` | `24.23%` | `10.10%` | `21.32%` | `-2.91` pts |
| TinyResNet backbone reserve | `99%` | `24.45%` | `10.10%` | `17.75%` | `-6.70` pts |
| TinyResNet balanced route | `99%` | `24.31%` | `10.10%` | `25.68%` | `+1.37` pts |
| TinyResNet balanced route replicate | `99%` | `24.93%` | `10.13%` | `23.40%` | `-1.53` pts |
| TinyResNet projection/readout split | `99%` | `24.89%` | not rerun | `25.63%` | `+0.74` pts |
| TinyResNet predicted deficit split | `99%` | `25.04%` | not rerun | `25.12%` | `+0.08` pts |
| TinyResNet fixed predictor fresh | `99%` | `24.98%` | not rerun | `24.59%` | `-0.39` pts |
| TinyResNet target-matched optimizer | `99%` | `25.79%` | not rerun | `24.52%` | `-1.27` pts |
| TinyResNet diversity optimizer | `99%` | `24.80%` | not rerun | `25.85%` | `+1.05` pts |
| DeepTinyResNet diversity optimizer | `99%` | `26.40%` | `9.96%` | `29.14%` | `+2.74` pts |
| DeepTinyResNet reserve replicate | `99%` | `28.48%` | `9.98%` | `30.20%` | `+1.72` pts |
| ResNet-20-style reserve | `99%` | `28.48%` | `9.99%` | `32.41%` | `+3.93` pts |
| ResNet-20-style reserve replicate | `99%` | `27.25%` | `10.03%` | `33.22%` | `+5.97` pts |
| Full CIFAR ResNet-20-style reserve | `99%` | `37.72%` | `10.00%` | `39.21%` | `+1.50` pts |
| Full CIFAR ResNet-20-style SGD recipe | `99%` | `42.87%` | `10.00%` | `49.43%` | `+6.57` pts |
| Full CIFAR ResNet-20-style predicted split | `99%` | `42.09%` | not rerun | `46.92%` | `+4.82` pts |
| Full CIFAR-100 ResNet-20-style reserve | `99%` | `6.58%` | `1.00%` | `7.64%` | `+1.06` pts |
| Full CIFAR-100 readout-main route split | `99%` | `6.53%` | not rerun | `9.12%` | `+2.59` pts |
| Full CIFAR-100 conservative predicted split | `99%` | `6.98%` | not rerun | `8.79%` | `+1.81` pts |
| Full CIFAR-100 tradeoff selector V1 | `99%` | `6.98%` | not rerun | `8.21%` | `+1.24` pts |
| Full CIFAR-100 tradeoff selector V2 policy | `99%` | `6.98%` | not rerun | `8.87%` | `+1.89` pts |
| Full CIFAR-100 tradeoff selector V2 prospective | `99%` | `6.54%` | not rerun | `9.26%` | `+2.72` pts |
| Ecology selector CIFAR-10 selected reserve | `99%` | `43.93%` | not rerun | `46.74%` | `+2.81` pts |
| Ecology selector CIFAR-100 selected split | `99%` | `5.41%` | not rerun | `9.08%` | `+3.68` pts |
| DeepTinyResNet ecology selector policy | `99%` | `27.21%` | not rerun | `31.50%` | `+4.29` pts |
| Full CIFAR ResNet-20 ecology selector SGD-40 | `99%` | `48.73%` | not rerun | `52.63%` | `+3.90` pts |
| TinyViT minimal liveness repair | `98%` | `10.07%` | `11.00%` | `11.31%` | `+1.24` pts |
| TinyViT selective MLP/readout repair | `98%` | `10.07%` | `11.00%` | `9.59%` | `-0.48` pts |
| TinyViT all-route liveness floor | `98%` | `10.07%` | `11.00%` | `9.96%` | `-0.11` pts |
| TinyViT MLP/readout reserve | `98%` | `10.07%` | `11.00%` | `9.95%` | `-0.12` pts |
| TinyViT minimal liveness repair | `95%` | `9.87%` | `10.76%` | `10.87%` | `+1.00` pts |
| TinyViT attention+MLP/readout repair | `95%` | `9.87%` | `10.78%` | `10.31%` | `+0.44` pts |
| TinyViT feature-subspace diagnostic SynFlow | `95%` | `11.02%` | `16.86%` | `16.86%` | `+5.84` pts |
| TinyViT feature-subspace selector | `95%` | `8.82%` | `13.57%` | `12.24%` | `+3.42` pts |
| TinyViT feature-route margin policy | `95%` | `8.82%` | `13.57%` | `13.57%` | `+4.75` pts |
| TinyViT feature-route margin selector | `95%` | `8.95%` | `14.60%` | `14.60%` | `+5.65` pts |
| TinyViT feature-route margin selector | `90%` | `10.32%` | `14.56%` | `14.56%` | `+4.24` pts |
| TinyViT feature-route margin strong pilot | `90%` | `16.70%` | `14.35%` | `14.35%` | `-2.35` pts |
| TinyViT feature-route margin V2 strong | `90%` | `12.18%` | `10.25%` | `12.18%` | `+0.00` pts |
| TinyViT V2 strong replicate | `90%` | `13.82%` | `14.24%` | `13.82%` | `+0.00` pts |
| TinyViT V3 strong selector | `90%` | `11.39%` | `14.52%` | `14.52%` | `+3.13` pts |
| TinyViT V3 strong replicate | `90%` | `8.07%` | `15.05%` | `15.05%` | `+6.98` pts |
| TinyViT V4 strong selector | `90%` | `10.22%` | `17.09%` | `17.09%` | `+6.87` pts |
| TinyViT V4 seed 306 | `90%` | `10.99%` | `13.29%` | `11.58%` | `+0.59` pts |
| TinyViT V5 strong selector | `90%` | `7.25%` | `10.18%` | `10.18%` | `+2.93` pts |
| TinyViT V5 SynFlow-prior branch | `90%` | `5.86%` | `15.16%` | `15.16%` | `+9.30` pts |
| TinyViT V5 unselected fresh seed | `90%` | `9.79%` | `15.91%` | `15.91%` | `+6.12` pts |
| TinyViT V5 live-repair miss | `90%` | `11.89%` | `10.68%` | `11.87%` | `-0.02` pts |
| TinyViT V6 magnitude guardrail | `90%` | `16.44%` | `10.10%` | `16.44%` | `+0.00` pts |
| TinyViT V6 live-repair miss | `90%` | `12.42%` | `9.55%` | `12.34%` | `-0.08` pts |
| TinyViT V7 SynFlow no-regression | `90%` | `9.01%` | `11.93%` | `11.93%` | `+2.92` pts |
| TinyImageNet-200 external proxy ecology selector | `99%` | `2.32%` | not rerun | `2.90%` | `+0.58` pts |
| TinyImageNet pretrained ResNet-18 ecology selector | `99%` | `1.07%` | not rerun | `0.80%` | `-0.27` pts |
| TinyImageNet pretrained ResNet-18 ecology selector | `95%` | `15.23%` | not rerun | `3.97%` | `-11.27` pts |
| TinyImageNet pretrained feature-viability repair | `95%` | `15.13%` | not rerun | `15.03%` | `-0.10` pts |
| TinyImageNet pretrained feature-viability repair replicate | `95%` | `14.87%` | not rerun | `14.82%` | `-0.05` pts |
| TinyImageNet pretrained feature-viability repair | `99%` | `1.20%` | not rerun | `1.43%` | `+0.23` pts |
| TinyImageNet pretrained tradeoff selector | `95%` | `15.67%` | not rerun | `15.70%` | `+0.03` pts |
| Full CIFAR feature repair vs homeostasis | `99%` | `45.91%` | not rerun | `49.26%` | `+3.35` pts |
| Full CIFAR tradeoff selector | `99%` | `44.60%` | not rerun | `49.97%` | `+5.37` pts |

Interpretation:

- Capacity constraints reliably prevent the obvious topological death caused by global SynFlow.
- On CNN dense-tail settings, capacity reserve can improve extreme-sparsity recovery over magnitude.
- On TinyResNet at `99%`, output liveness is not enough; residual-route quality needs a stronger constraint.
- A naive activation-supported reserve was tested and did not solve the residual case.
- Projection-backbone protection improves TinyResNet `98%` but worsens `99%`, so shortcut protection alone is not the full residual theory.
- Balanced residual route allocation narrowed the TinyResNet `99%` failure and beat plain reserve, but its four-seed replicate still trails magnitude.
- Projection/readout-balanced allocation is the current residual winner: it beats magnitude by `+0.74` points and wins `3/4` fresh seeds at TinyResNet `99%`.
- The first route-deficit predictor is tiny and mixed: one batch narrowly beats magnitude, a fresh batch trails slightly, but both close most of the plain-reserve gap.
- Target matching without a diversity penalty overconcentrates projection capacity and still trails magnitude.
- Adding a degeneracy-style diversity penalty fixes that failure in the latest batch: the optimizer beats magnitude by `+1.05` points and wins `4/4` seeds.
- Stronger residual transfer is positive: on DeepTinyResNet, path-capacity methods beat magnitude across the replicate, with plain reserve strongest in the four-seed run.
- Standard-style residual transfer is strongly positive: on a CIFAR ResNet-20-style replicate, plain reserve beats magnitude by `+5.97` points with `4/4` wins.
- Full CIFAR-10 transfer remains positive on the short recipe: reserve beats magnitude by `+1.50` points with `6/6` wins.
- Under a stronger full-CIFAR SGD/cosine recipe, the effect gets larger: reserve beats magnitude by `+6.57` points with `4/4` wins.
- The conservative predictor adapts back to CIFAR-10 by choosing main/projection-heavy splits and still beats magnitude by `+4.82` points, but plain reserve remains better on this task ecology.
- CIFAR-100 is much harder in absolute recovery, but the mechanism transfers: reserve beats magnitude by `+1.06` points, and a readout/main-biased split improves the gain to `+2.59` points with `2/2` wins.
- The new CIFAR-100 result suggests output diversity changes the viable circuit constraint: class-rich tasks need stronger readout preservation, not only generic liveness.
- A conservative pre-finetune route-deficit selector now predicts the CIFAR-100 split without using recovery labels: `+1.81` points over magnitude and better than plain reserve on both fresh seeds.
- The tradeoff selector exposed a sharper CIFAR-100 boundary: V1 overweights feature preservation, but a V2 output/readout-pressure policy selects route split. In a fresh prospective run, V2 selects route split on both seeds and beats magnitude by `+2.72` points, matching the best mean candidate.
- An ecology-aware selector now chooses the intervention family from readout deficit: it keeps broad reserve on CIFAR-10 and switches to predicted split on CIFAR-100, winning `2/2` seeds on both tasks.
- The same selector transfers to a deeper residual backbone: on DeepTinyResNet it chooses broad reserve from the high readout ratio and beats magnitude by `+4.29` points with `2/2` wins.
- Under a longer 40-epoch dense CIFAR-10 schedule, the fixed selector still chooses broad reserve and beats magnitude by `+3.90` points; the route split is negative, so family selection matters.
- The first TinyViT analogue is mixed but real: magnitude kills all tested MLP down-projection rows and many attention-output rows. But the feature-subspace diagnostic changes the transformer direction: global SynFlow beats magnitude by `+5.84` points while liveness repairs underperform, and centered CLS alignment correlates with recovery. Fresh prospective feature-route margin selectors choose SynFlow and beat magnitude in weak TinyViT runs at `95%` and `90%`. Stronger full-train TinyViT pilots show the missing term: V1 overselects SynFlow, V2 fixes one seed by selecting magnitude, a V2 replicate shows all-route liveness can be best, and V3 identifies the feature-dominant strong regime. Across three fresh strong V3 seeds, the feature-margin branch selects SynFlow and beats magnitude; the two-seed replicate improves mean recovery by `+6.98` points with `2/2` wins. Strong-transformer transfer still needs a predictive three-way policy over residual-stream representation, route liveness, and trainable capacity.
- Across twelve completed strong TinyViT seeds, V4 is positive vs magnitude on `8/12`, matches the best evaluated candidate on `9/12`, and averages `+2.88` points over magnitude. Seed 306 and seed 310 define the important failure boundary: V4-style live repairs can beat or match magnitude while still missing a much more trainable SynFlow basin.
- V5 adds a SynFlow masked-recovery prior: if SynFlow's masked-before accuracy is at least magnitude and close to the selected repair, prefer SynFlow. Across the twelve completed strong TinyViT seeds, V5 is positive vs magnitude on `9/12`, matches the best evaluated candidate on `11/12`, averages `+3.80` points over magnitude, and leaves only `0.02` points mean gap to the evaluated oracle.
- Four fresh V5 prospective seeds now test both branches plus unselected follow-ups. Seed 308 validates the feature/SynFlow branch at `+2.93` points over magnitude. Seed 310 validates the SynFlow masked-recovery-prior branch at `+9.30` points over magnitude while every liveness-first repair stays at the magnitude floor. Seed 311 was run without scanner selection and again chooses SynFlow, winning by `+6.12` points. Seed 312 is the first V5 prospective miss: it chooses attention+MLP repair by a tiny feature margin, but minimal liveness is best and the selected policy is `-0.02` points vs magnitude. This makes the next selector target clear: tie-break inside the live-repair family.
- V6 adds exactly that tie-breaker: when live-repair feature margins are tiny, choose the live repair with stronger masked-before trainability. On the current `14`-seed boundary synthesis, V6 matches the best evaluated candidate on `13/14`, is positive vs magnitude on `10/14`, and averages `+3.27` points over magnitude. Seed 313 is a fresh V6 prospective guardrail run: V6 keeps magnitude, and magnitude is the best evaluated candidate. Seed 315 is the new limitation: V6 selects all-route liveness, but magnitude is still slightly better, so the live-repair branch needs a magnitude-vs-repair guardrail.
- V7 adds that magnitude-vs-live-repair guardrail. On the current `15`-seed boundary synthesis, V7 matches the best evaluated candidate on `15/15`, is positive vs magnitude on `11/15`, and averages `+3.25` points over magnitude. Seed 320 prospectively validates that V7 does not break the SynFlow feature branch.
- On a first TinyImageNet-200 external proxy subset, viability methods beat magnitude but the selector-picked split trails plain reserve; this is a boundary condition, not a solved external benchmark.
- On ImageNet-pretrained ResNet-18 TinyImageNet, the current viability selector fails: at `99%` all methods collapse, and at `95%` magnitude strongly beats the homeostatic masks. This is now the main external-validity limitation.
- A feature-preserving liveness repair fixes most of that failure: across two `95%` seeds it eliminates dead outputs while matching magnitude within `0.05` points on average, showing pretrained systems need minimal liveness repair on top of feature-subspace preservation.
- At the `99%` pretrained TinyImageNet cliff, feature-viability repair is the only positive intervention so far, but absolute recovery remains near chance.
- On a fresh pretrained TinyImageNet `95%` seed, the corrected tradeoff selector picks feature repair, preserves magnitude-level accuracy, eliminates dead outputs, and avoids homeostatic masks that collapse to `3-4%`.
- A fresh full-CIFAR from-scratch comparison complicates the selector: feature repair beats both magnitude and reserve, though it leaves more dead outputs. The family selector must optimize accuracy/liveness tradeoff, not just route-floor class.
- The corrected tradeoff selector now does this prospectively: on two fresh full-CIFAR ResNet-20-style seeds, it selected feature repair before fine-tuning and beat magnitude by `+5.37` points with `2/2` wins, while also beating plain reserve on mean.
