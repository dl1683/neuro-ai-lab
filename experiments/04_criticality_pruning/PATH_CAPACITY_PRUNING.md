# Path-Capacity Pruning

## Thesis

Pruning should preserve circuits, not just synapses.

Most pruning methods rank individual weights by saliency, then apply a global threshold. That is a synapse-level view. The SynFlow pathology experiments in this repo show why that is insufficient: a globally high-scoring mask can preserve many individual weights while deleting an entire functional communication bridge.

Path-Capacity Pruning adds a circuit-viability constraint: use saliency to choose which weights are valuable, but prevent the sparse graph from creating zero-capacity cutsets across required communication routes.

## Neuroscience motivation

Biological pruning is not arbitrary synapse deletion. During development and adaptation, brains remove weak or redundant synapses while preserving viable circuits.

Relevant principles:

- **Use-dependent stabilization:** active functional pathways are more likely to survive.
- **Homeostatic plasticity:** circuits avoid runaway silence or instability.
- **Degeneracy:** multiple different paths can support the same function, but some viable route capacity must remain.
- **Communication backbones:** system-level function depends on preserving bottlenecks and long-range routes.

The machine-learning analogue is direct:

- weights are synapses;
- channels, hidden units, and dense bridges are circuit elements;
- global saliency pruning is unconstrained synapse deletion;
- bridge collapse is circuit death.

## Failure mode discovered

Global SynFlow can allocate zero weights to the first dense classifier bridge in CNNs at severe sparsity.

Cross-dataset synthesis:

| Case | Magnitude after FT | Global SynFlow after FT | Delta | SynFlow `fc1` keep | Dead bridge units |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST CNN, `98%` | `80.86%` | `10.28%` | `-70.59` pts | `0.0000` | `128/128` |
| CIFAR-10 CNN, `98%` | `44.08%` | `9.76%` | `-34.32` pts | `0.0000` | `192/192` |
| CIFAR-10 CNN, `99%` | `33.24%` | `9.76%` | `-23.48` pts | `0.0000` | `192/192` |

This is not merely low accuracy. It is a topological cutset: the classifier bridge has zero capacity.

## First constructive prototype

The first implementation is deliberately simple:

1. Compute a base saliency score, currently SynFlow.
2. Reserve minimum capacity across critical cuts.
3. Use magnitude for dense classifier bridge ranking where SynFlow starves the bridge.
4. Fill the remaining global budget with saliency.
5. Preserve the same total parameter budget as ordinary global pruning.

Implementation artifacts:

- `shared/path_capacity_pruning.py`
- `experiments/04_criticality_pruning/cifar10_cnn_multicut_capacity_pruning.py`
- `results/04_criticality_pruning/cifar10_cnn_multicut_capacity_pruning.json`

## First result

CIFAR-10 CNN, 20k train subset, 5k test subset, four seeds, CUDA, same global parameter budget.

| Sparsity | Method | Before FT | After FT | Dead `fc1` units |
|---:|---|---:|---:|---:|
| `98%` | magnitude | `13.93%` | `44.05%` | `73.5` |
| `98%` | global SynFlow | `9.76%` | `9.76%` | `192.0` |
| `98%` | multi-cut capacity | `13.65%` | `40.09%` | `0.0` |
| `99%` | magnitude | `12.33%` | `32.62%` | `79.5` |
| `99%` | global SynFlow | `9.76%` | `9.76%` | `192.0` |
| `99%` | multi-cut capacity | `11.32%` | `33.41%` | `0.0` |

Interpretation:

- At `98%`, multi-cut capacity converts SynFlow from dead to trainable, but still trails magnitude.
- At `99%`, multi-cut capacity converts SynFlow from dead to trainable and slightly beats magnitude after fine-tuning in this four-seed run.
- The result is early, but constructive: circuit-viability constraints can change a fatal pruning mask into a recoverable one under the same parameter budget.

## What this does not prove yet

This is not yet a mature pruning algorithm.

Current limitations:

- Capacity floors are hand-chosen.
- Evidence is still on a small CNN and CIFAR subset.
- It has not been tested against SNIP, GraSP, ERK, or modern large-model pruning baselines.
- It does not beat magnitude at `98%` after fine-tuning.
- It has not been transferred to ResNets, ViTs, or transformers.

## Next technical step

The next version should replace hand-set floors with a predictive capacity model.

Needed objects:

- score-distribution-based dead-unit probability;
- expected keep rate per layer under global thresholding;
- min-cut capacity between input representation and output classes;
- activation-supported route capacity;
- class reachability after pruning.

A stronger method would solve:

> maximize saliency subject to minimum path capacity across every required communication cut.

## Working hypothesis

Severe global pruning fails when score-thresholded subgraphs create hidden topological cutsets.

Path-Capacity Pruning should be judged by whether it can:

1. predict those cutsets before training recovery;
2. prevent them under the same parameter budget;
3. preserve or improve post-prune recoverability;
4. generalize beyond the discovered SynFlow failure case.

## Adaptive capacity pruning result

A predictor-driven capacity allocator now beats magnitude at the harsher CIFAR `99%` sparsity point:

- magnitude: `32.83%` after FT
- global SynFlow: `9.76%` after FT
- adaptive capacity: `34.65%` after FT

At `98%`, adaptive capacity reaches `41.70%` after FT versus magnitude `44.11%`, so it is not universally better. But it converts SynFlow collapse into trainability and does so using a circuit-viability allocation rule instead of fixed per-layer floors.

## Risk-adaptive capacity negative result

A risk-adaptive capacity allocator was tested against the fixed `0.55` reserve rule. It computed reserve fraction and per-cut allocation from predicted global SynFlow dead-output risk.

Result: it prevented dense bridge death, but did not beat the fixed reserve allocator.

- `98%`: fixed capacity after FT `41.55%`; risk-adaptive `39.49%`; magnitude `43.70%`.
- `99%`: fixed capacity after FT `33.17%`; risk-adaptive `32.22%`; magnitude `32.61%`.

Interpretation: predicted dead-cut risk is necessary but not sufficient. The capacity allocator must account for score mass concentration and route quality, not merely output liveness.

See `experiments/04_criticality_pruning/CIFAR10_CNN_RISK_ADAPTIVE_CAPACITY_PRUNING.md`.

## Fashion-MNIST capacity transfer

Path-capacity reserve pruning transferred back to the original Fashion-MNIST CNN SynFlow-collapse setting.

Best results:

- `98%`: reserve `0.60` after FT `85.14%` vs magnitude `84.79%`, `+0.36` points.
- `99%`: reserve `0.60` after FT `81.75%` vs magnitude `80.24%`, `+1.51` points.
- All reserve variants eliminated dense-bridge death (`0.0` dead `fc1` units), while global SynFlow killed all `128/128` bridge units and stayed near chance.

This strengthens the constructive claim: path capacity is not just a CIFAR-specific repair. It transfers to Fashion-MNIST CNN and is most useful at the harsher sparsity cliff.

## TinyResNet residual transfer

The first residual-network transfer test is mixed and important.

CIFAR-10 TinyResNet, 20k train subset, 5k test subset, two seeds, CUDA:

| Sparsity | Method | After FT | Dead outputs |
|---:|---|---:|---:|
| `98%` | magnitude | `31.32%` | `132.0` |
| `98%` | global SynFlow | `9.79%` | `277.0` |
| `98%` | reserve `0.60` | `32.27%` | `1.0` |
| `99%` | magnitude | `24.23%` | `219.0` |
| `99%` | global SynFlow | `10.10%` | `297.0` |
| `99%` | reserve `0.60` | `21.32%` | `3.0` |

Interpretation:

- The pathology transfers: global SynFlow creates massive output death and stays near chance.
- Capacity reserve transfers partially: at `98%`, reserve `0.60` beats magnitude by `+0.95` points while eliminating almost all dead outputs.
- The current method fails at the harder `99%` residual setting: it prevents output death but trails magnitude by `-2.91` points.
- This is a real limitation, not a contradiction. It shows that output-count capacity is not enough under residual routing. The next method must estimate residual-route quality, activation support, and path diversity rather than only keeping outputs alive.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_CAPACITY_TRANSFER.md`.

## Activation-supported residual capacity

A use-dependent stabilization variant was tested on TinyResNet. It ranks protected capacity by saliency multiplied by presynaptic activation on calibration data.

Result:

| Sparsity | Method | After FT | Delta vs magnitude | Dead outputs |
|---:|---|---:|---:|---:|
| `98%` | reserve `0.60` | `33.35%` | `+3.63` pts | `1.0` |
| `98%` | activation reserve `0.60` | `29.59%` | `-0.13` pts | `1.0` |
| `99%` | reserve `0.60` | `22.10%` | `-2.40` pts | `3.0` |
| `99%` | activation reserve `0.60` | `20.05%` | `-4.45` pts | `2.5` |

Interpretation:

- Naive activity weighting is not the missing residual-network variable.
- It preserves liveness but does not preserve useful residual route quality.
- The neuroscience mapping needs a stronger form than local presynaptic activity. The next candidate should measure block-level route diversity or post-residual communication capacity.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_ACTIVATION_CAPACITY.md`.

## Residual-backbone capacity

A residual-specific variant gave extra protected capacity to projection shortcuts and the classifier route, using magnitude ranking inside those protected budgets.

Result:

| Sparsity | Method | After FT | Delta vs magnitude | Dead outputs |
|---:|---|---:|---:|---:|
| `98%` | reserve `0.60` | `31.46%` | `+2.03` pts | `1.0` |
| `98%` | backbone reserve `0.60` | `32.19%` | `+2.76` pts | `1.0` |
| `99%` | reserve `0.60` | `20.58%` | `-3.87` pts | `3.0` |
| `99%` | backbone reserve `0.60` | `17.75%` | `-6.70` pts | `3.5` |

Interpretation:

- Projection shortcuts do matter: the residual-backbone variant improves the `98%` TinyResNet mean over plain reserve.
- But overprotecting projection shortcuts worsens the `99%` cliff, likely because the parameter budget is so small that extra shortcut capacity steals from other required routes.
- The residual failure is therefore not solved by naming a single backbone cut. It needs a global route allocation model across main paths, shortcuts, and classifier routes.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_BACKBONE_CAPACITY.md`.

## TinyResNet route-quality audit

A route-quality audit was added to explain why residual `99%` remains hard. It computes block-level route metrics for each mask family, then correlates those metrics with after-fine-tuning accuracy.

Key correlations with after-FT accuracy:

| Scope | Route min | Projection min | Dead outputs |
|---|---:|---:|---:|
| `98%` | `+0.852` | `+0.830` | `-0.927` |
| `99%` | `+0.215` | `+0.842` | `-0.370` |
| all | `+0.652` | `+0.807` | `-0.611` |

Interpretation:

- At `98%`, ordinary liveness and route capacity both explain much of the difference between dead SynFlow masks and recoverable capacity masks.
- At `99%`, dead-output count becomes a weak signal. Many masks are technically alive, but recoverability still differs.
- Projection-route capacity becomes the strongest measured correlate at `99%`, but the backbone experiment shows that maximizing projection capacity alone is harmful because it drains capacity from classifier and main routes.
- The residual version of Path-Capacity Pruning should be a balanced route allocator, not a single-cut protector.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_ROUTE_QUALITY_AUDIT.md`.

## Balanced residual route capacity

The route-quality audit suggested a more specific residual method: do not maximize one route family. Balance protected capacity across:

- main transformations;
- projection shortcuts;
- classifier readout.

The first balanced allocator uses a `0.60` reserve split as:

- `50%` main-path convs;
- `25%` projection shortcuts;
- `25%` classifier readout.

Result:

| Sparsity | Method | After FT | Delta vs magnitude | Dead outputs | Projection min | FC score |
|---:|---|---:|---:|---:|---:|---:|
| `98%` | magnitude | `29.28%` | baseline | `131.0` | `2.2884` | `3.3049` |
| `98%` | plain reserve `0.60` | `33.31%` | `+4.03` pts | `1.0` | `1.3545` | `3.4609` |
| `98%` | balanced route `0.60` | `32.98%` | `+3.70` pts | `0.5` | `1.2347` | `3.7086` |
| `99%` | magnitude | `24.31%` | baseline | `219.0` | `1.8136` | `2.7081` |
| `99%` | plain reserve `0.60` | `20.07%` | `-4.24` pts | `3.0` | `0.9650` | `1.7307` |
| `99%` | balanced route `0.60` | `25.68%` | `+1.37` pts | `1.5` | `0.8844` | `2.5649` |

Interpretation:

- This is the first positive TinyResNet `99%` result.
- Plain reserve keeps outputs alive but starves classifier readout at the residual cliff.
- Backbone reserve overprotects projection shortcuts and starves other routes.
- Balanced route capacity restores classifier readout while preserving enough projection/main capacity to remain trainable.
- This supports the circuit-remodeling thesis more strongly than one-layer liveness: the useful object is balanced route allocation across interacting paths.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_BALANCED_ROUTE_CAPACITY.md`.

## Balanced residual route 99% replicate

The two-seed positive TinyResNet `99%` result was tested on four fresh seeds.

Result:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs | Projection min | FC score |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `24.93%` | baseline | baseline | `220.8` | `1.7889` | `2.7874` |
| global SynFlow | `10.13%` | `-14.80` pts | `0/4` | `298.0` | `0.0000` | `3.6335` |
| plain reserve `0.60` | `19.39%` | `-5.54` pts | `1/4` | `3.5` | `1.0115` | `1.6283` |
| balanced route `0.60` | `23.40%` | `-1.53` pts | `0/4` | `3.0` | `0.8988` | `2.5720` |

Interpretation:

- The two-seed balanced-route win did not replicate against magnitude.
- Balanced route still materially improves over plain reserve at the residual `99%` cliff: `+4.01` points after FT.
- The route-quality diagnosis remains useful because it identified classifier-readout starvation as part of the plain-reserve failure.
- But the current balanced allocator is not yet a residual solution. It narrows the gap to magnitude rather than surpassing it robustly.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_BALANCED_ROUTE_99PCT_REPLICATE.md`.

## Projection/readout split sweep

The replicate showed the balanced route idea was directionally useful but underprotected projection capacity. A targeted split sweep increased projection share while preserving classifier readout.

Four fresh TinyResNet `99%` seeds:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `24.89%` | baseline | baseline | `1.7857` | `2.7890` | `221.0` |
| plain reserve `0.60` | `20.53%` | `-4.36` pts | `0/4` | `1.0143` | `1.6156` | `3.8` |
| balanced `50/25/25` | `23.30%` | `-1.58` pts | `1/4` | `0.9005` | `2.5700` | `2.8` |
| projection-heavy `45/35/20` | `24.22%` | `-0.67` pts | `1/4` | `1.0437` | `2.3946` | `3.2` |
| projection/readout `40/35/25` | `25.63%` | `+0.74` pts | `3/4` | `1.0437` | `2.5376` | `4.0` |

Interpretation:

- This is the strongest current residual result.
- The successful split protects projection routes more than the first balanced allocator while preserving enough readout capacity.
- The result supports the route-quality diagnosis: residual pruning needs balanced capacity across route families, not only dead-output prevention.
- It is still a tuned split, not a predicted theory. The next step is to predict the split from route-quality deficits.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_ROUTE_SPLIT_SWEEP_99PCT.md`.

## Predicted route-deficit split

The next test replaced hand-selected route shares with a pre-finetuning route-deficit predictor.

Prediction rule:

1. Build the magnitude mask as a viability template.
2. Build the plain reserve mask as the failing candidate.
3. Measure projection-route deficit and classifier-readout deficit before fine-tuning.
4. Keep a main-path floor.
5. Allocate the remaining reserve between projection and readout deficits, with projection weighted because the route audit identified it as the strongest residual-99 correlate.

Four TinyResNet `99%` seeds:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Projection min | FC score | Dead outputs |
|---|---:|---:|---:|---:|---:|---:|
| magnitude | `25.04%` | baseline | baseline | `1.7857` | `2.7889` | `220.8` |
| plain reserve `0.60` | `20.21%` | `-4.83` pts | `0/4` | `1.0114` | `1.6452` | `3.5` |
| tuned `40/35/25` | `24.88%` | `-0.17` pts | `3/4` | `1.0492` | `2.5357` | `4.0` |
| predicted deficit split | `25.12%` | `+0.08` pts | `2/4` | `1.0409` | `2.5533` | `4.0` |

Interpretation:

- The effect size is tiny and should not be overstated.
- But this is the first residual route split selected from measured route deficits rather than from a post-hoc sweep.
- The predictor recovers nearly the same projection/readout balance as the tuned winner and slightly beats it on mean in this run.
- The next step is to strengthen the predictor so it does not rely on a fixed main-path floor or hand-set projection reliability weight.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_PREDICTED_ROUTE_SPLIT_99PCT.md`.

## Derived route-predictor follow-up

A fresh four-seed run tested whether the predictor's remaining constants could be removed.

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

- Removing the constants did not produce a stronger predictor.
- Equal-family and width-derived priors are worse than the fixed predictor.
- The fixed predictor still closes most of the gap between plain reserve and magnitude, but it is not robustly better than magnitude.
- The next algorithm should solve for route-quality targets directly instead of guessing family priors.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_DERIVED_ROUTE_PREDICTORS_99PCT.md`.

## Target-matched route optimizer

The next variant searched route-family splits before fine-tuning and selected the split whose mask best matched route-quality targets.

Fresh TinyResNet `99%` seeds `[211, 212, 213, 214]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `25.79%` | baseline | baseline | `219.2` |
| plain reserve `0.60` | `18.74%` | `-7.05` pts | `0/4` | `2.0` |
| tuned `40/35/25` | `24.43%` | `-1.36` pts | `0/4` | `2.0` |
| fixed deficit predictor | `23.59%` | `-2.21` pts | `0/4` | `2.0` |
| target-matched optimizer | `24.52%` | `-1.27` pts | `1/4` | `8.0` |

Interpretation:

- The optimizer improved substantially over plain reserve, but still trailed magnitude.
- It selected `20/55/25` main/projection/readout for every seed.
- That overprotected projection routes and increased dead outputs.
- The next objective needs a route-diversity or overconcentration penalty, not just target matching.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_TARGET_MATCHED_ROUTE_OPTIMIZER_99PCT.md`.

## Diversity-penalized route optimizer

The target-matched optimizer failed by overconcentrating capacity into projection routes. A follow-up added a degeneracy-inspired penalty against route-family concentration and projection overuse.

Fresh TinyResNet `99%` seeds `[215, 216, 217, 218]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `24.80%` | baseline | baseline | `220.5` |
| plain reserve `0.60` | `20.13%` | `-4.67` pts | `0/4` | `2.8` |
| tuned `40/35/25` | `26.04%` | `+1.25` pts | `3/4` | `3.0` |
| diversity target optimizer | `25.85%` | `+1.05` pts | `4/4` | `7.2` |

Interpretation:

- This is the strongest optimizer-style residual result so far.
- The degeneracy penalty fixed the previous overprojection failure enough to beat magnitude on every seed.
- The tuned split is slightly higher on mean, but the optimizer has better paired consistency.
- The result supports the neuroscience framing: preserving circuit viability requires both route targets and route-family diversity.

See `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_DIVERSITY_ROUTE_OPTIMIZER_99PCT.md`.

## DeepTinyResNet transfer

The diversity route optimizer was transferred to a deeper residual model with two residual blocks per stage.

DeepTinyResNet, CIFAR-10 subset, `99%` sparsity, seeds `[219, 220]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `26.40%` | baseline | baseline | `301.5` |
| global SynFlow | `9.96%` | `-16.44` pts | `0/2` | `477.5` |
| plain reserve `0.60` | `29.54%` | `+3.14` pts | `2/2` | `0.5` |
| tuned `40/35/25` | `29.88%` | `+3.48` pts | `2/2` | `0.5` |
| diversity target optimizer | `29.14%` | `+2.74` pts | `2/2` | `0.5` |

Interpretation:

- The route-capacity idea transfers to a deeper residual model.
- The baseline pathology persists: global SynFlow collapses near chance and creates massive dead outputs.
- Unlike the original TinyResNet `99%` setting, even plain reserve beats magnitude here because magnitude itself has hundreds of dead outputs.
- Tuned route split is best on mean, but the diversity optimizer also beats magnitude on both seeds.

Caveat: this is only two seeds. It is a transfer signal, not a final benchmark.

See `experiments/04_criticality_pruning/CIFAR10_DEEP_TINY_RESNET_DIVERSITY_ROUTE_OPTIMIZER_99PCT.md`.

## DeepTinyResNet four-seed replicate

The deeper residual transfer was repeated on four fresh seeds.

DeepTinyResNet, CIFAR-10 subset, `99%` sparsity, seeds `[221, 222, 223, 224]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `28.48%` | baseline | baseline | `298.2` |
| global SynFlow | `9.98%` | `-18.50` pts | `0/4` | `479.5` |
| plain reserve `0.60` | `30.20%` | `+1.72` pts | `3/4` | `0.2` |
| tuned `40/35/25` | `29.88%` | `+1.41` pts | `3/4` | `0.2` |
| diversity target optimizer | `28.83%` | `+0.35` pts | `1/4` | `0.2` |

Interpretation:

- Deep residual capacity transfer replicated.
- In this deeper model, homeostatic output preservation is the dominant win: plain reserve beats magnitude and tuned route split.
- The diversity optimizer remains positive on mean but does not beat the simpler capacity rule in this setting.
- This narrows the mechanism: deeper residuals may first need broad liveness/homeostasis before fine route-family optimization matters.

See `experiments/04_criticality_pruning/CIFAR10_DEEP_TINY_RESNET_DIVERSITY_ROUTE_OPTIMIZER_99PCT_REPLICATE.md`.

## CIFAR ResNet-20-style transfer

A CIFAR ResNet-20-style model was tested as the first standard residual benchmark.

ResNet-20-style, CIFAR-10 subset, `99%` sparsity, seeds `[225, 226]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `28.48%` | baseline | baseline | `339.0` |
| global SynFlow | `9.99%` | `-18.49` pts | `0/2` | `666.5` |
| plain reserve `0.60` | `32.41%` | `+3.93` pts | `2/2` | `0.0` |
| tuned `40/35/25` | `29.92%` | `+1.44` pts | `2/2` | `0.0` |

Interpretation:

- The capacity result transfers to a ResNet-20-style architecture.
- The mechanism is again homeostatic: magnitude leaves hundreds of dead outputs; reserve eliminates dead outputs and beats magnitude.
- The route split also beats magnitude, but plain reserve is stronger in this standard residual setting.

See `experiments/04_criticality_pruning/CIFAR10_RESNET20_CAPACITY_99PCT.md`.

## CIFAR ResNet-20-style four-seed replicate

The ResNet-20-style result was repeated on four fresh seeds.

ResNet-20-style, CIFAR-10 subset, `99%` sparsity, seeds `[227, 228, 229, 230]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `27.25%` | baseline | baseline | `341.8` |
| global SynFlow | `10.03%` | `-17.22` pts | `0/4` | `670.5` |
| plain reserve `0.60` | `33.22%` | `+5.97` pts | `4/4` | `0.2` |
| tuned `40/35/25` | `30.60%` | `+3.36` pts | `4/4` | `0.2` |

Interpretation:

- This is the strongest standard-style residual evidence so far.
- The homeostatic capacity rule beats magnitude on every seed by a large margin.
- The mechanism is clean: magnitude leaves hundreds of dead outputs; reserve eliminates them.
- Route splitting also helps, but the simple broad reserve rule is best in this setting.

See `experiments/04_criticality_pruning/CIFAR10_RESNET20_CAPACITY_99PCT_REPLICATE.md`.

## Full CIFAR-10 ResNet-20-style transfer

The ResNet-20-style experiment was run on full CIFAR-10 train/test instead of the 20k/5k subset.

Full CIFAR-10, ResNet-20-style, `99%` sparsity, seeds `[231, 232]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude | Dead outputs |
|---|---:|---:|---:|---:|
| magnitude | `38.01%` | baseline | baseline | `181.0` |
| global SynFlow | `10.00%` | `-28.01` pts | `0/2` | `666.0` |
| plain reserve `0.60` | `39.61%` | `+1.60` pts | `2/2` | `0.0` |
| tuned `40/35/25` | `38.38%` | `+0.37` pts | `1/2` | `0.0` |

Interpretation:

- The full-CIFAR result remains positive but the margin is smaller than the subset result.
- Capacity reserve still eliminates dead outputs and beats magnitude on both seeds.
- Stronger dense training reduces the advantage of liveness alone, but does not remove it.
- Global SynFlow remains catastrophically broken in this setting.

See `experiments/04_criticality_pruning/CIFAR10_FULL_RESNET20_CAPACITY_99PCT.md`.
