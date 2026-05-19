# Path-Capacity Synthesis

Circuit-capacity constraints can convert global SynFlow's severe-sparsity cutset collapse into trainable masks under the same parameter budget. The strongest current CIFAR-10 CNN result is a 99% reserve sweep where reserve_0.60 beats magnitude by +2.56 points after fine-tuning and wins 4/4 paired seeds; a broad 0.45-0.65 reserve band beats magnitude on mean. Fashion-MNIST CNN transfer also beats magnitude on mean at 98% and 99%. Residual transfer is now positive in custom and standard-style settings: route-quality targets plus a degeneracy penalty beat magnitude on TinyResNet, broad reserve capacity beats magnitude on DeepTinyResNet, reserve beats magnitude by +5.97 points on a ResNet-20-style subset replicate, and remains positive on full CIFAR-10. The latest full-CIFAR tradeoff selector replaces the old route-floor-only family rule with a feature-preservation/liveness score and selects the best method on two fresh seeds.

## Cases

| Label | Sparsity | Method | Magnitude after FT | Global SynFlow after FT | Method after FT | Delta vs magnitude | Rescue vs SynFlow | Wins | Dead units | Dead metric |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| single bridge capacity | `0.98` | `pathcap_synflow_bridge_mag` | `0.4477` | `0.0976` | `0.3341` | `-0.1136` | `+0.2365` | `` | `0.0` | `dead_fc1_hidden_mean` |
| multi-cut fixed floors | `0.98` | `multicut_capacity` | `0.4405` | `0.0976` | `0.4009` | `-0.0396` | `+0.3033` | `` | `0.0` | `dead_fc1_hidden_mean` |
| multi-cut fixed floors | `0.99` | `multicut_capacity` | `0.3261` | `0.0976` | `0.3341` | `+0.0080` | `+0.2365` | `` | `0.0` | `dead_fc1_hidden_mean` |
| adaptive output-count capacity | `0.98` | `adaptive_capacity` | `0.4411` | `0.0976` | `0.4170` | `-0.0241` | `+0.3194` | `` | `0.0` | `dead_fc1_hidden_mean` |
| adaptive output-count capacity | `0.99` | `adaptive_capacity` | `0.3282` | `0.0976` | `0.3464` | `+0.0182` | `+0.2488` | `` | `0.0` | `dead_fc1_hidden_mean` |
| fixed capacity replicate | `0.99` | `fixed_capacity` | `0.3261` | `0.0976` | `0.3317` | `+0.0056` | `+0.2341` | `` | `0.0` | `dead_fc1_hidden_mean` |
| dead-risk adaptive capacity | `0.99` | `risk_adaptive_capacity` | `0.3261` | `0.0976` | `0.3222` | `-0.0039` | `+0.2246` | `` | `0.0` | `dead_fc1_hidden_mean` |
| fixed capacity replicate 2 | `0.99` | `fixed_capacity` | `0.3177` | `0.0976` | `0.3409` | `+0.0232` | `+0.2433` | `` | `0.0` | `dead_fc1_hidden_mean` |
| mass-risk adaptive capacity | `0.99` | `mass_risk_capacity` | `0.3177` | `0.0976` | `0.3291` | `+0.0114` | `+0.2315` | `` | `0.0` | `dead_fc1_hidden_mean` |
| cifar reserve sweep best | `0.99` | `reserve_0.60` | `0.3231` | `0.0976` | `0.3488` | `+0.0256` | `+0.2511` | `4/4` | `0.0` | `dead_fc1_hidden_mean` |
| cifar reserve sweep lower band | `0.99` | `reserve_0.50` | `0.3231` | `0.0976` | `0.3434` | `+0.0203` | `+0.2458` | `4/4` | `0.0` | `dead_fc1_hidden_mean` |
| cifar reserve sweep upper band | `0.99` | `reserve_0.65` | `0.3231` | `0.0976` | `0.3394` | `+0.0162` | `+0.2418` | `3/4` | `0.0` | `dead_fc1_hidden_mean` |
| fashion transfer 98 | `0.98` | `reserve_0.60` | `0.8479` | `0.0992` | `0.8514` | `+0.0036` | `+0.7522` | `` | `0.0` | `dead_fc1_hidden_mean` |
| fashion transfer 99 | `0.99` | `reserve_0.60` | `0.8024` | `0.0989` | `0.8175` | `+0.0151` | `+0.7186` | `` | `0.0` | `dead_fc1_hidden_mean` |
| tinyresnet transfer 98 | `0.98` | `reserve_0.60` | `0.3132` | `0.0979` | `0.3227` | `+0.0095` | `+0.2248` | `1/2` | `1.0` | `dead_outputs_mean` |
| tinyresnet transfer 99 | `0.99` | `reserve_0.60` | `0.2423` | `0.1010` | `0.2132` | `-0.0291` | `+0.1122` | `0/2` | `3.0` | `dead_outputs_mean` |
| tinyresnet activation reserve 98 | `0.98` | `activation_reserve_0.60` | `0.2972` | `0.1007` | `0.2959` | `-0.0013` | `+0.1952` | `1/2` | `1.0` | `dead_outputs_mean` |
| tinyresnet activation reserve 99 | `0.99` | `activation_reserve_0.60` | `0.2450` | `0.1026` | `0.2005` | `-0.0445` | `+0.0979` | `0/2` | `2.5` | `dead_outputs_mean` |
| tinyresnet backbone reserve 98 | `0.98` | `backbone_reserve_0.60` | `0.2943` | `0.0979` | `0.3219` | `+0.0276` | `+0.2240` | `2/2` | `1.0` | `dead_outputs_mean` |
| tinyresnet backbone reserve 99 | `0.99` | `backbone_reserve_0.60` | `0.2445` | `0.1010` | `0.1775` | `-0.0670` | `+0.0765` | `0/2` | `3.5` | `dead_outputs_mean` |
| tinyresnet balanced route 98 | `0.98` | `balanced_route_0.60` | `0.2928` | `0.0979` | `0.3298` | `+0.0370` | `+0.2319` | `2/2` | `0.5` | `dead_outputs_mean` |
| tinyresnet balanced route 99 | `0.99` | `balanced_route_0.60` | `0.2431` | `0.1010` | `0.2568` | `+0.0137` | `+0.1558` | `1/2` | `1.5` | `dead_outputs_mean` |
| tinyresnet balanced route 99 replicate | `0.99` | `balanced_route_0.60` | `0.2492` | `0.1013` | `0.2339` | `-0.0153` | `+0.1327` | `0/4` | `3.0` | `dead_outputs_mean` |
| tinyresnet projection-readout split 99 | `0.99` | `proj_readout_40_35_25` | `0.2489` | `n/a` | `0.2563` | `+0.0074` | `n/a` | `3/4` | `4.0` | `dead_outputs_mean` |
| tinyresnet predicted route split 99 | `0.99` | `predicted_deficit_split` | `0.2504` | `n/a` | `0.2512` | `+0.0008` | `n/a` | `2/4` | `4.0` | `dead_outputs_mean` |
| tinyresnet fixed predictor fresh 99 | `0.99` | `fixed_deficit_predictor` | `0.2498` | `n/a` | `0.2459` | `-0.0039` | `n/a` | `2/4` | `3.0` | `dead_outputs_mean` |
| tinyresnet target-matched optimizer 99 | `0.99` | `target_matched_optimizer` | `0.2579` | `n/a` | `0.2452` | `-0.0127` | `n/a` | `1/4` | `8.0` | `dead_outputs_mean` |
| tinyresnet diversity optimizer 99 | `0.99` | `diversity_target_optimizer` | `0.2480` | `n/a` | `0.2585` | `+0.0105` | `n/a` | `4/4` | `7.2` | `dead_outputs_mean` |
| deep tinyresnet diversity optimizer 99 | `0.99` | `diversity_target_optimizer` | `0.2640` | `0.0996` | `0.2914` | `+0.0274` | `+0.1918` | `2/2` | `0.5` | `dead_outputs_mean` |
| deep tinyresnet reserve replicate 99 | `0.99` | `reserve_0.60` | `0.2848` | `0.0998` | `0.3020` | `+0.0172` | `+0.2022` | `3/4` | `0.2` | `dead_outputs_mean` |
| resnet20-style reserve 99 | `0.99` | `reserve_0.60` | `0.2848` | `0.0999` | `0.3241` | `+0.0393` | `+0.2242` | `2/2` | `0.0` | `dead_outputs_mean` |
| resnet20-style reserve replicate 99 | `0.99` | `reserve_0.60` | `0.2725` | `0.1003` | `0.3322` | `+0.0597` | `+0.2319` | `4/4` | `0.2` | `dead_outputs_mean` |
| full cifar resnet20-style reserve 99 | `0.99` | `reserve_0.60` | `0.3801` | `0.1000` | `0.3961` | `+0.0160` | `+0.2961` | `2/2` | `0.0` | `dead_outputs_mean` |
| full cifar resnet20-style reserve sixseed 99 | `0.99` | `reserve_0.60` | `0.3772` | `0.1000` | `0.3921` | `+0.0150` | `+0.2921` | `6/6` | `0.0` | `dead_outputs_mean` |
| full cifar resnet20-style SGD reserve fourseed 99 | `0.99` | `reserve_0.60` | `0.4287` | `0.1000` | `0.4943` | `+0.0657` | `+0.3943` | `4/4` | `1.8` | `dead_outputs_mean` |
| full cifar resnet20-style predicted route split 99 | `0.99` | `predicted_route_split` | `0.4209` | `n/a` | `0.4692` | `+0.0482` | `n/a` | `2/2` | `0.0` | `dead_outputs_mean` |
| full cifar100 resnet20-style SGD reserve 99 | `0.99` | `reserve_0.60` | `0.0658` | `0.0100` | `0.0764` | `+0.0106` | `+0.0664` | `2/2` | `0.5` | `dead_outputs_mean` |
| full cifar100 readout-main route split 99 | `0.99` | `readout_main_45_15_40` | `0.0653` | `n/a` | `0.0912` | `+0.0259` | `n/a` | `2/2` | `0.0` | `dead_outputs_mean` |
| full cifar100 predicted route split 99 | `0.99` | `predicted_route_split` | `0.0709` | `n/a` | `0.0980` | `+0.0272` | `n/a` | `2/2` | `1.5` | `dead_outputs_mean` |
| full cifar100 conservative predicted route split 99 | `0.99` | `predicted_route_split` | `0.0698` | `n/a` | `0.0878` | `+0.0181` | `n/a` | `2/2` | `0.5` | `dead_outputs_mean` |
| full cifar100 tradeoff selector v1 99 | `0.99` | `tradeoff_policy` | `0.0698` | `n/a` | `0.0821` | `+0.0123` | `n/a` | `2/2` | `61.5` | `dead_outputs_mean` |
| full cifar100 tradeoff selector v2 policy 99 | `0.99` | `tradeoff_v2_policy` | `0.0698` | `n/a` | `0.0887` | `+0.0189` | `n/a` | `` | `1.0` | `dead_outputs_mean` |
| full cifar100 tradeoff selector v2 prospective 99 | `0.99` | `tradeoff_policy` | `0.0654` | `n/a` | `0.0926` | `+0.0272` | `n/a` | `2/2` | `0.0` | `dead_outputs_mean` |
| ecology selector cifar10 selected reserve 99 | `0.99` | `ecology_selected` | `0.4393` | `n/a` | `0.4674` | `+0.0281` | `n/a` | `2/2` | `0.5` | `dead_outputs_mean` |
| ecology selector cifar100 selected split 99 | `0.99` | `ecology_selected` | `0.0541` | `n/a` | `0.0908` | `+0.0367` | `n/a` | `2/2` | `1.0` | `dead_outputs_mean` |
| deep tinyresnet ecology selector policy 99 | `0.99` | `ecology_policy` | `0.2721` | `n/a` | `0.3150` | `+0.0429` | `n/a` | `2/2` | `1.0` | `dead_outputs_mean` |
| full cifar resnet20 ecology selector SGD40 99 | `0.99` | `ecology_policy` | `0.4873` | `n/a` | `0.5262` | `+0.0390` | `n/a` | `2/2` | `3.0` | `dead_outputs_mean` |
| tinyvit minimal liveness repair 98 | `0.98` | `minimal_liveness_repair` | `0.1007` | `0.1100` | `0.1131` | `+0.0124` | `+0.0031` | `1/2` | `62.5` | `dead_outputs_mean` |
| tinyvit mlp-readout reserve 98 | `0.98` | `mlp_readout_reserve` | `0.1007` | `0.1100` | `0.0995` | `-0.0012` | `-0.0105` | `1/2` | `2181.0` | `dead_outputs_mean` |
| tinyvit minimal liveness repair 95 | `0.95` | `minimal_liveness_repair` | `0.0987` | `0.1078` | `0.1087` | `+0.0100` | `+0.0009` | `2/2` | `72.5` | `dead_outputs_mean` |
| tinyvit selective mlp-readout repair 95 | `0.95` | `selective_mlp_readout_repair` | `0.0987` | `0.1078` | `0.1023` | `+0.0036` | `-0.0055` | `1/2` | `760.5` | `dead_outputs_mean` |
| tinyvit feature-subspace diagnostic synflow 95 | `0.95` | `global_synflow` | `0.1102` | `0.1686` | `0.1686` | `+0.0584` | `+0.0000` | `2/2` | `2269.5` | `dead_outputs_mean` |
| tinyvit feature-subspace diagnostic liveness 95 | `0.95` | `minimal_liveness_repair` | `0.1102` | `0.1686` | `0.1031` | `-0.0071` | `-0.0655` | `1/2` | `78.5` | `dead_outputs_mean` |
| tinyvit feature-subspace selector 95 | `0.95` | `feature_subspace_policy` | `0.0882` | `0.1357` | `0.1224` | `+0.0342` | `-0.0133` | `1/2` | `1711.5` | `dead_outputs_mean` |
| tinyvit feature-route margin policy 95 | `0.95` | `feature_route_margin_policy` | `0.0882` | `0.1357` | `0.1357` | `+0.0475` | `+0.0000` | `` | `2224.5` | `dead_outputs_mean` |
| tinyvit feature-route margin selector prospective 95 | `0.95` | `feature_route_margin_policy` | `0.0895` | `0.1460` | `0.1460` | `+0.0565` | `+0.0000` | `2/2` | `2243.5` | `dead_outputs_mean` |
| tinyvit feature-route margin selector prospective 90 | `0.90` | `feature_route_margin_policy` | `0.1032` | `0.1456` | `0.1456` | `+0.0424` | `+0.0000` | `2/2` | `1775.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector strong pilot 90 | `0.90` | `feature_route_margin_policy` | `0.1670` | `0.1435` | `0.1435` | `-0.0235` | `+0.0000` | `` | `1662.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector v2 strong 90 | `0.90` | `feature_route_margin_policy` | `0.1218` | `0.1025` | `0.1218` | `+0.0000` | `+0.0193` | `` | `83.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector v2 strong replicate 90 | `0.90` | `feature_route_margin_policy` | `0.1382` | `0.1424` | `0.1382` | `+0.0000` | `-0.0042` | `` | `78.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector v3 strong 90 | `0.90` | `feature_route_margin_policy` | `0.1139` | `0.1452` | `0.1452` | `+0.0313` | `+0.0000` | `` | `1709.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector v3 strong replicate 90 | `0.90` | `feature_route_margin_policy` | `0.0806` | `0.1505` | `0.1505` | `+0.0698` | `+0.0000` | `2/2` | `1685.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector v4 strong 90 | `0.90` | `feature_route_margin_policy` | `0.1022` | `0.1709` | `0.1709` | `+0.0687` | `+0.0000` | `1/1` | `1707.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector v4 seed306 90 | `0.90` | `feature_route_margin_policy` | `0.1099` | `0.1329` | `0.1158` | `+0.0059` | `-0.0171` | `1/1` | `3.0` | `dead_outputs_mean` |
| tinyvit feature-route margin selector v5 strong 90 | `0.90` | `feature_route_margin_policy` | `0.0725` | `0.1018` | `0.1018` | `+0.0293` | `+0.0000` | `1/1` | `1709.0` | `dead_outputs_mean` |
| tinyimagenet ecology selector external proxy 99 | `0.99` | `ecology_policy` | `0.0232` | `n/a` | `0.0290` | `+0.0058` | `n/a` | `` | `0.0` | `dead_outputs_mean` |
| tinyimagenet pretrained resnet18 ecology selector 99 | `0.99` | `ecology_policy` | `0.0107` | `n/a` | `0.0080` | `-0.0027` | `n/a` | `` | `0.0` | `dead_outputs_mean` |
| tinyimagenet pretrained resnet18 ecology selector 95 | `0.95` | `ecology_policy` | `0.1523` | `n/a` | `0.0397` | `-0.1127` | `n/a` | `` | `0.0` | `dead_outputs_mean` |
| tinyimagenet pretrained feature-viability repair 95 | `0.95` | `feature_viability_repair` | `0.1513` | `n/a` | `0.1503` | `-0.0010` | `n/a` | `` | `0.0` | `dead_outputs_mean` |
| tinyimagenet pretrained feature-viability repair twoseed 95 | `0.95` | `feature_viability_repair` | `0.1487` | `n/a` | `0.1482` | `-0.0005` | `n/a` | `` | `0.0` | `dead_outputs_mean` |
| tinyimagenet pretrained feature-viability repair 99 | `0.99` | `feature_viability_repair` | `0.0120` | `n/a` | `0.0143` | `+0.0023` | `n/a` | `` | `4.0` | `dead_outputs_mean` |
| tinyimagenet pretrained tradeoff selector 95 | `0.95` | `tradeoff_policy` | `0.1567` | `n/a` | `0.1570` | `+0.0003` | `n/a` | `` | `0.0` | `dead_outputs_mean` |
| full cifar resnet20 feature-vs-homeostasis 99 | `0.99` | `feature_viability_repair` | `0.4591` | `n/a` | `0.4926` | `+0.0335` | `n/a` | `2/2` | `35.5` | `dead_outputs_mean` |
| full cifar resnet20 tradeoff selector 99 | `0.99` | `tradeoff_policy` | `0.4460` | `n/a` | `0.4997` | `+0.0538` | `n/a` | `2/2` | `32.0` | `dead_outputs_mean` |
| unified viability selector retrospective | `n/a` | `unified_selector` | `0.2356` | `n/a` | `0.2603` | `+0.0248` | `n/a` | `5/6` | `1.6` | `selected_dead_outputs` |

## Best current result

Best delta vs magnitude: `tinyvit feature-route margin selector v3 strong replicate 90` at `0.90` with `+0.0698` after-FT accuracy.
Best rescue vs global SynFlow: `fashion transfer 98` at `0.98` with `+0.7522` after-FT accuracy.
Positive cases vs magnitude in this synthesis: `55/75`.

## Readout

Path-capacity constraints reliably prevent dense-bridge death and rescue SynFlow collapse. The strongest CIFAR evidence is the `99%` reserve sweep: a broad reserve band from `0.45` through `0.65` beats magnitude on mean, and `reserve_0.60` beats magnitude by `+2.56` points after fine-tuning with `4/4` paired wins.

The constructive result also transfers to Fashion-MNIST CNN: `reserve_0.60` beats magnitude on mean at both `98%` and `99%`, with the larger gain at the harsher `99%` cliff.

TinyResNet is the first residual transfer test and gives a mixed result: capacity reserve beats magnitude at `98%` but trails at `99%` even while nearly eliminating dead outputs. That is a useful boundary condition: the next method must estimate residual-route quality, not merely output liveness.

The activation-supported TinyResNet follow-up is a negative result. Multiplying protected-capacity rank by presynaptic activation did not outperform plain reserve capacity and worsened the `99%` gap. Simple use-dependent stabilization is therefore too local; the residual case likely needs path interaction, block-level cut geometry, or post-residual route diversity.

The residual-backbone TinyResNet follow-up improves the `98%` residual case but worsens the `99%` cliff. Projection shortcuts matter, but simply overprotecting them is not enough; at extreme sparsity it appears to steal capacity from other required routes.

The balanced residual route allocator is a useful but incomplete residual advance. It uses the route-quality audit to split protected capacity across main transformations, projection shortcuts, and classifier readout. In the first two-seed TinyResNet run it beat magnitude at `99%`, but a four-seed fresh replicate did not confirm that win: balanced route capacity improved over plain reserve by `+4.01` points but still trailed magnitude by `-1.53` points. The residual gap is narrowed, not solved.

A targeted projection/readout split sweep then closed the residual gap in the same four-seed `99%` setting. The `40/35/25` main/projection/readout split beats magnitude by `+0.745` points and wins `3/4` seeds, while plain reserve trails magnitude by `-4.36` points. This is the strongest current evidence that residual path-capacity needs balanced route-family allocation, not output liveness alone.

The first route-deficit predictor is small but important. It compares a magnitude viability template to the plain reserve candidate before fine-tuning, then allocates projection/readout capacity from the measured deficits. Follow-ups showed that target matching alone overconcentrates projection capacity and trails magnitude. Adding a degeneracy-style diversity penalty fixes that failure in the latest batch: the diversity optimizer beats magnitude by `+1.05` points and wins `4/4` seeds while plain reserve trails by `-4.67` points. The residual result is now a concrete route-target plus degeneracy mechanism, though penalty weights are still hand-set.

The deeper residual transfer is also positive and now has a four-seed replicate. On DeepTinyResNet at `99%`, magnitude leaves hundreds of dead outputs. The replicate shows plain reserve at `30.20%` versus magnitude at `28.48%`, with `3/4` wins. This suggests the mechanism is not confined to the original TinyResNet, but the hierarchy is clearer: broad homeostatic capacity dominates in deeper residuals, while route-family diversity matters most once liveness is no longer the limiting factor.

The feature-vs-homeostasis tests correct the earlier selector. Route-floor-only family selection is too crude: feature-preserving liveness repair can beat broad reserve even in a from-scratch full-CIFAR ResNet-20 setting where magnitude has a dead main-path floor. A new tradeoff selector scores realized candidate masks by feature overlap, liveness, readout preservation, and dead-output penalty. On two fresh full-CIFAR seeds, it selected feature repair before fine-tuning and beat magnitude by `+5.37` points with `2/2` wins, while also beating plain reserve on mean. On a fresh pretrained TinyImageNet ResNet-18 seed at `95%`, the same selector avoided the homeostatic masks that collapsed performance, selected feature repair, removed all dead outputs, and slightly beat magnitude. CIFAR-100 exposed the next correction: V1 still overweighted feature preservation, while a V2 task-ecology pressure term selected route split on both projection and fresh prospective seeds. In the fresh prospective run, V2 selected the best mean candidate and beat magnitude by `+2.72` points with `2/2` wins. TinyViT now defines the transformer analogue but also the limitation: row-liveness repairs remove MLP/attention route death but do not dominate. In weak dense TinyViT runs, feature-route margin selectors select SynFlow and beat magnitude at both `95%` and `90%`. Stronger full-train TinyViT pilots show a different regime: V1 overselects SynFlow, V2 corrects one seed by selecting magnitude, a replicate favors all-route liveness, and V3 identifies the complementary feature-dominant regime. Across three fresh strong V3 seeds, the feature-margin branch selects SynFlow and beats magnitude; the two-seed replicate improves mean recovery by `+6.98` points with `2/2` wins. V4 adds masked pre-finetune accuracy as a trainability diagnostic and on its first fresh seed again selects the feature-dominant SynFlow branch, beating magnitude by `+6.87` points. A fresh non-SynFlow V4 seed falsifies the perfect projection: the selected attention+MLP repair beats magnitude by `+0.59` points, but SynFlow beats it by `+1.71` points. V5 adds a SynFlow masked-recovery prior; the first fresh V5 seed selects SynFlow and beats magnitude by `+2.93` points while the zero-dead liveness masks stay near magnitude. The strong-transformer selector is improving but remains unsolved: it needs prospective validation across both feature-dominant and ambiguous branches.

This is still not a final theory. The next scientific step is to predict the useful reserve band and feature/liveness tradeoff from route-quality features instead of sweeping or hand-weighting it.
