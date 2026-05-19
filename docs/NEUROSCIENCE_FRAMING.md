# Neuroscience Framing: Circuit Viability Under Pruning

This repo's central idea is not simply that biological brains prune synapses. The stronger idea is that biological pruning is constrained by circuit viability.

## Core translation

Biological development removes many synapses, but it does not optimize isolated synapse deletion while ignoring whether the organism can still perceive, act, and learn. Useful circuits must remain viable while redundant or weak connections are removed.

The machine-learning analogue is severe neural-network pruning:

- a weight is a synapse;
- a channel, hidden unit, attention head, or residual block is a circuit element;
- a layer-to-layer path family is a communication route;
- a global saliency threshold is unconstrained synapse deletion;
- a zero-capacity bridge is a circuit lesion;
- path capacity is the minimum viable communication substrate left after pruning.

## Biological principles and algorithmic counterparts

| Neuroscience principle | Biological meaning | Pruning counterpart | Current repo evidence |
|---|---|---|---|
| Synaptic pruning | Remove weak or redundant synapses. | Use saliency or magnitude to remove individual weights. | Magnitude and SynFlow baselines. |
| Circuit viability | Preserve functional routes through a circuit. | Maintain nonzero capacity across required communication cuts. | SynFlow bridge collapse and capacity rescue experiments. |
| Homeostatic plasticity | Prevent circuits from becoming silent or unstable. | Prevent dead outputs, dead bridges, and zero-capacity cutsets. | Dense bridge and TinyResNet dead-output diagnostics. |
| Use-dependent stabilization | Active pathways are more likely to survive. | Use activation-supported path ranking. | TinyResNet activation reserve is a negative first test. |
| Degeneracy | Multiple different routes can support similar function. | Preserve route diversity rather than a single surviving wire. | Diversity route optimization and strong TinyViT branch selection. |
| Communication backbones | Long-range/bottleneck routes support system-level function. | Estimate min-cut or path capacity between representation stages. | Current path-capacity constraints approximate this in CNNs. |

## What the current experiments show

The SynFlow pathology is a machine analogue of a circuit lesion.

Global SynFlow preserves some high-scoring synapses but deletes the dense bridge that carries representation into the classifier. Fine-tuning cannot recover because the mask has removed the route. The problem is not only that there are too few weights. The problem is that the surviving weights no longer form a viable circuit.

Path-capacity pruning adds a viability constraint. It still uses saliency, but saliency is no longer allowed to delete every route through a required communication cut.

## What the current experiments do not show

The current method is not yet a full biological theory of pruning.

Limitations:

- The best evidence is still mostly CNN severe-sparsity work.
- Residual networks reveal that output liveness alone is not route quality.
- Naive presynaptic activation support did not solve the residual case.
- The repo does not yet model richer biological mechanisms such as local competition, neuromodulation, dendritic compartment constraints, or multi-timescale plasticity.
- The transformer analogue is currently TinyViT-scale; it has branch evidence, not broad transformer or LLM transfer.

## Important lesson from the negative activation result

A simple use-dependent rule is not enough:

> saliency multiplied by presynaptic activation did not fix TinyResNet 99% and underperformed plain reserve capacity.

That is useful. It means the right neuroscience translation is not merely “active inputs survive.” The stronger translation is probably:

- active routes must connect through complete downstream paths;
- residual additions create alternate communication geometry;
- route diversity matters more than local activation;
- useful capacity should be measured after block composition, not only at individual weight tensors.

## Important lesson from residual route-quality audits

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

## Important lesson from TinyViT residual-stream branch selection

TinyViT makes the neuroscience connection sharper because it separates anatomical liveness from functional viability.

In the strong TinyViT runs, masks that eliminate measured dead rows can still remain at the magnitude floor. Seed 310 is the cleanest branch example: all-route liveness leaves zero measured MLP/attention dead rows but recovers only `5.86%`, while SynFlow keeps many dead rows by that metric and recovers `15.16%`. Seed 311 repeats the pattern without branch scanning: all-route liveness reaches `9.97%`, while SynFlow reaches `15.91%`. The V5 selector chooses SynFlow before fine-tuning because the masked-before behavior indicates a more trainable residual-stream basin.

The biological analogy is that keeping every local pathway anatomically present is not the same as preserving the functional ensemble. A sparse transformer circuit needs:

- residual-stream feature preservation;
- globally coherent signal paths;
- enough trainable capacity for masked recovery;
- liveness repair only when silence is the actual limiting deficit.

That is why the current theory is not simply "keep units alive." It is constrained circuit remodeling: preserve the route family that keeps the computation recoverable under the task ecology.

## Next neuroscience-grounded method hypothesis

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

## Working thesis

Pruning should not be framed as choosing the best isolated synapses. It should be framed as constrained circuit remodeling:

> maximize synaptic efficiency subject to preserving viable, diverse, task-relevant communication routes.

That is the neuro-AI idea this repo should make real.
