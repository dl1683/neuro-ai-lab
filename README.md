# Neuro-AI Lab

This repo tests a neuroscience-inspired claim about sparse intelligence:

**Useful sparse networks are not just collections of important synapses. They are viable circuits.**

The biological motivation is straightforward. Brains are massively sparse, metabolically constrained systems, but they do not work because each synapse is independently "important." They work because distributed assemblies keep communication routes alive, preserve functional representations, maintain enough redundancy for plasticity, and avoid degenerating into anatomically present but behaviorally dead pathways. A sparse artificial network should be judged the same way: not only by which weights survive, but by whether information can still flow through the functional circuit.

This project translates that idea into concrete pruning experiments. Instead of asking only "which individual weights score highest?", we measure whether a mask preserves classifier bridges, residual routes, readout diversity, transformer residual-stream features, and trainable sparse capacity. The core hypothesis is that severe pruning fails when it destroys circuit viability, and succeeds when it preserves the right functional ensemble even under extreme parameter pressure.

The current standout result is not a toy demo: **global SynFlow can catastrophically fail at severe CNN sparsity by allocating zero weights to the dense classifier bridge.** That is the artificial-network analogue of preserving local tissue while severing the communication tract needed for behavior. The failure replicated on Fashion-MNIST and CIFAR-10 CNNs, survived masked fine-tuning checks, and is packaged with diagnostics so future pruning runs can catch the pathology directly.

The constructive direction is Path-Capacity Pruning and circuit-viability selection: preserve communication capacity through vulnerable cuts, then fill the remaining parameter budget by saliency. Current experiments rescue SynFlow collapse, beat magnitude pruning in several severe-sparsity regimes, and expose architecture-specific viability rules. CNNs need bridge capacity; residual networks need route-quality and readout constraints; TinyViT shows a transformer analogue where residual-stream feature preservation can matter more than row-level liveness, but trainability still matters.

## Headline finding

At `98-99%` global sparsity, global SynFlow can assign `0` surviving weights to `fc1`, the first dense classifier bridge in CNNs. Once that bridge is gone, masked fine-tuning cannot recover the model.

Cross-dataset synthesis:

| Case | Magnitude after FT | Global SynFlow after FT | Delta | Global `fc1` keep | Dead bridge units |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST CNN, `98%` | `80.86%` | `10.28%` | `-70.59` pts | `0.0000` | `128/128` |
| CIFAR-10 CNN, `98%` | `44.08%` | `9.76%` | `-34.32` pts | `0.0000` | `192/192` |
| CIFAR-10 CNN, `99%` | `33.24%` | `9.76%` | `-23.48` pts | `0.0000` | `192/192` |

Aggregate:

- Global SynFlow zero-bridge cases: `3/3`.
- Mean after-fine-tuning delta vs magnitude: `-42.80` points.
- Layerwise SynFlow partially repairs allocation, but still averages `-22.21` points vs magnitude after fine-tuning.

Primary artifact:

- `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md`

## Why this matters

SynFlow is often treated as a label-free pruning baseline. These experiments show a concrete severe-sparsity failure mode: a global saliency allocation can preserve early convolutional weights while deleting the dense bridge required for classification. The scalar pruning score looks valid, but the mask is structurally unrecoverable.

The practical guardrail is simple: severe global pruning methods should report per-layer keep rates, dead bridge units, and classifier reachability before their scores are trusted.

Reusable diagnostic code lives in:

- `shared/pruning_diagnostics.py`

## Constructive path-capacity result

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
- Across ten completed strong TinyViT seeds, V4 is positive vs magnitude on `7/10`, matches the best evaluated candidate on `8/10`, and averages `+2.85` points over magnitude. Seed 306 and seed 310 define the important failure boundary: V4-style live repairs can beat or match magnitude while still missing a much more trainable SynFlow basin.
- V5 adds a SynFlow masked-recovery prior: if SynFlow's masked-before accuracy is at least magnitude and close to the selected repair, prefer SynFlow. Across the ten completed strong TinyViT seeds, V5 is positive vs magnitude on `8/10`, matches the best evaluated candidate on `10/10`, averages `+3.95` points over magnitude, and leaves zero gap to the evaluated oracle.
- Two fresh V5 prospective seeds now test both branches. Seed 308 validates the feature/SynFlow branch at `+2.93` points over magnitude. Seed 310 validates the SynFlow masked-recovery-prior branch at `+9.30` points over magnitude while every liveness-first repair stays at the magnitude floor. This is the strongest transformer-analogue result so far, but it is still a small TinyViT boundary set rather than a large-model pruning claim.
- On a first TinyImageNet-200 external proxy subset, viability methods beat magnitude but the selector-picked split trails plain reserve; this is a boundary condition, not a solved external benchmark.
- On ImageNet-pretrained ResNet-18 TinyImageNet, the current viability selector fails: at `99%` all methods collapse, and at `95%` magnitude strongly beats the homeostatic masks. This is now the main external-validity limitation.
- A feature-preserving liveness repair fixes most of that failure: across two `95%` seeds it eliminates dead outputs while matching magnitude within `0.05` points on average, showing pretrained systems need minimal liveness repair on top of feature-subspace preservation.
- At the `99%` pretrained TinyImageNet cliff, feature-viability repair is the only positive intervention so far, but absolute recovery remains near chance.
- On a fresh pretrained TinyImageNet `95%` seed, the corrected tradeoff selector picks feature repair, preserves magnitude-level accuracy, eliminates dead outputs, and avoids homeostatic masks that collapse to `3-4%`.
- A fresh full-CIFAR from-scratch comparison complicates the selector: feature repair beats both magnitude and reserve, though it leaves more dead outputs. The family selector must optimize accuracy/liveness tradeoff, not just route-floor class.
- The corrected tradeoff selector now does this prospectively: on two fresh full-CIFAR ResNet-20-style seeds, it selected feature repair before fine-tuning and beat magnitude by `+5.37` points with `2/2` wins, while also beating plain reserve on mean.

Primary artifacts:

- `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md`
- `figures/04_circuit_viability/README.md`
- `experiments/04_criticality_pruning/PATH_CAPACITY_SYNTHESIS.md`
- `experiments/04_criticality_pruning/CIFAR10_TINY_VIT_CIRCUIT_VIABILITY_98PCT.md`
- `experiments/04_criticality_pruning/PATH_CAPACITY_PRUNING.md`
- `docs/CLAIM_EVIDENCE_LEDGER.md`
- `docs/CLAIM_AUDIT.md`
- `docs/NEUROSCIENCE_FRAMING.md`

## Secondary result

A tiny dense-tail path correction can improve one-shot severe pruning, but it is not a universal replacement for magnitude pruning.

Six-seed CIFAR-10 GPU replicate:

| Sparsity | Alpha | One-shot delta vs magnitude | One-shot wins | After-FT delta vs magnitude | After-FT wins |
|---:|---:|---:|---:|---:|---:|
| `95%` | `0.03` | `+1.49` pts | `5/6` | `-0.44` pts | `1/6` |
| `98%` | `0.03` | `+1.13` pts | `4/6` | `-0.29` pts | `3/6` |
| `99%` | `0.03` | `-0.19` pts | `3/6` | `-1.42` pts | `1/6` |

Current rule encoded in `shared/adaptive_path_pruning.py`:

- `balanced` / `one_shot`: tiny `alpha=0.03` near the severe sparsity cliff.
- `recovery`: `alpha=0.0` unless a domain-specific sweep proves otherwise.

## Repository map

| Path | Purpose |
|---|---|
| `experiments/04_criticality_pruning/README.md` | Main experiment navigation page. |
| `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md` | Strongest cross-dataset result. |
| `experiments/04_criticality_pruning/PATH_CAPACITY_SYNTHESIS.md` | Current constructive path-capacity evidence and limitations. |
| `experiments/04_criticality_pruning/TINY_VIT_STRONG_SELECTOR_BOUNDARY_SYNTHESIS.md` | Strong TinyViT selector-boundary synthesis. |
| `experiments/04_criticality_pruning/PATH_CAPACITY_PRUNING.md` | Method framing, neuroscience mapping, and experiment readout. |
| `experiments/04_criticality_pruning/LOW_ALPHA_TRANSFER_SYNTHESIS.md` | Secondary low-alpha pruning result and limitations. |
| `experiments/04_criticality_pruning/CIFAR10_CNN_SYNFLOW_PATHOLOGY.md` | CIFAR-10 CUDA replication of the SynFlow failure. |
| `docs/CLAIM_EVIDENCE_LEDGER.md` | Claim-by-claim evidence boundary and limitations. |
| `docs/CLAIM_AUDIT.md` | Generated audit proving headline numbers match JSON artifacts. |
| `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md` | Canonical paper-style synthesis of the mechanism and evidence. |
| `docs/NEUROSCIENCE_FRAMING.md` | Neuroscience-to-algorithm mapping for circuit viability. |
| `docs/ROUTE_DEFICIT_PREDICTOR.md` | Current residual route-deficit predictor and evidence. |
| `docs/ROADMAP.md` | Next experiments needed to turn the mechanism into a stronger theory. |
| `shared/pruning_diagnostics.py` | Structural mask diagnostics for dense-bridge collapse. |
| `shared/path_capacity_pruning.py` | Reusable capacity-constrained pruning utility. |
| `shared/residual_route_capacity.py` | Reusable residual route-family capacity predictor and mask builder. |
| `shared/circuit_viability_selector.py` | Ecology and feature/liveness tradeoff selectors. |
| `shared/adaptive_path_pruning.py` | Reusable low-alpha dense-tail path correction utilities. |
| `results/04_criticality_pruning/` | JSON result artifacts used by synthesis/audit scripts. |

## Reproduce the strongest claim

The synthesis and audit are lightweight because they read checked-in result artifacts:

```powershell
python experiments\04_criticality_pruning\synthesize_synflow_pathology.py
python experiments\04_criticality_pruning\audit_synflow_pathology.py
python experiments\04_criticality_pruning\synthesize_path_capacity.py
python experiments\04_criticality_pruning\audit_circuit_viability_claims.py
```

To rerun the full CIFAR pathology experiment on GPU:

```powershell
python experiments\04_criticality_pruning\cifar10_cnn_synflow_pathology.py
```

That command trains small CIFAR-10 CNNs, compares magnitude/global SynFlow/layerwise SynFlow at `98%` and `99%` sparsity, and writes:

- `results/04_criticality_pruning/cifar10_cnn_synflow_pathology.json`
- `experiments/04_criticality_pruning/CIFAR10_CNN_SYNFLOW_PATHOLOGY.md`

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiments use PyTorch and torchvision. GPU is optional for synthesis/audits, but recommended for rerunning CIFAR training.

## Claim boundaries

Supported by current artifacts:

- Global SynFlow can structurally fail at severe CNN sparsity by allocating zero weights to dense classifier bridges.
- The failure replicated on Fashion-MNIST CNN and CIFAR-10 CNN.
- Layerwise SynFlow is not sufficient in the tested CNNs.
- Path-capacity reserve constraints can rescue SynFlow collapse under the same parameter budget.
- Path-capacity reserve beats magnitude in the strongest CIFAR-10 CNN `99%` reserve sweep and transfers positively to Fashion-MNIST CNN.
- Tiny path correction can modestly improve one-shot severe pruning around `95-98%` on CIFAR.

Not supported:

- Adaptive path correction is a universal replacement for magnitude pruning.
- Low-alpha path correction is the best fine-tuning initializer.
- Merely forcing every hidden unit to stay live is enough to improve masks.
- Current output-count capacity reserve is sufficient for all residual-network settings.
- Naive presynaptic activation weighting solves residual-route quality.
- Route-floor-only family selection is enough to choose between homeostasis and feature repair.
- The current TinyViT evidence proves robust transformer or large-model pruning transfer.

## Status

The repo has moved beyond the initial pilot phase. The strongest current contribution is the SynFlow dense-bridge collapse finding plus constructive path-capacity experiments showing that circuit-viability constraints can rescue and sometimes improve extreme-sparsity pruning. The main open problem is to replace hand-chosen capacity reserves and hand-weighted tradeoff scores with a predictive route-quality theory that transfers cleanly to residual, pretrained, and transformer architectures. TinyViT now has a real transformer analogue: V5 has validated both a feature-preservation branch and a SynFlow masked-recovery-prior branch prospectively, while the remaining limitation is scale and generality beyond this small transformer boundary set.
