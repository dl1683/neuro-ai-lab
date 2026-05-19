# Neuro-AI Lab Real Experiment Ledger

The project has moved from shallow synthetic pilots to one serious empirical thread: **label-free path-flow pruning**.

## Current strongest result

Path-flow pruning is the current exceptional candidate.

Files:

- `experiments/04_criticality_pruning/run.py`
- `experiments/04_criticality_pruning/DISCOVERY_REPORT.md`
- `experiments/04_criticality_pruning/ablate_path_flow.py`
- `experiments/04_criticality_pruning/calibrate_path_flow.py`
- `experiments/04_criticality_pruning/finetune_path_flow.py`
- `experiments/04_criticality_pruning/extreme_sparsity.py`

Headline evidence:

- Core benchmark: path-flow beats magnitude by `+17.44` accuracy points at `90-95%` sparsity.
- Core benchmark: path-flow slightly beats gradient saliency by `+0.76` points while using no labels/gradients at pruning time.
- Ablation: removing input activity drops accuracy by `7.85` points, making it the dominant ingredient.
- Calibration: `10%` unlabeled calibration drops only `1.53` points vs full calibration.
- Fine-tuning: after masked fine-tuning, path-flow reaches `89.03%` average accuracy at `90-95%` sparsity vs `76.10%` for magnitude and `88.74%` for gradient saliency.
- Extreme sparsity: at `99%` sparsity, path-flow gets `37.27%` vs `31.54%` gradient saliency and `20.03%` magnitude.

Interpretation: the original criticality hypothesis is not what survived. The result that survived is pathway preservation: score weights by whether they belong to active input-hidden-output routes.

## Other experiments

| Experiment | Status | Current interpretation |
|---|---|---|
| 01 Grokking Prediction | Failed | Current MLP modular-addition setup memorizes and does not generalize. Needs a known transformer grokking setup. |
| 02 Sleep Training | Modest real win | Compression/regularization tradeoff, but not exceptional yet. |
| 03 Reconsolidation | Modest real win | Beats EWC/naive on split sklearn digits, but needs stronger baselines. |
| 04 Path-Flow Pruning | Strongest result | Real multi-seed evidence, ablations, calibration, fine-tuning, and extreme-sparsity boundary. |
| 05 DDM as Depth | Efficiency win, weak theory | Early exit works on easy digits task, but DDM fit is weak. |

## Current priority

Stop spreading effort across five ideas. The next serious escalation should be path-flow pruning on a harder model/dataset, ideally Fashion-MNIST MLP/CNN or CIFAR-10 small CNN, with SNIP/SynFlow-style comparisons.

## Fashion-MNIST update

The first cross-dataset test falsified the naive path-flow formula: on a Torch Fashion-MNIST MLP, pure magnitude beat full path-flow. Follow-up experiments found why and recovered a narrower result.

Best corrected Fashion-MNIST variant:

- Remove hidden-balance term.
- Use per-pixel standard deviation rather than mean absolute normalized input.
- Blend path signal weakly with magnitude using `alpha=0.25`.

Result versus magnitude:

- `90%` sparsity: `-1.05` points.
- `95%` sparsity: `+2.73` points.
- `98%` sparsity: `+7.02` points.

Updated hypothesis: path-flow is not a universal replacement for magnitude. It is a severe-sparsity correction that helps near the compression cliff when the activation statistic is appropriate for the data modality.

## Fashion-MNIST fine-tuning update

Corrected path blend after masked fine-tuning:

- `95%` sparsity: `83.40%` vs `83.10%` magnitude (`+0.30`).
- `98%` sparsity: `80.33%` vs `79.65%` magnitude (`+0.68`).

Interpretation: after recovery training, the edge over magnitude is small but still positive. The larger value of path-flow is as a severe-sparsity mask initializer before fine-tuning.

## Evidence synthesis update

Command: `python experiments/04_criticality_pruning/synthesize_evidence.py`

The current claim is now narrowed and statistically summarized:

> Path-aware pruning is not universally better than magnitude. It is strongest as a label-free severe-sparsity path-preservation correction. On sklearn digits MLPs, full path-flow dominates magnitude and slightly beats gradient saliency. On Fashion-MNIST MLPs, naive path-flow fails, but a no-balance weak path/magnitude blend improves severe-sparsity masks and keeps a small edge after fine-tuning.

Key paired evidence:

- Sklearn digits `90-95%` sparsity, path-flow vs magnitude: mean delta `+17.44` points, bootstrap 95% CI `[+13.39, +22.02]`, wins `10/10`.
- Sklearn digits `90-95%` sparsity, path-flow vs gradient saliency: mean delta `+0.76` points, CI `[-1.67, +3.34]`, wins `6/10`. This is promising but not decisive.
- Sklearn digits `97-99%` sparsity, path-flow vs magnitude: mean delta `+20.37` points, CI `[+16.40, +24.71]`, wins `15/15`.
- Sklearn digits `97-99%` sparsity, path-flow vs gradient saliency: mean delta `+4.22` points, CI `[+0.56, +8.22]`, wins `8/15`.
- Fashion-MNIST adaptive rule vs magnitude: mean delta `+3.25` points, CI `[+0.37, +7.05]`, wins `3/6`, ties `2/6`.

New artifact: `experiments/04_criticality_pruning/EVIDENCE_SYNTHESIS.md`.

## CIFAR-10 exploratory transfer

A Torch CIFAR-10 MLP check also supports the narrowed severe-sparsity version of the hypothesis:

- `90%` sparsity: best path blend `+0.22` points over magnitude.
- `95%` sparsity: best path blend `+0.27` points over magnitude.
- `98%` sparsity: best path blend `+1.98` points over magnitude.

This is not a large result, but it transfers the same pattern seen on Fashion-MNIST: path correction is most useful near the sparsity cliff.

## CNN boundary update

Fashion-MNIST CNN transfer is negative:

- `90%` sparsity: magnitude `74.29%`, best path correction `72.65%`.
- `95%` sparsity: magnitude `60.54%`, best path correction `56.19%`.
- `98%` sparsity: magnitude `44.19%`, best path correction `43.33%`.

Conclusion: current path-flow is an MLP/path-matrix result, not yet a CNN pruning method.

## CNN dense-hybrid recovery

A corrected CNN test recovered the path-flow pattern:

- Conv layers use magnitude.
- Dense layers use corrected no-balance path-flow blend.

Results:

- `90%` sparsity: `+0.36` points over magnitude.
- `95%` sparsity: `+2.16` points over magnitude.
- `98%` sparsity: `+4.80` points over magnitude.

Interpretation: path-flow works for dense bottleneck/path matrices inside a CNN, but not for naive convolutional filter pruning.

## CNN dense-hybrid fine-tuning boundary

Dense-hybrid path-flow remained slightly ahead at `95%` after fine-tuning (`85.00%` vs `84.69%`) but lost at `98%` after fine-tuning (`79.55%` vs `80.70%`).

Interpretation: in CNNs, the dense-hybrid path correction is a one-shot severe-sparsity mask-quality improvement, not yet a consistently better post-recovery sparse model.

## Sparsity-cliff path correction law

A meta-analysis across Fashion-MNIST MLP, CIFAR-10 MLP, and Fashion-MNIST CNN dense-hybrid checks found:

- `90%` sparsity: mean best alpha `0.10`, mean gain `+0.19` points.
- `95%` sparsity: mean best alpha `0.28`, mean gain `+2.21` points.
- `98%` sparsity: mean best alpha `0.48`, mean gain `+4.62` points.

Current best formulation: path correction should be sparsity-dependent. Keep magnitude dominant at moderate sparsity; increase path correction only near the pruning cliff.

## Adaptive image rule result

Across Fashion-MNIST MLP, CIFAR-10 MLP, and Fashion-MNIST CNN dense-hybrid, the adaptive path rule beats/ties magnitude in `15/18` paired image-model cases with mean `+1.71` points.

This is the most general current result: path correction should be sparsity-adaptive and applied only where dense path structure exists.

## SynFlow comparison update

Fashion-MNIST MLP SynFlow baseline:

- `90%`: adaptive path `80.32%`, SynFlow `79.05%`.
- `95%`: adaptive path `71.33%`, SynFlow `68.68%`.
- `98%`: adaptive path `55.30%`, SynFlow `44.25%`.

Mean adaptive path gain: `+4.99` points over SynFlow and `+1.50` over magnitude.

This is the strongest label-free baseline comparison so far.

## SynFlow CNN comparison

Fashion-MNIST CNN with SynFlow:

- SynFlow wins at `90%` and slightly at `95%`.
- At `98%`, SynFlow collapses to `10.08%`; adaptive dense-hybrid gets `49.46%`, magnitude gets `43.51%`.
- Averaged over `90/95/98%`, adaptive dense-hybrid beats SynFlow by `+12.11` points and magnitude by `+2.70` points.

This strengthens the severe-sparsity guardrail framing: adaptive path correction helps avoid pruning cliffs that can break other label-free methods.

## Label-free baseline synthesis

Across Fashion-MNIST MLP/CNN with SynFlow baselines:

- Adaptive path vs SynFlow overall: `+8.55` points, wins `9/12`.
- At `98%` sparsity: adaptive path vs SynFlow `+25.22` points, wins `4/4`.
- Adaptive path vs magnitude overall is mixed: `+2.10` mean, wins/ties/losses `4/4/4`.

Best current framing: adaptive path correction is a label-free guardrail against severe-sparsity collapse, especially compared with SynFlow. Magnitude remains a strong baseline.

## Claim card and audit

Added reproducible claim card workflow:

- `python experiments/04_criticality_pruning/claim_card.py`
- `python experiments/04_criticality_pruning/audit_claim_card.py`

Audit passed. The current claim card grades the result as `strong severe-sparsity guardrail evidence` and ties it to source artifacts.

## Reusable implementation

The method now has reusable code in `shared/adaptive_path_pruning.py` plus a minimal runnable example at `experiments/04_criticality_pruning/example_adaptive_path_pruning.py`.

## Latest pressure test: SynFlow mechanism at 98%

The strongest real finding is now mechanistic, not just scoreboard-based:

- Global SynFlow at `98%` on the Fashion-MNIST CNN assigns `0.0%` keep rate to `fc1`, killing every dense bridge unit and staying at chance after masked fine-tuning.
- Layerwise SynFlow fixes the total bridge starvation but only recovers to `50.21%` after fine-tuning.
- Magnitude recovers to `83.41%`; adaptive dense-hybrid recovers to `82.44%` and has much better one-shot accuracy than SynFlow.
- This means the practical discovery is a severe-sparsity guardrail: path-aware allocation avoids a catastrophic pruning cliff, but magnitude remains the current best recovery initializer after fine-tuning.

See:

- `experiments/04_criticality_pruning/SYNFLOW_CNN_MASK_FORENSICS_98PCT.md`
- `experiments/04_criticality_pruning/SYNFLOW_CNN_LAYERWISE_RESCUE_98PCT.md`

## Practical rule update: weak path correction beats the old aggressive schedule

A CNN `98%` alpha sweep found the usable setting:

- Magnitude (`alpha=0.00`): `44.55%` one-shot, `83.34%` after masked fine-tuning.
- Balanced path correction (`alpha=0.08`): `49.38%` one-shot, `83.65%` after fine-tuning.
- Recovery path correction (`alpha=0.05`): `45.58%` one-shot, `83.94%` after fine-tuning.
- Aggressive correction (`alpha=0.50`): collapses to `27.74%` one-shot and only `80.34%` after fine-tuning.

The implementation in `shared/adaptive_path_pruning.py` now defaults to the low-alpha balanced rule and exposes explicit `balanced`, `recovery`, and `one_shot` objectives.

## CIFAR transfer result

The low-alpha rule transferred to real CIFAR-10 CNN, but only narrowly:

- Magnitude: `16.65%` one-shot, `45.28%` after FT.
- `alpha=0.03`: `18.14%` one-shot, `45.79%` after FT.
- `alpha=0.05`: `16.50%` one-shot, `46.40%` after FT.
- `alpha>=0.08`: hurts; `alpha=0.15` collapses after FT to `24.24%`.

This is now a real, falsified, transfer-tested finding: tiny path correction can help at the severe pruning cliff, but larger correction does not transfer.

## Six-seed CIFAR GPU replicate

The low-alpha idea survived as a one-shot signal but not as a fine-tuning initializer:

- `95%`, `alpha=0.03`: `+1.49` one-shot points, after-FT `-0.44`.
- `98%`, `alpha=0.03`: `+1.13` one-shot points, after-FT `-0.29`.
- `99%`, unstable; recovery gets worse.

So the next real mechanism to test is not more alpha sweeping. It is structural bridge preservation: can we keep the one-shot benefit without damaging recovery by explicitly preventing dense-tail bridge collapse?

## CIFAR SynFlow pathology replicated

Global SynFlow failed catastrophically on CIFAR-10 CNN:

- `98%`: `fc1_keep_rate=0.0000`, all `192/192` dense bridge units dead, after-FT accuracy `9.76%` vs magnitude `44.08%`.
- `99%`: same total `fc1` starvation, after-FT accuracy `9.76%` vs magnitude `33.24%`.
- Layerwise SynFlow partially restores `fc1`, but still loses every paired comparison to magnitude.

This is now the strongest result: a transferable global SynFlow allocation pathology for CNNs with dense classifier tails.

## SynFlow pathology synthesis artifact

The strongest current artifact is `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md`:

- `3/3` severe-sparsity CNN cases had global SynFlow allocate zero weights to `fc1`.
- Mean after-FT delta vs magnitude was `-42.80` points for global SynFlow.
- Layerwise SynFlow partially repairs allocation but still averages `-22.21` points vs magnitude.

Added `shared/pruning_diagnostics.py` to catch this class of collapse directly.

## Constructive path-capacity result

The first multi-cut capacity method is now in the repo:

- `shared/path_capacity_pruning.py`
- `experiments/04_criticality_pruning/cifar10_cnn_path_capacity_pruning.py`
- `experiments/04_criticality_pruning/cifar10_cnn_multicut_capacity_pruning.py`
- `experiments/04_criticality_pruning/CIFAR10_CNN_MULTICUT_CAPACITY_PRUNING.md`

The important result is at `99%` sparsity on CIFAR-10 CNN: multi-cut capacity turns SynFlow from chance (`9.76%`) into a trainable mask (`33.41%` after FT), slightly above magnitude (`32.62%`) in this four-seed run. This is early, but it is the first constructive evidence for the circuit-viability thesis.
