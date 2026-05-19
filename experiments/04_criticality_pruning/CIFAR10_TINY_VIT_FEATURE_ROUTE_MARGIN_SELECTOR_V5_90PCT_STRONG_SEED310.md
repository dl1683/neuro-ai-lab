# CIFAR-10 TinyViT V5 SynFlow-Prior Branch Validation at 90%

Fresh full-train TinyViT CIFAR-10 90% sparsity validation of the V5 selector on seed 310, chosen by a dense-only branch scan because V5 selected global SynFlow through the SynFlow masked-recovery-prior branch before any masked fine-tuning.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[310]`
Dense accuracy mean: `0.7209`

| Method | Before FT | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0482` | `0.0586` |  |  | `0.0047` | `96.0` | `2.0` |
| `global_synflow` | `0.0827` | `0.1516` | `+0.0930` | `1/1` | `0.0164` | `98.0` | `83.0` |
| `minimal_liveness_repair` | `0.0496` | `0.0590` | `+0.0004` | `1/1` | `0.0053` | `1.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.0496` | `0.0588` | `+0.0002` | `1/1` | `0.0045` | `1.0` | `0.0` |
| `all_route_liveness_floor` | `0.0489` | `0.0586` | `+0.0000` | `0/1` | `0.0050` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.0827` | `0.1516` | `+0.0930` | `1/1` | `0.0164` | `98.0` | `83.0` |

## Selector decisions

- seed `310`: selected `global_synflow` via `synflow_masked_recovery_prior`; global_synflow=0.0164/before0.0827/dead181, minimal_liveness_repair=0.0053/before0.0496/dead1, all_route_liveness_floor=0.0050/before0.0489/dead0, magnitude=0.0047/before0.0482/dead98, attn_mlp_readout_repair=0.0045/before0.0496/dead1

## Interpretation

This run tests the V5 rule's most important unresolved branch. The seed was selected only from dense-model diagnostics, then evaluated with full masked fine-tuning across all candidates.

The neuroscientific interpretation is the trainability side of circuit viability: a sparse circuit can preserve live pathways and still fail if the remaining masked network cannot recover function. The SynFlow prior is a developmental-stability heuristic for preserving globally connected signal paths before task-specific repair.
