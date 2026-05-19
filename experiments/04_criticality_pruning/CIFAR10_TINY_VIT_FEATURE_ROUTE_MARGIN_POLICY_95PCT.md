# CIFAR-10 TinyViT Feature-Route Margin Policy at 95%

Policy projection over the fresh TinyViT feature-subspace selector run. If feature alignment is within a small margin, prefer the candidate with lower transformer route death.

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.0882` |  |  | `0.0164` | `512.0` | `305.5` |
| `global_synflow` | `0.1357` | `+0.0475` | `2/2` | `0.0193` | `235.0` | `134.5` |
| `feature_subspace_policy` | `0.1224` | `+0.0342` | `1/2` | `0.0210` | `376.5` | `217.5` |
| `feature_route_margin_policy` | `0.1357` | `+0.0475` | `2/2` | `0.0193` | `235.0` | `134.5` |

## Decisions

- seed `292`: selected `global_synflow` via `feature_argmax` centered CLS `0.0218` route dead `376`
- seed `293`: selected `global_synflow` via `synflow_margin_route_risk` centered CLS `0.0168` route dead `363`

## Interpretation

This is a policy projection, not a fresh training run. It uses the already evaluated candidate masks from the prospective feature-subspace selector. The result shows why argmax feature alignment is not enough: when scores are close, route-death risk can resolve the ambiguity.
