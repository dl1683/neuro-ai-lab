# Path-Flow Extreme Sparsity

One-shot pruning at `97%`, `98%`, and `99%` sparsity on trained sklearn-digits MLPs across five seeds.

| Sparsity | Method | Mean accuracy | Std | Mean retention | Hidden coverage |
|---:|---|---:|---:|---:|---:|
| `0.97` | `magnitude` | `0.3695` | `0.0434` | `0.3855` | `0.3422` |
| `0.97` | `gradient_saliency` | `0.5889` | `0.0503` | `0.6142` | `0.4484` |
| `0.97` | `path_flow` | `0.6051` | `0.0808` | `0.6310` | `0.2750` |
| `0.98` | `magnitude` | `0.3215` | `0.0465` | `0.3354` | `0.2141` |
| `0.98` | `gradient_saliency` | `0.4715` | `0.0613` | `0.4920` | `0.3047` |
| `0.98` | `path_flow` | `0.5246` | `0.0862` | `0.5472` | `0.2141` |
| `0.99` | `magnitude` | `0.2003` | `0.0252` | `0.2089` | `0.0844` |
| `0.99` | `gradient_saliency` | `0.3154` | `0.0810` | `0.3292` | `0.1234` |
| `0.99` | `path_flow` | `0.3727` | `0.0546` | `0.3887` | `0.1375` |
