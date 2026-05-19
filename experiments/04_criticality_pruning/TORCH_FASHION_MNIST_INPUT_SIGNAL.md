# Torch Fashion-MNIST Input Signal Check

Test whether Fashion-MNIST path-flow failure is caused by bad input activity estimates on normalized images. Compare abs normalized input vs per-pixel std input signal.

| Sparsity | Method | Mean accuracy | Std | Retention |
|---:|---|---:|---:|---:|
| `0.90` | `magnitude` | `0.8032` | `0.0215` | `0.9485` |
| `0.90` | `path_abs_input` | `0.5768` | `0.0545` | `0.6808` |
| `0.90` | `path_std_input` | `0.5770` | `0.0567` | `0.6810` |
| `0.90` | `path_std_no_balance` | `0.7290` | `0.0100` | `0.8610` |
| `0.95` | `magnitude` | `0.6860` | `0.0340` | `0.8100` |
| `0.95` | `path_abs_input` | `0.4833` | `0.0167` | `0.5707` |
| `0.95` | `path_std_input` | `0.5122` | `0.0318` | `0.6047` |
| `0.95` | `path_std_no_balance` | `0.6605` | `0.0042` | `0.7802` |
| `0.98` | `magnitude` | `0.5353` | `0.0573` | `0.6318` |
| `0.98` | `path_abs_input` | `0.3470` | `0.1187` | `0.4110` |
| `0.98` | `path_std_input` | `0.3638` | `0.0485` | `0.4302` |
| `0.98` | `path_std_no_balance` | `0.4350` | `0.0027` | `0.5138` |
