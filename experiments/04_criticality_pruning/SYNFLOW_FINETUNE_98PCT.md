# SynFlow Fine-Tuning Pressure Test at 98% Sparsity

This test compares magnitude, SynFlow, and adaptive path masks at `98%` sparsity before and after masked fine-tuning on Fashion-MNIST MLP and CNN.

| Model | Method | Before FT | After FT | After retention |
|---|---|---:|---:|---:|
| Fashion-MNIST CNN | `magnitude` | `0.4591` | `0.8086` | `0.9550` |
| Fashion-MNIST CNN | `synflow` | `0.1008` | `0.1028` | `0.1213` |
| Fashion-MNIST CNN | `adaptive_path` | `0.4726` | `0.7953` | `0.9392` |
| Fashion-MNIST MLP | `magnitude` | `0.5353` | `0.7967` | `0.9410` |
| Fashion-MNIST MLP | `synflow` | `0.4425` | `0.7775` | `0.9184` |
| Fashion-MNIST MLP | `adaptive_path` | `0.5530` | `0.7893` | `0.9323` |

Mean adaptive minus SynFlow after fine-tuning: `+0.3522`.

Mean adaptive minus magnitude after fine-tuning: `-0.0104`.

Interpretation: adaptive path correction survives the pressure test as a severe-sparsity guardrail against SynFlow collapse, but it does not beat magnitude as a post-fine-tuning initializer on this run.
