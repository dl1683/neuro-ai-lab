# Adaptive Path Rule Image Evaluation

This document evaluates the current practical rule on completed image-model runs.

## Rule

Use magnitude at 90% sparsity; use a weak path correction at 95%; use a stronger path correction at 98%, with CNNs using dense-layer-only path correction.

## Aggregate

Overall mean delta vs pure magnitude: `0.0171` over `18` paired cases; wins `9`, ties `6`, losses `3`.

| Study | Mean delta | Wins | Ties | Losses | N |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 MLP | `0.0075` | `4` | `2` | `0` | `6` |
| Fashion-MNIST CNN dense-hybrid | `0.0232` | `3` | `2` | `1` | `6` |
| Fashion-MNIST MLP | `0.0206` | `2` | `2` | `2` | `6` |

## Paired rows

| Study | Seed | Sparsity | Alpha | Adaptive accuracy | Magnitude accuracy | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST MLP | `31` | `0.90` | `0.00` | `0.7817` | `0.7817` | `0.0000` |
| Fashion-MNIST MLP | `31` | `0.95` | `0.25` | `0.7073` | `0.6520` | `0.0553` |
| Fashion-MNIST MLP | `31` | `0.98` | `0.40` | `0.5730` | `0.4780` | `0.0950` |
| Fashion-MNIST MLP | `32` | `0.90` | `0.00` | `0.8247` | `0.8247` | `0.0000` |
| Fashion-MNIST MLP | `32` | `0.95` | `0.25` | `0.7193` | `0.7200` | `-0.0007` |
| Fashion-MNIST MLP | `32` | `0.98` | `0.40` | `0.5667` | `0.5927` | `-0.0260` |
| CIFAR-10 MLP | `41` | `0.90` | `0.00` | `0.4304` | `0.4304` | `0.0000` |
| CIFAR-10 MLP | `41` | `0.95` | `0.25` | `0.3890` | `0.3886` | `0.0004` |
| CIFAR-10 MLP | `41` | `0.98` | `0.50` | `0.3072` | `0.2938` | `0.0134` |
| CIFAR-10 MLP | `42` | `0.90` | `0.00` | `0.3898` | `0.3898` | `0.0000` |
| CIFAR-10 MLP | `42` | `0.95` | `0.25` | `0.3086` | `0.3036` | `0.0050` |
| CIFAR-10 MLP | `42` | `0.98` | `0.50` | `0.2592` | `0.2330` | `0.0262` |
| Fashion-MNIST CNN dense-hybrid | `51` | `0.90` | `0.00` | `0.7602` | `0.7602` | `0.0000` |
| Fashion-MNIST CNN dense-hybrid | `51` | `0.95` | `0.20` | `0.6520` | `0.6230` | `0.0290` |
| Fashion-MNIST CNN dense-hybrid | `51` | `0.98` | `0.20` | `0.5513` | `0.4195` | `0.1318` |
| Fashion-MNIST CNN dense-hybrid | `52` | `0.90` | `0.00` | `0.7325` | `0.7325` | `0.0000` |
| Fashion-MNIST CNN dense-hybrid | `52` | `0.95` | `0.20` | `0.6075` | `0.5933` | `0.0142` |
| Fashion-MNIST CNN dense-hybrid | `52` | `0.98` | `0.20` | `0.4323` | `0.4680` | `-0.0358` |
