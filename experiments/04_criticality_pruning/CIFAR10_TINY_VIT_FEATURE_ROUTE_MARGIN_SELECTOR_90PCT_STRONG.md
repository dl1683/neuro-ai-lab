# CIFAR-10 TinyViT Feature-Route Margin Selector at 90%: Stronger Recipe

TinyViT CIFAR-10 full-train stronger-recipe 90% sparsity feature-route margin selector pilot. Uses full CIFAR-10 train/test, 20 dense epochs, and 5 masked fine-tune epochs.

Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`
Seeds: `[298]`
Dense accuracy mean: `0.7162`

| Method | After FT | Delta vs magnitude | Wins | Centered CLS cosine | MLP-down dead | Attn-out dead |
|---|---:|---:|---:|---:|---:|---:|
| `magnitude` | `0.1670` |  |  | `0.0358` | `73.0` | `1.0` |
| `global_synflow` | `0.1435` | `-0.0235` | `0/1` | `0.0440` | `92.0` | `70.0` |
| `minimal_liveness_repair` | `0.1654` | `-0.0016` | `0/1` | `0.0371` | `0.0` | `0.0` |
| `attn_mlp_readout_repair` | `0.1660` | `-0.0010` | `0/1` | `0.0382` | `0.0` | `0.0` |
| `all_route_liveness_floor` | `0.1646` | `-0.0024` | `0/1` | `0.0371` | `0.0` | `0.0` |
| `feature_route_margin_policy` | `0.1435` | `-0.0235` | `0/1` | `0.0440` | `92.0` | `70.0` |

## Selector decisions

- seed `298`: selected `global_synflow` via `feature_argmax`; global_synflow=0.0440/dead162, attn_mlp_readout_repair=0.0382/dead0, minimal_liveness_repair=0.0371/dead0, all_route_liveness_floor=0.0371/dead0, magnitude=0.0358/dead74

## Interpretation

This is a one-seed pilot to test whether the TinyViT feature-route margin selector remains meaningful when dense training is stronger and the evaluation uses the full CIFAR-10 train/test split. If positive, this should be expanded to more seeds.
