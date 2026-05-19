# Route-Deficit Predicted Capacity

This note documents the current residual-network version of Path-Capacity Pruning.

## Motivation

Plain output-count capacity reserve fixes the obvious death problem in TinyResNet masks, but it still fails at the `99%` sparsity cliff. The route-quality audit showed why: once most outputs are technically alive, raw dead-output count stops explaining recovery. Residual recovery depends on route-family balance across:

- main transformation paths;
- projection shortcuts;
- classifier readout.

The goal is to select a capacity split before recovery fine-tuning, not by reading the final accuracy table.

## Predictor

Implementation:

- `shared/residual_route_capacity.py`

The first predictor does this:

1. Build a magnitude mask as a viability template.
2. Build a plain reserve mask as the candidate that prevents death but underperforms.
3. Compute route-quality diagnostics for both masks.
4. Measure projection deficit: template projection capacity minus candidate projection capacity.
5. Measure readout deficit: template classifier-readout score minus candidate classifier-readout score.
6. Keep a main-path floor.
7. Allocate remaining protected capacity between projection and readout deficits.
8. Build a route-family capacity mask and fill the rest globally by SynFlow.

The current predictor still contains two hand-set choices:

- main-path floor: `0.40`
- projection reliability weight: `2.0`

These are not final theory. They are scaffolding for converting the tuned route split into a measured-deficit rule.

## Current evidence

Four TinyResNet `99%` seeds:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `25.04%` | baseline | baseline |
| plain reserve `0.60` | `20.21%` | `-4.83` pts | `0/4` |
| tuned `40/35/25` split | `24.88%` | `-0.17` pts | `3/4` |
| predicted route-deficit split | `25.12%` | `+0.08` pts | `2/4` |

Primary artifact:

- `experiments/04_criticality_pruning/CIFAR10_TINY_RESNET_PREDICTED_ROUTE_SPLIT_99PCT.md`
- `results/04_criticality_pruning/cifar10_tiny_resnet_predicted_route_split_99pct.json`

## Interpretation

The effect size is tiny. The result should not be sold as a solved residual pruning method.

The important step is methodological:

- output liveness was insufficient;
- route-quality audit identified projection/readout imbalance;
- a tuned projection/readout split beat magnitude;
- a measured-deficit predictor recovered nearly the same split before fine-tuning and narrowly beat magnitude on mean.

This makes the neuroscience story more concrete. Circuit viability is not a slogan here; it becomes a pre-recovery route-deficit measurement that changes the mask.

## Next step

The predictor needs to stop relying on fixed constants.

Concrete next experiments:

1. Derive the main-path floor from parameter budget and block depth.
2. Derive projection reliability weight from route-quality correlations rather than setting it manually.
3. Validate on more TinyResNet seeds.
4. Transfer to a stronger residual backbone.
5. Add transformer route-family analogues once residual prediction is stable.

## Derived predictor follow-up

A fresh four-seed follow-up tested whether the remaining constants could be removed or weakened.

Compared methods:

- `tuned_40_35_25`: hand-selected route-family split.
- `fixed_deficit_predictor`: the first route-deficit predictor with main floor and projection reliability weight.
- `relative_deficit_predictor`: equal route-family priors, split by relative route deficits.
- `sqrt_width_deficit_predictor`: route-family prior derived from square root of output width, then adjusted by deficits.

Fresh TinyResNet `99%` seeds `[207, 208, 209, 210]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `24.98%` | baseline | baseline |
| plain reserve `0.60` | `20.70%` | `-4.28` pts | `0/4` |
| tuned `40/35/25` | `24.61%` | `-0.37` pts | `2/4` |
| fixed deficit predictor | `24.59%` | `-0.39` pts | `2/4` |
| relative deficit predictor | `23.81%` | `-1.17` pts | `1/4` |
| sqrt-width deficit predictor | `22.75%` | `-2.23` pts | `0/4` |

Interpretation:

- Removing the constants did not improve the method.
- The equal-family predictor overallocated readout and increased dead outputs.
- The width-derived predictor underallocated readout and performed worse.
- The fixed predictor and tuned split still close most of the plain-reserve gap, but they do not robustly beat magnitude on fresh seeds.

Current honest status:

**Route-deficit prediction is promising as a diagnostic and gap-closer, but not yet a robust residual pruning method.**

Next step:

Rather than guessing priors, derive the split from an optimization objective over route-quality targets, then solve for the cheapest capacity allocation that matches those targets.

## Target-matched optimizer follow-up

A first optimizer-style route allocator searched route-family splits before fine-tuning. The objective matched magnitude's projection/readout route targets while preserving the plain-reserve main-path floor.

Fresh TinyResNet `99%` seeds `[211, 212, 213, 214]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `25.79%` | baseline | baseline |
| plain reserve `0.60` | `18.74%` | `-7.05` pts | `0/4` |
| tuned `40/35/25` | `24.43%` | `-1.36` pts | `0/4` |
| fixed deficit predictor | `23.59%` | `-2.21` pts | `0/4` |
| target-matched optimizer | `24.52%` | `-1.27` pts | `1/4` |

The optimizer selected `20/55/25` main/projection/readout on every seed. That improved over plain reserve but did not beat magnitude.

Interpretation:

- Matching projection/readout targets is not enough.
- The optimizer overconcentrated protected capacity into projection routes.
- It increased dead outputs relative to tuned/fixed predictors.
- The missing term is a route-diversity or overconcentration penalty.

Next optimization objective:

**Match projection and readout targets while penalizing route-family overconcentration and preserving main-path diversity.**

## Diversity-penalized optimizer

A follow-up optimizer added route-family concentration and projection-overuse penalties. This directly tested the degeneracy hypothesis: do not allow the allocator to satisfy one route target by collapsing route-family diversity.

Fresh TinyResNet `99%` seeds `[215, 216, 217, 218]`:

| Method | After FT | Delta vs magnitude | Wins vs magnitude |
|---|---:|---:|---:|
| magnitude | `24.80%` | baseline | baseline |
| plain reserve `0.60` | `20.13%` | `-4.67` pts | `0/4` |
| tuned `40/35/25` | `26.04%` | `+1.25` pts | `3/4` |
| diversity target optimizer | `25.85%` | `+1.05` pts | `4/4` |

The optimizer selected `25/50/25` main/projection/readout on every seed. It still emphasizes projection, but the diversity penalty prevents the previous `20/55/25` collapse and improves seed stability.

Interpretation:

- The degeneracy constraint worked.
- The optimizer now beats magnitude on every seed in this batch.
- It is slightly below the tuned split on mean, but has stronger paired consistency.
- This is the best current evidence that route-quality optimization can become a method, not just a diagnostic.

Remaining caveat:

The penalty weights are still hand-set. The next step is to derive them from route-family sensitivity or validate them on a stronger residual backbone.
