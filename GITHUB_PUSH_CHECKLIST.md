# GitHub Push Checklist

Use this checklist before publishing or opening a PR.

## Required evidence path

The strongest claim is the SynFlow dense-bridge collapse finding. These files should be present in the commit:

- `README.md`
- `experiments/04_criticality_pruning/README.md`
- `experiments/04_criticality_pruning/SYNFLOW_PATHOLOGY_SYNTHESIS.md`
- `experiments/04_criticality_pruning/CIFAR10_CNN_SYNFLOW_PATHOLOGY.md`
- `experiments/04_criticality_pruning/audit_synflow_pathology.py`
- `experiments/04_criticality_pruning/synthesize_synflow_pathology.py`
- `results/04_criticality_pruning/synflow_pathology_synthesis.json`
- `results/04_criticality_pruning/cifar10_cnn_synflow_pathology.json`
- `shared/pruning_diagnostics.py`
- `.github/workflows/research-audit.yml`

## Local audit

Run:

```powershell
python experiments\04_criticality_pruning\synthesize_synflow_pathology.py
python experiments\04_criticality_pruning\audit_synflow_pathology.py
```

Expected audit output:

```json
{
  "status": "ok",
  "cases": 3,
  "global_synflow_after_delta_mean": -0.4279583333333332
}
```

## What not to commit

Do not commit raw datasets or local caches:

- `data/`
- `__pycache__/`
- `.venv/`
- model checkpoints
- downloaded archives

`.gitignore` is configured for these.

## Suggested staging command

```powershell
git add README.md .gitignore .gitattributes .github GITHUB_PUSH_CHECKLIST.md QUICK_WINS.md run_all_pilots.py shared experiments results
```

Then inspect:

```powershell
git status --short
git diff --cached --stat
```

## Suggested commit message

```text
Document SynFlow dense-bridge collapse experiments
```

## Claim to use publicly

Global SynFlow can catastrophically starve dense classifier bridges in CNNs at severe global sparsity. In the checked synthesis, global SynFlow allocated zero `fc1` weights in `3/3` severe-sparsity CNN cases and averaged `-42.80` after-fine-tuning points versus magnitude pruning.
