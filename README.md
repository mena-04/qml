# QBreast — Quantum-Enhanced Breast Cancer Classification

Exploratory classical vs quantum machine-learning research on the Wisconsin Diagnostic Breast Cancer dataset. The target is **malignant = 1**. This is a research demonstration, not a clinical diagnostic tool.

## Main method

A four-feature classical MLP receives an optional residual correction from either a small classical neural branch or a four-qubit, two-layer variational quantum circuit. The residual scale is fixed at 0.5. All three models use the same preprocessing and folds.

## Reproduce a small demonstration

```bash
pip install -r requirements.txt
pytest -q
python run_experiment.py --indices limited_data/data_n40_seed42.npz --samples 40
```

**Data prerequisite:** Copy the original saved training-index NPZ file into `limited_data/`. It is not included in this starter package; do not regenerate a different split and compare it as if it were the original.

The command runs 3-fold CV for one 40-sample subset. It is **not** the complete 40/80/160-sample multi-seed experiment and its output is not expected to reproduce the table below exactly.

## Previously reported Phase 8.10 results

| Training samples | Model | ROC-AUC | Malignant recall | F1 |
|---:|---|---:|---:|---:|
| 160 | Classical MLP | 0.9862 | 0.7722 | 0.8610 |
| 160 | Classical residual | 0.9867 | 0.8222 | 0.8882 |
| 160 | Quantum residual | 0.9881 | 0.8500 | 0.9125 |

These are exploratory repeated-CV results on overlapping training subsets. Gains are small and the quantum model is substantially slower in CPU simulation. No quantum advantage or clinical efficacy is established. Full Phase 8.10 CSVs are in `results/`.

## Research archive

Keep existing exploratory notebooks in the repository's `testing/` directory. The submitted modules are extracted from the uploaded Phase 8 notebook. Literature references, a clean-environment execution check, and the other phases' final artifacts still need verification.
