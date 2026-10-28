"""Small reproducible Phase 8 demonstration; NOT the full reported multi-seed benchmark."""
import argparse, random, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from src.data import load_data, make_preprocessor, load_subset_indices
from src.residual import MODEL_TYPES, ResidualClassifier
from src.evaluation import evaluate

def run(args):
    X, y = load_data()
    indices = load_subset_indices(args.indices)
    if len(indices) != args.samples:
        raise ValueError(f"Expected {args.samples} indices; got {len(indices)}")
    X, y = X[indices], y[indices]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        prep = make_preprocessor()
        Xtr = torch.tensor(prep.fit_transform(X[tr]), dtype=torch.float64)
        Xval = torch.tensor(prep.transform(X[val]), dtype=torch.float64)
        ytr = torch.tensor(y[tr], dtype=torch.float64)
        for name in args.models:
            seed = args.seed + fold
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = ResidualClassifier(MODEL_TYPES[name]).double()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
            criterion = torch.nn.BCEWithLogitsLoss()
            start = time.perf_counter()
            for _ in range(args.epochs):
                model.train(); optimizer.zero_grad()
                loss = criterion(model(Xtr), ytr)
                loss.backward(); optimizer.step()
            elapsed = time.perf_counter() - start
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(Xval)).numpy()
            rows.append(dict(model=name, fold=fold, training_seconds=elapsed, **evaluate(y[val], probs)))
            print(f"Fold {fold}: {name} completed")
    df = pd.DataFrame(rows)
    print(df.groupby('model')[['roc_auc', 'recall', 'f1', 'specificity']].mean().round(4))
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
    return df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--indices', required=True, help='Existing limited_data/data_n40_seed42.npz')
    p.add_argument('--samples', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--models', nargs='+', choices=list(MODEL_TYPES), default=['Classical MLP', 'Classical Residual', 'Quantum Residual'])
    p.add_argument('--output', default='results/demo_run.csv')
    run(p.parse_args())
