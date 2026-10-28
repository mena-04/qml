"""Small 3-fold Phase 8 demo, not the full multi-seed benchmark."""
import argparse, random, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from src.data import load_data, make_preprocessor, load_subset_indices
from src.residual import MODEL_TYPES, ResidualClassifier
from src.evaluation import evaluate

def run(args):
    X,y=load_data()
    idx=load_subset_indices(args.indices)
    if len(idx)!=args.samples: raise ValueError('Subset size mismatch')
    X,y=X[idx],y[idx]
    rows=[]
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=args.seed)
    for fold,(tr,va) in enumerate(cv.split(X,y)):
        prep=make_preprocessor()
        Xtr=torch.tensor(prep.fit_transform(X[tr]),dtype=torch.float64)
        Xva=torch.tensor(prep.transform(X[va]),dtype=torch.float64)
        ytr=torch.tensor(y[tr],dtype=torch.float64)
        for name,branch in MODEL_TYPES.items():
            seed=args.seed+fold
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=ResidualClassifier(branch).double()
            opt=torch.optim.Adam(model.parameters(),lr=0.03)
            criterion=torch.nn.BCEWithLogitsLoss()
            t=time.perf_counter()
            for _ in range(args.epochs):
                model.train();opt.zero_grad()
                loss=criterion(model(Xtr),ytr);loss.backward();opt.step()
            elapsed=time.perf_counter()-t
            model.eval()
            with torch.no_grad(): probs=torch.sigmoid(model(Xva)).numpy()
            rows.append(dict(fold=fold,model=name,training_seconds=elapsed,**evaluate(y[va],probs)))
            print(f'Completed fold {fold}: {name}')
    df=pd.DataFrame(rows)
    print(df.groupby('model')[['roc_auc','recall','f1','specificity']].mean().round(4))
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output,index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--indices',required=True)
    p.add_argument('--samples',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--output',default='demo_run.csv')
    run(p.parse_args())
