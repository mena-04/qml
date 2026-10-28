import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix

def evaluate(y, probs):
    pred = (np.asarray(probs)>=0.5).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=roc_auc_score(y,probs),recall=recall_score(y,pred,zero_division=0),precision=precision_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),specificity=tn/(tn+fp),false_negatives=int(fn),false_positives=int(fp))
