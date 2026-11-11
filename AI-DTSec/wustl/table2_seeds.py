import os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
df=pd.read_csv(os.environ.get("WUSTL_CSV", "data/wustl-ehms-2020.csv")); df.columns=[c.strip() for c in df.columns]
y=df["Label"].astype(int).values; cat=df["Attack Category"].astype(str).str.lower().values
rows=[]
for s in range(5):
    b=np.load(f'results/wustl/base{s}/baseline.npz',allow_pickle=True)
    mu,sd,thr,te,cols=b['mu'],b['sd'],float(b['thr']),b['te'],list(b['cols'])
    X=df[cols].apply(pd.to_numeric,errors='coerce').fillna(0).values.astype(np.float32); Z=((X-mu)/sd).astype(np.float32)
    ae=nn.Sequential(nn.Linear(34,24),nn.ReLU(),nn.Linear(24,12),nn.ReLU(),nn.Linear(12,6),nn.ReLU(),nn.Linear(6,12),nn.ReLU(),nn.Linear(12,24),nn.ReLU(),nn.Linear(24,34))
    ae.load_state_dict(torch.load(f'results/wustl/base{s}/ae_state.pt')); ae.eval()
    with torch.no_grad(): t=torch.tensor(Z[te]); sc=((ae(t)-t)**2).mean(1).numpy()
    yt=y[te]; ct=cat[te]; r=dict(seed=s,thr=thr,AUC=roc_auc_score(yt,sc),FPR=(sc[yt==0]>thr).mean())
    for c,k in (('data alteration','alt'),('spoofing','spo')):
        m=ct==c; r['rec_'+k]=(sc[m]>thr).mean(); r['auc_'+k]=roc_auc_score(np.r_[np.zeros((yt==0).sum()),np.ones(m.sum())],np.r_[sc[yt==0],sc[m]]); r['n_det_'+k]=int((sc[m]>thr).sum())
    rows.append(r)
R=pd.DataFrame(rows); R.to_csv('results/wustl/table2_5seeds.csv',index=False); print(R.round(3)); print(R.agg(['mean','std']).round(3).T)
