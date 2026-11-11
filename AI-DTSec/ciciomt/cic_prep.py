import numpy as np, pandas as pd, glob, os
D=os.environ.get("CIC_DIR", "data/CICIoMT2024")
def load(split):
    fr=[]
    for f in sorted(glob.glob(f'{D}/{split}/*.csv')):
        c=os.path.basename(f).replace(f'_{split}.pcap.csv','')
        x=pd.read_csv(f); x['cls']=c; fr.append(x)
    return pd.concat(fr,ignore_index=True)
tr=load('train'); te=load('test'); feats=[c for c in tr.columns if c!='cls']
print(tr.cls.value_counts().to_dict()); print(te.cls.value_counts().to_dict())
print('min values <0:',[c for c in feats if (tr[c]<0).any()])
const=[c for c in feats if tr[c].nunique()<=1]; print('constant',const)
binary=[c for c in feats if set(np.unique(tr[c].dropna()))<= {0.0,1.0} and c not in const]; print('binary',len(binary),binary)
# duplicates of test rows in train
key=lambda d: pd.util.hash_pandas_object(d[feats].round(9),index=False)
htr=set(key(tr)); hte=key(te); dup=hte.isin(htr).values
print('test rows identical to a train row:',dup.sum(),'of',len(te)); print(te[dup].cls.value_counts().to_dict())
from sklearn.metrics import roc_auc_score
y=(te.cls!='Benign').astype(int)
for c in ['IAT','Number','Weight','Magnitue','Tot size','Rate','Header_Length','Protocol Type','Duration']:
    a=roc_auc_score(y,te[c]); print(c, round(max(a,1-a),3))
print(tr.groupby('cls').IAT.describe()[['mean','50%','min','max']])
