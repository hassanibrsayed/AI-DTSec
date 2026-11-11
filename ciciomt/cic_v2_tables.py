import pandas as pd,glob,numpy as np,json
D='results/ciciomt/v2'
R=pd.concat([pd.read_csv(f) for f in glob.glob(f'{D}/B_[EPM]*_c*.csv')]);O=pd.concat([pd.read_csv(f) for f in glob.glob(f'{D}/B_ops_*.csv')])
def lab(h,r): return 'No hardening' if r=='round 0 (clean)' else {'ES':'ES rounds (2)','PGD':'PGD rounds (2)','MADRY':'Madry (on-the-fly PGD)'}[h]
keep=lambda X: X[((X['round']=='round 0 (clean)')&(X.hard=='ES'))|X['round'].isin(['round 2','Madry'])].copy()
R=keep(R);O=keep(O);R['m']=[lab(h,r) for h,r in zip(R.hard,R['round'])];O['m']=[lab(h,r) for h,r in zip(O.hard,O['round'])]
out=[]
for c in [0,1]:
  for t in ['ARP_Spoofing','MQTT-Malformed_Data']:
    for m in ['No hardening','ES rounds (2)','PGD rounds (2)','Madry (on-the-fly PGD)']:
      o=O[(O.constrained==c)&(O.m==m)].set_index('seed'); r=R[(R.constrained==c)&(R.m==m)&(R.target==t)]
      d=dict(cons=c,target=t,method=m,rec=o['rec_'+t].mean(),rec_sd=o['rec_'+t].std(),AUC=o.AUC.mean())
      for e in [0.25,0.5,1.0]:
        x=r[r.eps==e]; w=x.groupby('seed').evasion.max(); e2e=o['rec_'+t]*(1-w)
        d[f'wb{e}']=x[x.attacker=='PGD white-box'].evasion.mean(); d[f'tr{e}']=x[x.attacker=='PGD transfer'].evasion.mean()
        d[f'es5{e}']=x[x.attacker=='ES 5x budget'].evasion.mean()
        d[f'worst{e}']=w.mean(); d[f'e2e{e}']=e2e.mean(); d[f'e2e{e}_sd']=e2e.std()
      out.append(d)
T=pd.DataFrame(out); T.to_csv(f'{D}/summary_hardening.csv',index=False)
pd.set_option('display.width',250); print(T.round(3).to_string())
A=pd.read_csv(f'{D}/A_redteam.csv'); S=A.groupby(['constrained','target','eps','method']).evasion.agg(['mean','std']).reset_index(); S.to_csv(f'{D}/summary_redteam.csv',index=False)
print(S[S.target.isin(['ARP_Spoofing','MQTT-Malformed_Data'])&S.eps.isin([0.5,1.0,2.0])].round(3).to_string())
print(A.groupby(['constrained','target']).n.mean())
