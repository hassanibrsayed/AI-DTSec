# CICIoMT2024 replication of the AI-DTSec red-team and hardening experiments (network features only)
import numpy as np, pandas as pd, torch, torch.nn as nn, glob, os, time, sys
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2); torch.use_deterministic_algorithms(True); dev='cpu'
D=os.environ.get("CIC_DIR", "data/CICIoMT2024"); OUT='results/ciciomt'; os.makedirs(OUT, exist_ok=True)
def load(split):
    fr=[]
    for f in sorted(glob.glob(f'{D}/{split}/*.csv')):
        c=os.path.basename(f).replace(f'_{split}.pcap.csv',''); x=pd.read_csv(f); x['cls']=c; fr.append(x)
    return pd.concat(fr,ignore_index=True)
TR=load('train'); TE=load('test')
feats=[c for c in TR.columns if c not in ('cls','Drate','IAT')]          # Drate constant; IAT is a capture-time field
key=lambda d: pd.util.hash_pandas_object(d[feats].round(9),index=False)
TE=TE[~key(TE).isin(set(key(TR))).values].reset_index(drop=True)        # drop test rows identical to a training row
grp=lambda c: 'Recon' if c.startswith('Recon') else c
TR['g']=TR.cls.map(grp); TE['g']=TE.cls.map(grp)
# fit/validation split inside each training file: last 20% of each class file is validation
TR['part']='fit'
for c,ix in TR.groupby('cls').groups.items():
    ix=np.array(sorted(ix)); TR.loc[ix[int(len(ix)*0.8):],'part']='val'
f=lambda d: np.log1p(d[feats].clip(lower=0).values.astype(np.float64))
Xtr,Xte=f(TR),f(TE); ben_fit=(TR.part=='fit')&(TR.cls=='Benign')
mu,sd=Xtr[ben_fit.values].mean(0),Xtr[ben_fit.values].std(0)+1e-6
Ztr=((Xtr-mu)/sd).astype(np.float32); Zte=((Xte-mu)/sd).astype(np.float32)
ytr=(TR.cls!='Benign').astype(int).values; yte=(TE.cls!='Benign').astype(int).values
fit=np.where(TR.part=='fit')[0]; val=np.where(TR.part=='val')[0]; Dm=Ztr.shape[1]
lo=torch.tensor(np.percentile(Ztr[ben_fit.values],0.5,axis=0)); hi=torch.tensor(np.percentile(Ztr[ben_fit.values],99.5,axis=0))
Tte=torch.tensor(Zte); Ttr=torch.tensor(Ztr)
print('features',Dm,'| train',len(TR),'| test after dedup',len(TE),dict(TE.g.value_counts()),flush=True)
TARGETS=['ARP_Spoofing','MQTT-Malformed_Data','Recon','MQTT-DoS-Connect_Flood']
def cap(ix,n,seed): 
    r=np.random.RandomState(seed); return np.sort(r.choice(ix,min(n,len(ix)),replace=False))
# ---------------- attacks ----------------
def attack_es(fn,x0,eps,seed,gens=25,pop=16,sigma_div=2.0):
    g=torch.Generator().manual_seed(seed); S,Dd=x0.shape; lo_s=torch.minimum(lo,x0); hi_s=torch.maximum(hi,x0); bx,bs=x0.clone(),fn(x0)
    for _ in range(gens):
        c=bx[:,None,:]+torch.randn(S,pop,Dd,generator=g)*(eps/sigma_div); c=x0[:,None,:]+(c-x0[:,None,:]).clamp(-eps,eps)
        c=torch.minimum(torch.maximum(c,lo_s[:,None,:]),hi_s[:,None,:]); sc=fn(c.reshape(-1,Dd)).reshape(S,pop); mn,ai=sc.min(1); u=mn<bs
        bs=torch.where(u,mn,bs); bx=torch.where(u[:,None],c[torch.arange(S),ai],bx)
    return bs,bx
def attack_random(fn,x0,eps,seed,gens=25,pop=16):
    g=torch.Generator().manual_seed(seed); S,Dd=x0.shape; lo_s=torch.minimum(lo,x0); hi_s=torch.maximum(hi,x0); bx,bs=x0.clone(),fn(x0)
    for _ in range(gens):
        c=x0[:,None,:]+(torch.rand(S,pop,Dd,generator=g)*2-1)*eps; c=torch.minimum(torch.maximum(c,lo_s[:,None,:]),hi_s[:,None,:])
        sc=fn(c.reshape(-1,Dd)).reshape(S,pop); mn,ai=sc.min(1); u=mn<bs; bs=torch.where(u,mn,bs); bx=torch.where(u[:,None],c[torch.arange(S),ai],bx)
    return bs,bx
def attack_pgd(model,x0,eps,seed,steps=50,restarts=2):
    g=torch.Generator().manual_seed(seed); lo_s=torch.minimum(lo,x0); hi_s=torch.maximum(hi,x0); a=eps/10; bx=x0.clone(); bl=model(x0).squeeze(-1).detach()
    for r in range(restarts):
        x=x0+(torch.rand(x0.shape,generator=g)*2-1)*eps*float(r>0); x=torch.minimum(torch.maximum(x,lo_s),hi_s)
        for _ in range(steps):
            x=x.detach().requires_grad_(True); model(x).squeeze(-1).sum().backward()
            with torch.no_grad():
                x=x-a*x.grad.sign(); x=x0+(x-x0).clamp(-eps,eps); x=torch.minimum(torch.maximum(x,lo_s),hi_s)
                l=model(x).squeeze(-1); u=l<bl; bl=torch.where(u,l,bl); bx=torch.where(u[:,None],x,bx)
    return torch.sigmoid(bl),bx
# ---------------- Part A: autoencoder baseline + red team (5 seeds) ----------------
def ae_model():
    return nn.Sequential(nn.Linear(Dm,24),nn.ReLU(),nn.Linear(24,12),nn.ReLU(),nn.Linear(12,6),nn.ReLU(),nn.Linear(6,12),nn.ReLU(),nn.Linear(12,24),nn.ReLU(),nn.Linear(24,Dm))
def part_a(nseeds=5,epochs=30):
    ben=np.where(ben_fit.values)[0]; rows=[]; base=[]
    for s in range(nseeds):
        torch.manual_seed(s); m=ae_model(); opt=torch.optim.Adam(m.parameters(),1e-3); A=Ttr[cap(ben,60000,s)]
        for ep in range(epochs):
            p=torch.randperm(len(A))
            for i in range(0,len(A),256):
                b=A[p[i:i+256]]; opt.zero_grad(); ((m(b)-b)**2).mean().backward(); opt.step()
        m.eval()
        sc=lambda T: ((m(T)-T)**2).mean(-1).detach()
        thr=float(np.percentile(sc(Ttr[val[ytr[val]==0]]).numpy(),95)); ste=sc(Tte).numpy()
        r=dict(seed=s,AUC=roc_auc_score(yte,ste),FPR=float((ste[yte==0]>thr).mean()))
        for g in TARGETS:
            mk=(TE.g==g).values; r['rec_'+g]=float((ste[mk]>thr).mean())
            r['auc_'+g]=roc_auc_score(np.r_[np.zeros((yte==0).sum()),np.ones(mk.sum())],np.r_[ste[yte==0],ste[mk]])
        base.append(r); print('AE',s,{k:round(v,3) for k,v in r.items()},flush=True)
        for g in TARGETS:
            ix=np.where(((TE.g==g).values)&(ste>thr))[0]
            if len(ix)<20: continue
            ix=cap(ix,400,s); x0=Tte[ix]
            for eps in [0.1,0.25,0.5,1.0,2.0]:
                for meth,fn in [('guided',attack_es),('random',attack_random)]:
                    bs,_=fn(sc,x0,eps,1000+s); rows.append(dict(seed=s,target=g,eps=eps,method=meth,n=len(ix),evasion=float((bs<=thr).float().mean())))
        pd.DataFrame(base).to_csv(f'{OUT}/cic_baseline.csv',index=False); pd.DataFrame(rows).to_csv(f'{OUT}/cic_redteam.csv',index=False)
    R=pd.DataFrame(rows); print(R.groupby(['target','eps','method']).evasion.agg(['mean','std']).unstack('method').round(3).to_string(),flush=True)
# ---------------- Part B: supervised detector, hardening and independent attackers (3 seeds) ----------------
PW=float((1-ytr[fit].mean())/ytr[fit].mean())
def train_clf(Xt,yt,seed,epochs):
    torch.manual_seed(seed); m=nn.Sequential(nn.Linear(Dm,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    A=torch.tensor(Xt); t=torch.tensor(yt,dtype=torch.float32); opt=torch.optim.Adam(m.parameters(),1e-3); pw=torch.tensor(PW)
    Vx=Ttr[val]; Vy=torch.tensor(ytr[val],dtype=torch.float32); best=(1e9,None,0); bad=0
    for ep in range(40 if epochs=='ES' else epochs):
        p=torch.randperm(len(A)); m.train()
        for i in range(0,len(A),256):
            ix=p[i:i+256]; opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(m(A[ix]).squeeze(1),t[ix],pos_weight=pw).backward(); opt.step()
        if epochs=='ES':
            m.eval()
            with torch.no_grad(): l=float(nn.functional.binary_cross_entropy_with_logits(m(Vx).squeeze(1),Vy,pos_weight=pw))
            if l<best[0]-1e-6: best=(l,{k:v.clone() for k,v in m.state_dict().items()},ep+1); bad=0
            else:
                bad+=1
                if bad>=5: break
    if epochs=='ES': m.load_state_dict(best[1])
    m.eval(); return m,(lambda T: torch.sigmoid(m(T).squeeze(-1)).detach())
def part_b(budget):
    EPS=[0.25,0.5,1.0]; rows=[]; ops=[]
    tgt_fit=np.where((TR.part=='fit').values & TR.g.isin(TARGETS).values)[0]; nor_te=Tte[yte==0]
    for s in range(3):
        gen=np.sort(np.concatenate([cap(np.where((TR.part=='fit').values & (TR.g==g).values)[0],1000,s) for g in ['ARP_Spoofing','MQTT-Malformed_Data','Recon']])); Xf,yf=Ztr[fit],ytr[fit]
        m0,f0=train_clf(Xf,yf,s,budget)
        adv=np.concatenate([attack_es(f0,Ttr[gen],e,200+s*10+k)[1].numpy() for k,e in enumerate(EPS)])
        m1,f1=train_clf(np.vstack([Xf,adv]),np.r_[yf,np.ones(len(adv),int)],s,budget)
        adv2=np.concatenate([attack_es(f1,Ttr[gen],e,300+s*10+k)[1].numpy() for k,e in enumerate(EPS)])
        m2,f2=train_clf(np.vstack([Xf,adv,adv2]),np.r_[yf,np.ones(len(adv)+len(adv2),int)],s,budget)
        sur,_=train_clf(Xf,yf,s+100,budget)
        for rn,(m,fn) in {'round 0 (clean)':(m0,f0),'round 1':(m1,f1),'round 2':(m2,f2)}.items():
            thr=float(np.percentile(fn(nor_te).numpy(),95)); pte=fn(Tte).numpy()
            o=dict(budget=str(budget),seed=s,round=rn,AUC=roc_auc_score(yte,pte),FPR=float((pte[yte==0]>thr).mean()))
            for g in TARGETS: o['rec_'+g]=float((pte[(TE.g==g).values]>thr).mean())
            ops.append(o)
            for g in ['ARP_Spoofing','MQTT-Malformed_Data','Recon']:
                ix=np.where(((TE.g==g).values)&(pte>thr))[0]
                if len(ix)<20: continue
                x0=Tte[cap(ix,400,s)]
                for eps in EPS:
                    res={'ES (training attacker)':attack_es(fn,x0,eps,1000+s)[0],'ES 5x budget':attack_es(fn,x0,eps,5000+s,gens=125,sigma_div=3.0)[0],
                         'PGD white-box':attack_pgd(m,x0,eps,7000+s)[0],'PGD transfer':fn(attack_pgd(sur,x0,eps,8000+s)[1])}
                    for a,v in res.items(): rows.append(dict(budget=str(budget),seed=s,round=rn,target=g,eps=eps,attacker=a,evasion=float((v<=thr).float().mean())))
            print(budget,s,rn,{k:(round(v,3) if isinstance(v,float) else v) for k,v in o.items()},flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/cic_hardening_{budget}.csv',index=False); pd.DataFrame(ops).to_csv(f'{OUT}/cic_hardening_ops_{budget}.csv',index=False)
    R=pd.DataFrame(rows); print(R.groupby(['target','attacker','eps','round']).evasion.mean().unstack('round').round(3).to_string(),flush=True)
if __name__=='__main__':
    t0=time.time(); part=sys.argv[1]
    if part=='A': part_a()
    else: part_b('ES' if part=='B_ES' else 40)
    print('done',round(time.time()-t0),'s')
