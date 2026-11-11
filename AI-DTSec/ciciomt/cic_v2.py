# CICIoMT2024: three pre-registered improvements
#  (1) PGD-based hardening (PGD rounds, and Madry-style on-the-fly adversarial training)
#  (2) realistic constraint: protocol-composition and TCP-flag features frozen
#  (3) ES with annealed step size (same 400-query budget) to remove the search artefact at large epsilon
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cic_exp import *
OUT='results/ciciomt/v2'; os.makedirs(OUT,exist_ok=True)
FROZEN=['Protocol Type','HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC','TCP','UDP','DHCP','ARP','ICMP','IGMP','IPv','LLC',
        'fin_flag_number','syn_flag_number','rst_flag_number','psh_flag_number','ack_flag_number','ece_flag_number','cwr_flag_number']
MASK={0:torch.ones(Dm),1:torch.tensor([0. if c in FROZEN else 1. for c in feats])}
print('mutable features under constraint:',int(MASK[1].sum()),'of',Dm,flush=True)
def box(x,x0,eps,M):
    x=x0+((x-x0)*M).clamp(-eps,eps); return torch.minimum(torch.maximum(x,torch.minimum(lo,x0)),torch.maximum(hi,x0))
def es(fn,x0,eps,seed,M,gens=25,pop=16,sigma_div=2.0,anneal=False):
    g=torch.Generator().manual_seed(seed); S,Dd=x0.shape; bx,bs=x0.clone(),fn(x0)
    for t in range(gens):
        sig=(eps/sigma_div)*((0.05**(t/(gens-1))) if anneal else 1.0)
        c=box(bx[:,None,:]+torch.randn(S,pop,Dd,generator=g)*sig*M,x0[:,None,:],eps,M)
        sc=fn(c.reshape(-1,Dd)).reshape(S,pop); mn,ai=sc.min(1); u=mn<bs
        bs=torch.where(u,mn,bs); bx=torch.where(u[:,None],c[torch.arange(S),ai],bx)
    return bs,bx
def rnd(fn,x0,eps,seed,M,gens=25,pop=16):
    g=torch.Generator().manual_seed(seed); S,Dd=x0.shape; bx,bs=x0.clone(),fn(x0)
    for _ in range(gens):
        c=box(x0[:,None,:]+(torch.rand(S,pop,Dd,generator=g)*2-1)*eps,x0[:,None,:],eps,M)
        sc=fn(c.reshape(-1,Dd)).reshape(S,pop); mn,ai=sc.min(1); u=mn<bs; bs=torch.where(u,mn,bs); bx=torch.where(u[:,None],c[torch.arange(S),ai],bx)
    return bs,bx
def pgd(model,x0,eps,seed,M,steps=50,restarts=2,alpha=None):
    g=torch.Generator().manual_seed(seed); a=alpha or eps/10; bx=x0.clone(); bl=model(x0).squeeze(-1).detach()
    for r in range(restarts):
        x=box(x0+(torch.rand(x0.shape,generator=g)*2-1)*eps*float(r>0),x0,eps,M)
        for _ in range(steps):
            x=x.detach().requires_grad_(True); model(x).squeeze(-1).sum().backward()
            with torch.no_grad():
                x=box(x-a*x.grad.sign()*M,x0,eps,M); l=model(x).squeeze(-1); u=l<bl; bl=torch.where(u,l,bl); bx=torch.where(u[:,None],x,bx)
    return torch.sigmoid(bl),bx
# ---------------- Part A: autoencoder red team, constrained vs unconstrained, plain vs annealed ES ----------------
def part_a(nseeds=5,epochs=30):
    ben=np.where(ben_fit.values)[0]; rows=[]
    for s in range(nseeds):
        torch.manual_seed(s); m=ae_model(); opt=torch.optim.Adam(m.parameters(),1e-3); A=Ttr[cap(ben,60000,s)]
        for ep in range(epochs):
            p=torch.randperm(len(A))
            for i in range(0,len(A),256):
                b=A[p[i:i+256]]; opt.zero_grad(); ((m(b)-b)**2).mean().backward(); opt.step()
        m.eval(); sc=lambda T: ((m(T)-T)**2).mean(-1).detach()
        thr=float(np.percentile(sc(Ttr[val[ytr[val]==0]]).numpy(),95)); ste=sc(Tte).numpy()
        for g in TARGETS:
            ix=np.where(((TE.g==g).values)&(ste>thr))[0]
            if len(ix)<20: continue
            x0=Tte[cap(ix,400,s)]
            for cons in [0,1]:
                for eps in [0.1,0.25,0.5,1.0,2.0]:
                    for meth,fn in [('guided',lambda: es(sc,x0,eps,1000+s,MASK[cons])),('guided annealed',lambda: es(sc,x0,eps,1000+s,MASK[cons],anneal=True)),
                                    ('random',lambda: rnd(sc,x0,eps,1000+s,MASK[cons]))]:
                        rows.append(dict(seed=s,target=g,constrained=cons,eps=eps,method=meth,n=len(x0),evasion=float((fn()[0]<=thr).float().mean())))
        print('AE seed',s,'done',flush=True); pd.DataFrame(rows).to_csv(f'{OUT}/A_redteam.csv',index=False)
    R=pd.DataFrame(rows); print(R.groupby(['constrained','target','eps','method']).evasion.mean().unstack('method').round(3).to_string(),flush=True)
# ---------------- Part B: hardening variants ----------------
def train_madry(seed,M,eps_set=(0.25,0.5,1.0),steps=7,maxep=40):
    torch.manual_seed(seed); g=torch.Generator().manual_seed(9000+seed)
    m=nn.Sequential(nn.Linear(Dm,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    A=Ttr[fit]; t=torch.tensor(ytr[fit],dtype=torch.float32); opt=torch.optim.Adam(m.parameters(),1e-3); pw=torch.tensor(PW)
    bce=lambda X,y: nn.functional.binary_cross_entropy_with_logits(m(X).squeeze(1),y,pos_weight=pw)
    def adv(X,y,eps,gen,st):
        X=X.clone(); a=y==1; x0=X[a]
        if len(x0): X[a]=pgd(m,x0,eps,int(torch.randint(0,10**9,(1,),generator=gen)),M,steps=st,restarts=1,alpha=2.5*eps/st)[1]
        for p_ in m.parameters(): p_.grad=None
        return X.detach()
    Vx=Ttr[val]; Vy=torch.tensor(ytr[val],dtype=torch.float32); best=(1e9,None,0); bad=0
    for ep in range(maxep):
        p=torch.randperm(len(A)); m.train()
        for i in range(0,len(A),256):
            ix=p[i:i+256]; e=eps_set[int(torch.randint(0,len(eps_set),(1,),generator=g))]; Xa=adv(A[ix],t[ix],e,g,steps)
            opt.zero_grad(); (0.5*bce(A[ix],t[ix])+0.5*bce(Xa,t[ix])).backward(); opt.step()
        m.eval(); vg=torch.Generator().manual_seed(1)
        Va=adv(Vx,Vy,0.5,vg,10)
        with torch.no_grad(): l=float(0.5*bce(Vx,Vy)+0.5*bce(Va,Vy))      # robust validation loss (clean + PGD eps 0.5)
        if l<best[0]-1e-6: best=(l,{k:v.clone() for k,v in m.state_dict().items()},ep+1); bad=0
        else:
            bad+=1
            if bad>=5: break
        print('  madry ep',ep+1,'val',round(l,4),flush=True)
    m.load_state_dict(best[1]); m.eval(); print('  madry best epoch',best[2],flush=True)
    return m,(lambda T: torch.sigmoid(m(T).squeeze(-1)).detach())
def part_b(hard,cons):
    M=MASK[cons]; EPS=[0.25,0.5,1.0]; rows=[]; ops=[]; nor_te=Tte[yte==0]; tag=f'{hard}_c{cons}'
    for s in range(3):
        t0=time.time(); Xf,yf=Ztr[fit],ytr[fit]
        gen=np.sort(np.concatenate([cap(np.where((TR.part=='fit').values & (TR.g==g).values)[0],1000,s) for g in ['ARP_Spoofing','MQTT-Malformed_Data','Recon']]))
        m0,f0=train_clf(Xf,yf,s,'ES'); sur,_=train_clf(Xf,yf,s+100,'ES')
        if hard in ('ES','PGD'):
            mk=lambda m,f,base: np.concatenate([(es(f,Ttr[gen],e,base+s*10+k,M)[1] if hard=='ES' else pgd(m,Ttr[gen],e,base+s*10+k,M)[1]).detach().numpy() for k,e in enumerate(EPS)])
            adv=mk(m0,f0,200); m1,f1=train_clf(np.vstack([Xf,adv]),np.r_[yf,np.ones(len(adv),int)],s,'ES')
            adv2=mk(m1,f1,300); m2,f2=train_clf(np.vstack([Xf,adv,adv2]),np.r_[yf,np.ones(len(adv)+len(adv2),int)],s,'ES')
            models={'round 0 (clean)':(m0,f0),'round 1':(m1,f1),'round 2':(m2,f2)}
        else:
            mr,fr=train_madry(s,M); models={'round 0 (clean)':(m0,f0),'Madry':(mr,fr)}
        for rn,(m,fn) in models.items():
            thr=float(np.percentile(fn(nor_te).numpy(),95)); pte=fn(Tte).numpy()
            o=dict(hard=hard,constrained=cons,seed=s,round=rn,AUC=roc_auc_score(yte,pte),FPR=float((pte[yte==0]>thr).mean()))
            for g in TARGETS: o['rec_'+g]=float((pte[(TE.g==g).values]>thr).mean())
            ops.append(o)
            for g in ['ARP_Spoofing','MQTT-Malformed_Data','Recon']:
                ix=np.where(((TE.g==g).values)&(pte>thr))[0]
                if len(ix)<20: continue
                x0=Tte[cap(ix,400,s)]
                for eps in EPS:
                    res={'ES (training attacker)':es(fn,x0,eps,1000+s,M)[0],'ES 5x budget':es(fn,x0,eps,5000+s,M,gens=125,sigma_div=3.0)[0],
                         'ES annealed':es(fn,x0,eps,6000+s,M,anneal=True)[0],
                         'PGD white-box':pgd(m,x0,eps,7000+s,M)[0],'PGD transfer':fn(pgd(sur,x0,eps,8000+s,M)[1])}
                    for a,v in res.items(): rows.append(dict(hard=hard,constrained=cons,seed=s,round=rn,target=g,eps=eps,attacker=a,n=len(x0),evasion=float((v<=thr).float().mean())))
            print(tag,s,rn,{k:(round(v,3) if isinstance(v,float) else v) for k,v in o.items()},flush=True)
        print(tag,'seed',s,round(time.time()-t0),'s',flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/B_{tag}.csv',index=False); pd.DataFrame(ops).to_csv(f'{OUT}/B_ops_{tag}.csv',index=False)
    R=pd.DataFrame(rows); print(R.groupby(['target','attacker','eps','round']).evasion.mean().unstack('round').round(3).to_string(),flush=True)
if __name__=='__main__':
    t0=time.time()
    if sys.argv[1]=='A': part_a()
    else: part_b(sys.argv[1],int(sys.argv[2]))
    print('done',round(time.time()-t0),'s',flush=True)
