import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
R='results/wustl/'; F='figures/'
C=pd.read_csv(R+'expC_attackers.csv'); rounds=["round 0 (clean)","round 1","round 2"]; xl=["Round 0\n(clean)","Round 1","Round 2"]
DB,GR,LB,RD,OR='#1f4e79','#7f7f7f','#4f81bd','#c0504d','#e8a33d'
# Fig 5
fig,ax=plt.subplots(figsize=(6.2,3.9))
for e,c in zip((0.25,0.5,1.0),(LB,RD,DB)):
    g=C[(C.threshold=='validation-calibrated')&(C.attacker=='ES (training attacker)')&(C.eps==e)].groupby('round').evasion
    ax.errorbar(range(3),[g.mean()[r] for r in rounds],[g.std()[r] for r in rounds],marker='o',color=c,capsize=3,label=f'ε = {e} std')
ax.set_xticks(range(3)); ax.set_xticklabels(xl); ax.set_ylim(-0.03,1.1); ax.set_ylabel('Evasion rate (guided agent)'); ax.legend(frameon=False,loc='center left')
fig.tight_layout(); fig.savefig(F+'fig5.png',dpi=200)
# Fig 6
A=[("ES (training attacker)","ES (training attacker)",DB),("Random search","Random search",GR),("ES 5x budget, sigma=eps/3","ES 5x budget",LB),("PGD white-box","PGD white-box",RD),("PGD transfer (surrogate)","PGD transfer",OR)]
fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,e in zip(axs,(0.5,1.0)):
    for k,(a,lab,c) in enumerate(A):
        g=C[(C.threshold=='validation-calibrated')&(C.attacker==a)&(C.eps==e)].groupby('round').evasion
        ax.bar(np.arange(3)+(k-2)*0.16,[g.mean()[r] for r in rounds],0.16,yerr=[g.std()[r] for r in rounds],color=c,capsize=2.5,label=lab,error_kw=dict(lw=1))
    ax.set_xticks(range(3)); ax.set_xticklabels(xl); ax.set_title(f'ε = {e} std'); ax.set_ylim(0,1.15)
axs[0].set_ylabel('Evasion rate'); axs[0].legend(frameon=False,fontsize=8.5)
fig.tight_layout(); fig.savefig(F+'fig6.png',dpi=200)
# Fig 7
A9=pd.concat([pd.read_csv(R+'expA_early_stopping.csv'),pd.read_csv(R+'expA_local10.csv')]); r9=["round 0","round 1","round 2"]
S=[("central, 10 epochs","Centralized, 10 epochs",DB,'-'),("central, early stopping","Centralized, early stopping",LB,'--'),("central, 40 epochs","Centralized, 40 epochs",GR,'-'),
   ("local-only, 10 epochs","Local-only, 10 epochs",OR,'-'),("federated (FedAvg, E=1, R=40)","Federated (FedAvg, 40 rounds)",RD,'-')]
fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,e in zip(axs,(0.5,1.0)):
    for s,lab,c,ls in S:
        g=A9[A9.setting==s].groupby('round')[f'PGD@{e}']
        ax.errorbar(range(3),[g.mean()[r] for r in r9],[g.std()[r] for r in r9],marker='o',color=c,ls=ls,capsize=3,label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(xl); ax.set_title(f'White-box PGD, ε = {e} std'); ax.set_ylim(-0.03,1.05)
axs[0].set_ylabel('Evasion rate'); axs[0].legend(frameon=False,fontsize=8.5)
fig.tight_layout(); fig.savefig(F+'fig7.png',dpi=200)
