import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
R='results/wustl/'; F='figures/'
B=pd.read_csv(R+'table2_5seeds.csv'); DB,LB,RD,GR='#1f4e79','#4f81bd','#c0504d','#7f7f7f'
fig,ax=plt.subplots(figsize=(5.3,3.5)); x=np.arange(2); w=0.34
rec=[B.rec_alt,B.rec_spo]; auc=[B.auc_alt,B.auc_spo]
ax.bar(x-w/2,[r.mean() for r in rec],w,yerr=[r.std() for r in rec],color=DB,capsize=3,label='Recall at 5% FPR threshold')
ax.bar(x+w/2,[a.mean() for a in auc],w,yerr=[a.std() for a in auc],color=LB,capsize=3,label='AUC-ROC')
for i in range(2):
    ax.text(i-w/2,rec[i].mean()+rec[i].std()+0.02,f'{rec[i].mean():.3f}',ha='center',fontsize=8.5)
    ax.text(i+w/2,auc[i].mean()+auc[i].std()+0.02,f'{auc[i].mean():.3f}',ha='center',fontsize=8.5)
ax.axhline(0.5,ls='--',color=GR,lw=1); ax.text(1.45,0.515,'chance AUC',fontsize=8,color=GR,ha='right')
ax.set_xticks(x); ax.set_xticklabels(['Data alteration\n(n = 264)','Spoofing\n(n = 188)']); ax.set_ylim(0,1.18); ax.legend(frameon=False,fontsize=8.5,loc='upper right')
fig.tight_layout(); fig.savefig(F+'fig3.png',dpi=200)
T=pd.read_csv(R+'table3_5detseeds.csv')
fig,axs=plt.subplots(1,3,figsize=(10,3.3),sharey=True)
for ax,(g,tt) in zip(axs,[('all (34)','All 34 features'),('network (26)','Network only (26)'),('biometric (8)','Biometric only (8)')]):
    for m,lab,c,mk,ls in [('guided','Guided (1+λ) ES',RD,'o','-'),('random','Random search',GR,'s','--')]:
        s=T[(T.group==g)&(T.method==m)].groupby('eps').evasion_rate
        ax.errorbar(range(5),s.mean().values,s.std().values,color=c,marker=mk,ls=ls,capsize=3,label=lab)
    ax.set_xticks(range(5)); ax.set_xticklabels(['0.1','0.25','0.5','1.0','2.0']); ax.set_title(tt); ax.set_xlabel('Perturbation bound ε (std)'); ax.set_ylim(0,1.02)
axs[0].set_ylabel('Evasion rate'); axs[0].legend(frameon=False,fontsize=8.5,loc='upper left')
fig.tight_layout(); fig.savefig(F+'fig4.png',dpi=200)
