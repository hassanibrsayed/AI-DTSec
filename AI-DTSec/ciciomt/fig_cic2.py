import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9.5,'axes.spines.top':False,'axes.spines.right':False})
T=pd.read_csv('results/ciciomt/v2/summary_hardening.csv')
M=['No hardening','ES rounds (2)','PGD rounds (2)','Madry (on-the-fly PGD)']; XL=['None','Agent samples,\n2 rounds','PGD samples,\n2 rounds','On-the-fly\nPGD']
B=[('rec','rec_sd','Clean recall','#bfbfbf'),('e2e0.25','e2e0.25_sd','Detected after attack, ε = 0.25','#4f81bd'),('e2e0.5','e2e0.5_sd','Detected after attack, ε = 0.5','#c0504d')]
fig,axs=plt.subplots(2,2,figsize=(10,6.2),sharey=True)
for i,(c,cn) in enumerate([(0,'all 43 features mutable'),(1,'protocol and flag features frozen')]):
    for j,(t,tn) in enumerate([('ARP_Spoofing','ARP spoofing'),('MQTT-Malformed_Data','Malformed MQTT data')]):
        ax=axs[i,j]; X=T[(T.cons==c)&(T.target==t)].set_index('method').loc[M]
        for k,(m,s,l,col) in enumerate(B):
            ax.bar(np.arange(4)+(k-1)*0.26,X[m],0.26,yerr=X[s],color=col,capsize=2.5,label=l,error_kw=dict(lw=1))
        ax.set_xticks(range(4)); ax.set_xticklabels(XL,fontsize=8.5); ax.set_ylim(0,1.0)
        ax.set_title(f'({"abcd"[2*i+j]}) {tn}, {cn}',fontsize=9.5)
    axs[i,0].set_ylabel('Share of attack windows')
axs[0,1].legend(frameon=False,fontsize=8.5,loc='upper right')
fig.tight_layout(); fig.savefig('figures/fig_cic2.png',dpi=200)
