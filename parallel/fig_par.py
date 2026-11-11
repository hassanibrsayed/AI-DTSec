import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
R=pd.read_csv('results/parallel/parallel_redteam_pcores.csv'); R=R[R.workers>0]
f=0.04527774; am=lambda n: 1/(f+(1-f)/n)
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
ax=axs[0]; n=np.linspace(1,24,200)
ax.plot([1,24],[1,24],':',color='#7f7f7f',label='Ideal (linear)')
ax.plot(n,am(n),'--',color='#4f81bd',label='Amdahl fit to 2 and 4 workers (f = 4.5%)')
ax.plot(R.workers,R.speedup,'o-',color='#c0504d',label='Measured')
for x,lab in [(8,'8 performance cores'),(16,'16 logical CPUs')]:
    ax.axvline(x,color='#bbbbbb',lw=0.8); ax.text(x+0.3,15.5,lab,fontsize=8,color='#555555',rotation=90,va='top')
ax.set_xlabel('Workers'); ax.set_ylabel('Speedup'); ax.set_xticks([1,2,4,8,12,16,24]); ax.set_ylim(0,16); ax.legend(frameon=False,fontsize=8.5,loc='lower right')
ax.set_title('(a) Speedup')
ax=axs[1]; ax.bar(range(len(R)),R.efficiency*100,color='#1f4e79'); ax.axhline(100,ls=':',color='#7f7f7f')
ax.set_xticks(range(len(R))); ax.set_xticklabels(R.workers); ax.set_xlabel('Workers'); ax.set_ylabel('Efficiency (%)'); ax.set_title('(b) Parallel efficiency')
for i,v in enumerate(R.efficiency*100): ax.text(i,v+2,f'{v:.0f}',ha='center',fontsize=8)
fig.tight_layout(); fig.savefig('figures/fig_par.png',dpi=200)
