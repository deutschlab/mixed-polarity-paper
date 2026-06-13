import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import os
import time
import pickle
import navis 
import sys
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
#%%
nodesG=nodesG.dropna(subset=['primary_type','axon_correct','dend_correct'])
#%%



#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
syn_df=syn_df[syn_df['comp'].isin(['AD','AA','DD','DA'])]
#%%
syn_df=syn_df.merge(nodesG[['neuron','super_class']],how='left',left_on='pre',right_on='neuron')
#%%
syn_df=syn_df.merge(nodesG[['neuron','super_class']],how='left',left_on='post',right_on='neuron')
#%%
aa=syn_df.head(1000)
#%%
syn_df=syn_df.dropna(subset=['neuron_x','neuron_y'])
#%%
aa=syn_df.query('SI_pre>=0.1 and SI_post>=0.1')

#%%

r_list=[]
for i in range(0,100,5):
    i_t=i/100
    mask=syn_df.query("SI_pre>= @i_t and SI_post>= @i_t")
    r=mask.groupby(by='comp').size()
    r2=r/r.sum()
    r2=r2.reset_index()
    r3=round(r2[r2['comp']=='AD'][0].values[0],3)
    print(i,r3)
    r_list.append([i,r3])
#%%
df_plot = pd.DataFrame(r_list, columns=['threshold', 'AD_ratio'])

plt.figure(figsize=(8,5))
plt.plot(df_plot['threshold']/100, df_plot['AD_ratio'], marker='o')

plt.xlabel("SI th")
plt.ylabel("AD proportion")
plt.title("AD proportion vs SI threshold")
plt.grid(True)

plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig3" / "supp_ad_content_si"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "AD_SI_content.svg")
plt.show()
#%%

df_plot = pd.DataFrame(r_list, columns=['threshold', 'AD_ratio'])

x = df_plot['threshold'] / 100
y = df_plot['AD_ratio']

plt.figure(figsize=(8,5))

# dashed part: x <= 0.1
mask = x <= 0.1
plt.plot(x[mask], y[mask], linestyle='--', marker='o',color='grey')

# solid part: start from the last dashed point
idx = mask[mask].index[-1]
plt.plot(x.loc[idx:], y.loc[idx:], marker='o',color='blue')
plt.xlim(0,0.7)
plt.ylim(0.4,1)

plt.xlabel("SI th")
plt.ylabel("AD proportion")
plt.title("AD proportion vs SI threshold")
plt.grid(False)
sns.despine(top=True,right=True)
plt.tight_layout()
plt.savefig(_out_dir / "AD_SI_content.svg")
plt.show()

#%%



syn_df2=syn_df.query('SI_post>=0.1 and SI_pre>=0.1')
#%%
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}
# count values
counts = syn_df2['comp'].value_counts().reindex(custom_palette.keys())

# colors in correct order
colors = [custom_palette[k] for k in counts.index]

plt.figure(figsize=(5,5))
plt.pie(
    counts,
    labels=counts.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90
)
plt.axis('equal')  # keep circle

plt.tight_layout()
plt.savefig(_out_dir / "syn_type_pie.svg")
plt.show()
