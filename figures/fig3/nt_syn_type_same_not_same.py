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
from config import SYNAPSE_TABLE_FTR, NEURON_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *


#%%
from sklearn.preprocessing import StandardScaler
#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
nodesG = pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
a=allsynapses.head(1000)
#%%
allsynapses=allsynapses.merge(nodesG[['neuron','primary_type']],left_on='pre',right_on='neuron',how='left')
#%%
allsynapses=allsynapses.merge(nodesG[['neuron','primary_type']],left_on='post',right_on='neuron',how='left')
#%%
allsynapses=allsynapses.merge(nodesG[['neuron','nt_type']],left_on='pre',right_on='neuron',how='left')

#%%
allsynapses=allsynapses.dropna(subset='nt_type')
#%%
allsynapses=allsynapses.query('SI_pre>=0.1 and SI_post>=0.1')
#%%

same=allsynapses[allsynapses['primary_type_x']==allsynapses['primary_type_y']]
notsame=allsynapses[allsynapses['primary_type_x']!=allsynapses['primary_type_y']]
#%%

custom_pallete = {
    "ACH": "#95A3CE",
    "GABA": "#D5A848",
    "GLUT": "#86A859",
    "SER": "#8C6295",
    "DA": "#B87969",
    "OCT": "#725C98",
}
#%%
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%%
main_nt = ['ACH', 'GABA', 'GLUT']

def plot_nt_pie(ax, df, comp_value, title, rotate_small=False):
    sub = df[(df['comp'] == comp_value) & (df['nt_type'].isin(main_nt))]
    
    counts = sub['nt_type'].value_counts().reindex(main_nt, fill_value=0)
    n = counts.sum()

    colors = [custom_pallete[k] for k in main_nt]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=main_nt,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        labeldistance=1.1,
        pctdistance=0.8,
        rotatelabels=True,
        textprops={'fontsize':13}
    )
    
    # rotate GABA and GLUT percentages only
    if rotate_small:
        autotexts[1].set_rotation(45)
        autotexts[2].set_rotation(45)

        # move GABA % slightly right
        x, y = autotexts[1].get_position()
        autotexts[1].set_position((x + 0.02, y))

    # title for every pie
    ax.set_title(f"{title}\n n = {n}", y=1.1, fontsize=14)



#%%
fig, axes = plt.subplots(2, 2, figsize=(13, 13))

plot_nt_pie(axes[0,0], same, 'DD', 'DD same-type', rotate_small=True)
plot_nt_pie(axes[0,1], same, 'AA', 'AA same-type')
plot_nt_pie(axes[1,0], notsame, 'DD', 'DD non-same-type')
plot_nt_pie(axes[1,1], notsame, 'AA', 'AA non-same-type')

plt.tight_layout()

_out_dir = OUTPUT_DIR / "fig3" / "nt_syn_type_same_not_same"
_out_dir.mkdir(parents=True, exist_ok=True)
save_path = _out_dir / 'nt_distribution_synapse_types.svg'
plt.savefig(save_path, format='svg', bbox_inches='tight')

plt.show()