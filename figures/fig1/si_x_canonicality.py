import sys
from pathlib import Path
# Make repo root importable so config.py can be found from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import pickle

from scipy.stats import gaussian_kde
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
#allsynapses=allsynapses.head(1000000)
nodesG=pd.read_feather(NEURON_TABLE_FTR)

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])

nodesG['correct_pos']=  ((nodesG['pre_A']  +  nodesG['post_D']    )/(nodesG['pre_A']  +  nodesG['post_A']+nodesG['pre_D']  +  nodesG['post_D']  ) )
df=nodesG[['neuron','super_class','SI','correct_pos']]
df_comb=df[['correct_pos','SI']].sort_values(by='SI').dropna().reset_index(drop=True)

df_comb=df[['correct_pos','SI']].sort_values(by='SI').dropna()

df_comb['SI']*=10000
df_comb_g=df_comb.groupby(by='SI')['correct_pos'].mean().reset_index()
df_comb_g['SI']=df_comb_g['SI']/10000

df_comb_g['rolling_correct'] = df_comb_g['correct_pos'].rolling(window=400).mean()
df_comb_g['rolling_correct']=df_comb_g['rolling_correct']*100


#%% FIG 1: Adult canonical percent (rolling mean) vs SI

fig, ax = plt.subplots(figsize=(2, 1.5))

sns.lineplot(
    data=df_comb_g,
    x='SI',
    y='rolling_correct',
    label='Adult (Rolling 400)',
    color='black',
    lw=0.3
)

ax.set_xlabel('Segregation Index (SI)')
ax.set_ylabel('Canonical percent')
ax.set_ylim(50, 100)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

# Text annotation for sample size
ax.text(0.05, 0.95, f'n = {len(df_comb)}', transform=ax.transAxes,
        fontsize=4, verticalalignment='top', color='black')

ax.legend(fontsize=5, loc='lower right', frameon=False)
sns.despine(ax=ax)

plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig1" / "si_x_canonicality"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    _out_dir / "canonical_vs_SI_adult.svg",
    format='svg'
)
plt.show()
