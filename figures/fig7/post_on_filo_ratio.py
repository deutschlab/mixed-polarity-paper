# Fig 7H: mean fraction of incoming synapses that sit on a filopodium, axon vs dendrite.
# Ported from the authors' original script Fig 7/filapodia_extraction/t/post_on_filo_ratio.py.
# Changes: paths via config.py, unused imports removed, a duplicated cell removed.
# The original copy read "syn_bouton_filopodia(new).ftr"; SYN_BOUTON_FTR names the file without "(new)".
# Also added: svg.fonttype='none' and Arial (repository convention). The original saved SVGs drew text as
# DejaVu Sans glyph paths, so text and spacing differ from his files; plotted values do not.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, SYN_BOUTON_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "fig7" / "post_on_filo_ratio"
_out_dir.mkdir(parents=True, exist_ok=True)
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
df=pd.read_feather(SYN_BOUTON_FTR)
#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
#%%
allsynapses=allsynapses.merge(df,on='synapse_id',how='left')
#%%
allsynapses=allsynapses[['post','comp','filopodia_on_post','SI_post']]
#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
#%%
allsynapses=allsynapses.merge(nodesG[['neuron','super_class']],left_on='post',right_on='neuron', how='left')
#%%
allsynapses=allsynapses.dropna(subset='super_class')
#%%

allsynapses['comp2'] = allsynapses['comp'].astype(str).str[1]
#%%

df2 = allsynapses[allsynapses['comp2'].isin(['A', 'D'])].copy()
#%%
df2=df2.query('SI_post>=0.1')
#%%


mean_filo = (
    df2.groupby('comp2', as_index=False)['filopodia_on_post']
    .mean()
    .rename(columns={'comp2': 'post_comp', 'filopodia_on_post': 'mean_filopodia_on_post'})
)

# make sure order is A then D
mean_filo['post_comp'] = pd.Categorical(mean_filo['post_comp'], categories=['A', 'D'], ordered=True)
mean_filo = mean_filo.sort_values('post_comp')

print(mean_filo)

# 3) bar plot
plt.figure(figsize=(4, 4))
plt.bar(mean_filo['post_comp'], mean_filo['mean_filopodia_on_post'])
plt.xlabel('Post compartment (2nd letter of comp)')
plt.ylabel('Mean filopodia_on_post')
plt.title('Mean filopodia_on_post by post compartment')
plt.tight_layout()
plt.savefig(_out_dir / "axon_dend_filo_ratio.svg")
plt.show()
#%%
# Mean filopodia-on-post by super_class and compartment (A/D)
mean_filo_sc = (
    df2.groupby(['super_class', 'comp2'], as_index=False)['filopodia_on_post']
    .mean()
    .rename(columns={'comp2': 'post_comp', 'filopodia_on_post': 'mean_filopodia_on_post'})
)

# Order compartments A then D
mean_filo_sc['post_comp'] = pd.Categorical(mean_filo_sc['post_comp'], categories=['A', 'D'], ordered=True)
mean_filo_sc = mean_filo_sc.sort_values(['super_class', 'post_comp'])

print(mean_filo_sc)

# Plot: grouped bars per super_class (hue = compartment)
plt.figure(figsize=(6, 4))
sns.barplot(
    data=mean_filo_sc,
    x='super_class',
    y='mean_filopodia_on_post',
    hue='post_comp'
)
plt.xlabel('Super class')
plt.ylabel('Mean filopodia_on_post')
plt.legend(title='Post compartment', loc='best')
plt.tight_layout()
plt.savefig(_out_dir / "axon_dend_filo_ratio_by_superclass.svg")
plt.show()
#%%7# Compute mean per super_class and compartment
mean_filo_sc = (
    df2.groupby(['super_class', 'comp2'], as_index=False)['filopodia_on_post']
    .mean()
    .rename(columns={'comp2': 'post_comp',
                     'filopodia_on_post': 'mean_filopodia_on_post'})
)

# Order compartments
mean_filo_sc['post_comp'] = pd.Categorical(
    mean_filo_sc['post_comp'],
    categories=['A', 'D'],
    ordered=True
)

# Line plot
plt.figure(figsize=(5, 4))

sns.lineplot(
    data=mean_filo_sc,
    x='post_comp',
    y='mean_filopodia_on_post',
    hue='super_class',
    marker='o'
)

plt.xlabel('Post compartment')
plt.ylabel('Mean filopodia_on_post')
plt.legend(title='Super class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(
    _out_dir / "axon_dend_filo_ratio_by_superclass_lines.svg"
)

plt.show()
#%%
# Mean per super_class and compartment
mean_filo_sc = (
    df2.groupby(['super_class', 'comp2'], as_index=False)['filopodia_on_post']
    .mean()
    .rename(columns={'comp2': 'post_comp',
                     'filopodia_on_post': 'mean_filopodia_on_post'})
)

# Add overall (all superclasses combined)
mean_filo_all = (
    df2.groupby('comp2', as_index=False)['filopodia_on_post']
    .mean()
    .rename(columns={'comp2': 'post_comp',
                     'filopodia_on_post': 'mean_filopodia_on_post'})
)

mean_filo_all['super_class'] = 'Overall'

# Combine
mean_filo_combined = pd.concat([mean_filo_sc, mean_filo_all], ignore_index=True)

# Order compartments
mean_filo_combined['post_comp'] = pd.Categorical(
    mean_filo_combined['post_comp'],
    categories=['A', 'D'],
    ordered=True
)

# Plot
plt.figure(figsize=(5, 4))

sns.lineplot(
    data=mean_filo_combined,
    x='post_comp',
    y='mean_filopodia_on_post',
    hue='super_class',
    marker='o'
)

plt.xlabel('Post compartment')
plt.ylabel('Mean filopodia_on_post')
plt.legend(title='Super class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(
    _out_dir / "axon_dend_filo_ratio_superclass_plus_overall.svg"
)

plt.show()


