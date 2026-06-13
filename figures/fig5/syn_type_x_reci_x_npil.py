# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:16:35 2024

@author: user
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import time
import datetime
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, CONNECTIONS_TABLE_FTR, OUTPUT_DIR, METHODS_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
# methods imported via config block above
import itertools
import gc
import time
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
#720575940631275181,
aa=nodesG[nodesG['neuron']==720575940631275181]

#720575940622726663=0.66
#720575940631275181=0.713
#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%          ------------   Fig 1a - SI_dist ---------
syn_df=syn_df[(syn_df['SI_pre']>=0.1)&(syn_df['SI_post']>=0.1)]
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)

syn_df=syn_df.merge(nodesG[['root_id','super_class']],left_on='pre',right_on='root_id',how='left')
syn_df=syn_df.merge(nodesG[['root_id','super_class']],left_on='post',right_on='root_id',how='left').dropna()
#%%
connections=pd.read_feather(CONNECTIONS_TABLE_FTR)
#%%
#connections=connections[connections['sum_syn']>=5]
#%%

connectionsp=connections[['pre','post','reciprocal']]

#%%
syn_df=syn_df.merge(connectionsp[['pre','post','reciprocal']],left_on=['pre','post'],right_on=['pre','post'],how='left')
#%%
syn_df=syn_df.dropna(subset='reciprocal')
#%%

r=syn_df.groupby(by=['npil'])['reciprocal'].mean()

#%%
aa=connections.head()
#%%
syn_df["npil"] = syn_df["npil"].str.replace(r'_(L|R)$', '', regex=True)


#%%
syn_df=syn_df[['pre', 'post',  'npil',
       'comp', 'SI_pre', 'SI_post',  'super_class_x', 
       'super_class_y', 'reciprocal']]
#%%
syn_df=syn_df[syn_df['comp'].isin(['AD','AA','DD','DA'])]
#%%
syn_df_c=syn_df[['npil','comp','reciprocal']].dropna()

syn_df_c_grouped=syn_df_c.groupby(by=['npil','comp','reciprocal']).size()

#%%
syn_df_c_grouped=syn_df_c_grouped.reset_index()
syn_df_c_grouped.columns=['npil','comp','reciprocal','count']

#%%
syn_df_c_grouped_reci=syn_df_c_grouped[syn_df_c_grouped['reciprocal']==1]
syn_df_c_grouped_reci=syn_df_c_grouped_reci[['npil','comp','count']]

syn_df_c_pivot_reci=pd.pivot_table(syn_df_c_grouped_reci,values='comp',index='npil',columns='comp')

syn_df_c_pivot_reci=syn_df_c_pivot_reci[['AD','AA','DD','DA']]
syn_df_c_pivot_na_reci=syn_df_c_pivot_reci.fillna(0)

syn_df_c_grouped_not_reci=syn_df_c_grouped[syn_df_c_grouped['reciprocal']==0]
syn_df_c_grouped_not_reci=syn_df_c_grouped_not_reci[['npil','comp','count']]

syn_df_c_pivot_not_reci=pd.pivot_table(syn_df_c_grouped_not_reci,values='comp',index='npil',columns='comp')

syn_df_c_pivot_not_reci=syn_df_c_pivot_not_reci[['AD','AA','DD','DA']]
syn_df_c_pivot_na_not_reci=syn_df_c_pivot_not_reci.fillna(0)
#%%
syn_df_c_pivot_na_not_reci=syn_df_c_pivot_na_not_reci.reset_index(drop=False)

syn_df_c_pivot_na_reci=syn_df_c_pivot_na_reci.reset_index(drop=False)
#%%
syn_df_c_pivot_na_not_reci.columns=['npil', 'AD_nr', 'AA_nr', 'DD_nr', 'DA_nr']
syn_df_c_pivot_na_reci.columns=['npil', 'AD_r', 'AA_r', 'DD_r', 'DA_r']
#%%
com_df=syn_df_c_pivot_na_not_reci.merge(syn_df_c_pivot_na_reci,on='npil',how='inner')
com_df=com_df[com_df['npil']!='None']
com_df_raw=com_df.copy()
com_df.iloc[:, 1:] = (com_df.iloc[:, 1:].div(com_df.iloc[:, 1:].sum(axis=1), axis=0) * 100).round(1)



#%%


#%%
import matplotlib.pyplot as plt

# Custom colors
custom_palette = {
    'AD': '#9F4800',
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'DA': '#B3B3B3'
}

# Define columns
nr_cols = ['AD_nr', 'AA_nr', 'DD_nr', 'DA_nr']
r_cols = ['AD_r', 'AA_r', 'DD_r', 'DA_r']
all_cols = nr_cols + r_cols

# Sorting helpers from normalized com_df
com_df['r_total'] = com_df[r_cols].sum(axis=1)
com_df['AD_total'] = com_df['AD_r'] + com_df['AD_nr']
com_df_sorted = com_df.sort_values(by=['r_total', 'AD_total'], ascending=[False, False]).reset_index(drop=True)
com_df_sorted = com_df_sorted.drop(columns=['r_total', 'AD_total'])

# Use same sorting index for raw values
com_df_raw_sorted = com_df_raw.set_index('npil').loc[com_df_sorted['npil'].values].reset_index()

# Plot setup
fig, ax = plt.subplots(figsize=(22, 12))
labels = com_df_sorted['npil']
bottom_nr = [0] * len(com_df_sorted)

# Plot nr group
for col in nr_cols:
    prefix = col.split('_')[0]
    ax.bar(
        labels,
        com_df_sorted[col],
        bottom=bottom_nr,
        color=custom_palette[prefix],
        edgecolor='black',
        label=f'{prefix} nr'
    )
    bottom_nr = [i + j for i, j in zip(bottom_nr, com_df_sorted[col])]

# Plot r group
bottom_r = bottom_nr.copy()
for col in r_cols:
    prefix = col.split('_')[0]
    ax.bar(
        labels,
        com_df_sorted[col],
        bottom=bottom_r,
        color=custom_palette[prefix],
        edgecolor='black',
        hatch='xx',
        label=f'{prefix} r'
    )
    bottom_r = [i + j for i, j in zip(bottom_r, com_df_sorted[col])]

# Annotate n= values (from com_df_raw)
for i, row in com_df_raw_sorted.iterrows():
    total_syns = row[all_cols].sum()
    ax.text(
        i,
        bottom_r[i] + 1,  # slightly above top
        f"{int(total_syns/1000):,}k",
        ha='center', va='bottom', fontsize=9, fontweight='bold',rotation=45
    )

# Final touches
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_ylabel("Percentage")
ax.set_title("neuropil, synaptic type, reciprocal")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt

# Custom colors per synapse type
custom_palette = {
    'AD': '#9F4800',
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'DA': '#B3B3B3'
}

# Define symmetric stacking order (top → bottom)
plot_order = ['AD_nr', 'AA_nr', 'DD_nr', 'DA_nr', 'DA_r', 'DD_r', 'AA_r', 'AD_r']

# Add sorting helpers
com_df['r_total'] = com_df[['AD_r', 'AA_r', 'DD_r', 'DA_r']].sum(axis=1)
com_df['AD_total'] = com_df['AD_r'] + com_df['AD_nr']

# Sort: by r_total descending, then AD_total descending
com_df_sorted = com_df.sort_values(by=['r_total', 'AD_total'], ascending=[False, False]).reset_index(drop=True)
com_df_sorted = com_df_sorted.drop(columns=['r_total', 'AD_total'])

# Setup plot
fig, ax = plt.subplots(figsize=(22,12))
labels = com_df_sorted['npil']
bottom = [0] * len(com_df_sorted)

# Plot in symmetric order
for col in plot_order:
    prefix = col.split('_')[0]
    is_r = col.endswith('_r')
    hatch = 'xx' if is_r else ''
    
    ax.bar(
        labels,
        com_df_sorted[col],
        bottom=bottom,
        color=custom_palette[prefix],
        edgecolor='black',
        hatch=hatch,
        label=f"{prefix} {'r' if is_r else 'nr'}"
    )
    bottom = [i + j for i, j in zip(bottom, com_df_sorted[col])]
for i, row in com_df_raw_sorted.iterrows():
    total_syns = row[all_cols].sum()
    ax.text(
        i,
        bottom_r[i] + 1,  # slightly above top
        f"{int(total_syns/1000):,}k",
        ha='center', va='bottom', fontsize=9, fontweight='bold',rotation=45
    )

# Format
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_ylabel("Percentage")
ax.set_title("neuropil, synaptic type, reciprocal")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#%%

# Custom colors per synapse type
custom_palette = {
    'AD': '#9F4800',
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'DA': '#B3B3B3'
}

# Define symmetric stacking order (top → bottom)
##plot_order = ['AD_nr', 'AA_nr', 'DD_nr', 'DA_nr', 'DA_r', 'DD_r', 'AA_r', 'AD_r']
plot_order = ['AD_nr','AD_r','AA_nr','AA_r','DD_nr','DD_r','DA_nr','DA_r']

# Add sorting helpers
# Add sorting helpers
'''
com_df['r_total'] = com_df[['AD_r', 'AA_r', 'DD_r', 'DA_r']].sum(axis=1)
com_df['AD_total'] = com_df['AD_r'] + com_df['AD_nr']

# Sort: by AD_total ascending, then r_total ascending
com_df_sorted = com_df.sort_values(by=['AD_total', 'r_total'], ascending=[True, True]).reset_index(drop=True)
com_df_sorted = com_df_sorted.drop(columns=['r_total', 'AD_total'])
'''


com_df['r_total'] = com_df[['AD_r', 'AA_r', 'DD_r', 'DA_r']].sum(axis=1)
com_df['AD_total'] = com_df['AD_r'] + com_df['AD_nr']
ax.tick_params(axis='x', labelsize=4)
ax.tick_params(axis='y', labelsize=8)
# Sort: by AD_total ascending, then r_total ascending
com_df_sorted = com_df.sort_values(by=['AD_total', 'r_total'], ascending=[False,False]).reset_index(drop=True)
com_df_sorted = com_df_sorted.drop(columns=['r_total', 'AD_total'])
#%%$
# --- Color map for neuropils ---
npil_colors = {
    'FB':'#12BFB2','EB':'#23C6C6','PB':'#0AA0C9','NO':'#0A76B2',
    'AMMC':'#4455FF','FLA':'#2F62FF','CAN':'#2F62FF','PRW':'#3333FF',
    'SAD':'#3333FF','GNG':'#3333FF','AL':'#3399FF','LH':'#3366FF','BU':'#3333FF',
    # MB family
    'MB_PED':'#FF9966','MB_VL':'#FF9966','MB_ML':'#FF9966','MB_CA':'#FF9966',
    # central yellow family
    'LAL':'#FFCC33','SLP':'#FFCC33','SIP':'#FFCC33','SMP':'#FFCC33','CRE':'#FFCC33',
    'IB':'#FFCC33','ATL':'#FFCC33',
    # green family
    'VES':'#33CC66','EPA':'#00CC66','GOR':'#00CC66','SPS':'#00CC66','IPS':'#00CC66','AOTU':'#00CC66',
    # light-blue family
    'AVLP':'#3399FF','PVLP':'#3399FF','PLP':'#3399FF','WED':'#3399FF',
    # purple family
    'ME':'#CC3399','AME':'#CC3399','LO':'#9933CC','LOP':'#9933CC','LA':'#9933CC','OCG':'#9933CC'
}

# --- Setup plot ---
fig, ax = plt.subplots(figsize=(4.5, 1.8))

labels = com_df_sorted['npil']
bottom = [0] * len(com_df_sorted)

ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("Percentage")

# --- Plot bars (unchanged logic) ---
for col in plot_order:
    prefix = col.split('_')[0]
    is_r = col.endswith('_r')
    hatch = 'xxxxxxxxxxx' if is_r else ''

    ax.bar(
        range(len(labels)),
        com_df_sorted[col],
        bottom=bottom,
        color=custom_palette[prefix],
        edgecolor='black',
        hatch=hatch,
        linewidth=0.15,
        label=f"{prefix} {'r' if is_r else 'nr'}"
    )

    bottom = [i + j for i, j in zip(bottom, com_df_sorted[col])]

# --- Annotate totals ---
for i, row in com_df_raw_sorted.iterrows():
    total_syns = row[all_cols].sum()
    ax.text(
        i + 0.4,
        bottom[i] + 1,
        f"{int(total_syns/1000):,}k",
        ha='right',
        va='bottom',
        fontsize=4,
        rotation=90
    )

# --- X-axis labels with neuropil colors ---
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90)

for tick_label in ax.get_xticklabels():
    npil = tick_label.get_text()
    tick_label.set_color(npil_colors.get(npil, '#000000'))

# --- Formatting ---
sns.despine(top=True, right=True)
ax.tick_params(axis='x', labelsize=4)
ax.tick_params(axis='y', labelsize=8)
mpl.rcParams['hatch.linewidth'] = 0.2

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5, ncol=1)
plt.tight_layout()

_out_dir = OUTPUT_DIR / "fig5" / "syn_type_x_reci_x_npil"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    _out_dir / "syn_type_npil_reci_princeton_non_filtered.pdf"
)
plt.show()


#%%
