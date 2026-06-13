
from nglui.statebuilder import ChainedStateBuilder

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'


import matplotlib.pyplot as plt
import numpy as np
from fafbseg import flywire
import navis
import pandas as pd

import time
import datetime
import os
import sys
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, CLASSIFICATION_CSV, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
client = CAVEclient('flywire_fafb_production')
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
import itertools
import gc
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.dropna(subset=['axon_correct','dend_correct','primary_type'])

#%%
nodesG=nodesG[nodesG['super_class'].isin(['optic', 'central', 'visual_projection','visual_centrifugal'])]
#%%                                        

#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_FTR)
       #%%
sid=3
#%%      
aa=syn_df.head(20)
#%%
aa=syn_df.iloc[1532:1536]
       #%%
syn_df=syn_df.merge(nodesG[['neuron','super_class']],left_on='pre',right_on='neuron',how='left')       
syn_df=syn_df.merge(nodesG[['neuron','super_class']],left_on='post',right_on='neuron',how='left')       
#%%
syn_df=syn_df.dropna(subset=['super_class_x','super_class_y'])
#%%          ------------   Fig 1a - SI_dist ---------
#syn_df=syn_df[(syn_df['SI_pre']>=0.1)&(syn_df['SI_post']>=0.1)]
#%%
nodesG=nodesG.dropna(subset='primary_type')
#%%
#%%
nodesG['primary_SI_mean'] = nodesG.groupby('primary_type')['SI'].transform('mean')
#%%
aa=syn_df.head()
#%%

#%%
aa=syn_df[['pre','post','cleft_score','pre_x','pre_y','pre_z','post_x','post_y','post_z']].head()
#%%
aa.columns=['pre_neuron','post_neuron','cleft_score','pre_x','pre_y','pre_z','post_x','post_y','post_z']

#%%
#%%
syn_df=syn_df[['pre','post','SI_pre','SI_post','comp']]
#%%
#syn_df=syn_df[(syn_df['SI_post']>=0.1)&(syn_df['SI_pre']>=0.1)]
#%%
syn_df_m=syn_df.merge(nodesG[['root_id','primary_type','primary_SI_mean']],left_on='pre',right_on='root_id',how='left')
#%%
syn_df_m=syn_df_m.merge(nodesG[['root_id','primary_type','primary_SI_mean']],left_on='post',right_on='root_id',how='left')
#%%
del syn_df
#%%
syn_df_m=syn_df_m[syn_df_m['comp'].isin(['AA','DD','DA','AD'])]
#%%
syn_df_m=syn_df_m[(syn_df_m['primary_SI_mean_x']>=0.1)&(syn_df_m['primary_SI_mean_y']>=0.1)]
#%%
syn_df_m=syn_df_m.dropna(subset=['primary_type_x','primary_type_y'])
#%%
syn_df_m_s=syn_df_m[syn_df_m['primary_type_x']==syn_df_m['primary_type_y']]

#%%
AA=syn_df_m[syn_df_m['comp']=='AA']
DD=syn_df_m[syn_df_m['comp']=='DD']
AD=syn_df_m[syn_df_m['comp']=='AD']
DA=syn_df_m[syn_df_m['comp']=='DA']

#%%
DA['same_type']=(DA['primary_type_x']==DA['primary_type_y']).astype(int)

AA['same_type']=(AA['primary_type_x']==AA['primary_type_y']).astype(int)
DD['same_type']=(DD['primary_type_x']==DD['primary_type_y']).astype(int)
AD['same_type']=(AD['primary_type_x']==AD['primary_type_y']).astype(int)
#%%
r=DD.groupby(by='primary_type_x').size()
#%%
AD['same_type'].sum()/AD['same_type'].count()

#%%
DD['same_type'].sum()/DD['same_type'].count()
#%%
AA['same_type'].sum()/AA['same_type'].count()
#%%
syn_df_m=syn_df_m[['pre','post','SI_pre','SI_post','comp','primary_type_x','primary_type_y']].dropna()
#%%%
syn_df_m['same_type']=syn_df_m['primary_type_x']==syn_df_m['primary_type_y']
#%%
syn_df_m['same_type']=syn_df_m['same_type'].astype(int)
aa=syn_df_m.head(1000)
#%%
syn_df_m=syn_df_m.dropna(subset=['primary_type_x','primary_type_y'])

#%%
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

import seaborn as sns
import matplotlib.pyplot as plt

# Compute mean values and sort by highest proportion
mean_values = syn_df_m.groupby('comp')['same_type'].mean().sort_values(ascending=False).reset_index()

# Compute raw count of same_type connections per compartment (in millions)
raw_counts = syn_df_m.groupby('comp')['same_type'].sum().reindex(mean_values['comp']) / 1_000_000
total_counts = syn_df_m.groupby('comp').size().reindex(mean_values['comp']) / 1_000_000

# Dynamically assign colors based on custom_palette or use a default color
default_color = "#999999"  # Use a neutral default if not in the custom palette
colors = [custom_palette.get(comp, default_color) for comp in mean_values['comp']]

# Create the barplot
plt.figure(figsize=(1.7,1.7))
ax = sns.barplot(x=mean_values['comp'], y=mean_values['same_type']*100, palette=colors)

# Annotate bars with raw values in x.xxM / y.xxM format (millions)
for bar, raw, total in zip(ax.patches, raw_counts.values, total_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{raw:.2f}M / {total:.2f}M",
            ha='center', va='bottom', fontsize=4)
plt.title("")
plt.xticks(size=8)
plt.yticks(size=8)
sns.despine(right=True)
# Add labels and title
plt.xlabel('',size=8)
plt.ylabel('Percent of Same Type',size=8)
plt.ylim(0, 20)  # Slightly extend the y-axis for annotation visibility
_out_dir = OUTPUT_DIR / "fig3" / "syntype_x_identity"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "syn_type_same_ptype_princeton.svg")
# Show the plot
plt.show()


#%%
import matplotlib.pyplot as plt

# Define dictionary with summed values
dic_ = {'AD': AD['same_type'].sum(), 'AA': AA['same_type'].sum(),
        'DD': DD['same_type'].sum(), 'DA': DA['same_type'].sum()}

# Calculate total sum
total_syn = sum(dic_.values())

# Custom color palette
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

# Create pie chart
plt.figure(figsize=(1, 1))
plt.pie(
    dic_.values(), 
    labels=dic_.keys(), 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=[custom_palette[key] for key in dic_.keys()],    textprops={'fontsize': 4}  # This sets the font size for both labels and autopct

)

# Add total sum text in the figure
plt.text(0, -1.2, f"Total Syn: {total_syn:,}", ha='center', fontsize=4, fontweight='bold')

# Title and layout
plt.title('Same type connections (SI >=0.1)',size=4)
plt.tight_layout()
plt.savefig(_out_dir / "syn_type_same_ptype_pie.svg")

# Show plot
plt.show()



#%%
df=pd.read_csv(CLASSIFICATION_CSV)
#%%
df=df[['root_id','side']]
#%%
df.columns=['neuron','side']
#%%
nodesG=nodesG.merge(df,how='left',on='neuron')
#%%
nodes_agg=pd.pivot_table(nodesG,columns='side',index='primary_type',aggfunc='size')[['left','right']]
#%%
nodes_agg_filtered=nodes_agg.query('left >=1 & right>=1')

#%%
syn_df_m_one_primary_filter = syn_df_m[
    (syn_df_m['primary_type_x'].isin(nodes_agg_filtered.index)) &
    (syn_df_m['primary_type_y'].isin(nodes_agg_filtered.index))
]



syn_df_m_one_primary_filter=syn_df_m_one_primary_filter.merge(nodesG[['neuron','side']],left_on='pre',right_on='neuron')
syn_df_m_one_primary_filter=syn_df_m_one_primary_filter.merge(nodesG[['neuron','side']],left_on='post',right_on='neuron')


syn_df_m_one_primary_filter=syn_df_m_one_primary_filter[syn_df_m_one_primary_filter['same_type']==1]


syn_df_m_one_primary_filter['same_side']=(syn_df_m_one_primary_filter['side_x']==syn_df_m_one_primary_filter['side_y']).astype(int)


number_of_primry_types=len(syn_df_m_one_primary_filter['primary_type_x'].unique())



syn_df_m_one_primary_filter['type']=syn_df_m_one_primary_filter['primary_type_x']


r_pivot=pd.pivot_table(syn_df_m_one_primary_filter,columns=['comp','same_side'],aggfunc='size',index='type').fillna(0)

r_pivot = (r_pivot > 0).astype(int)

aaa=syn_df_m_one_primary_filter.head(100)

r_pivot.columns = [f"{comp}_{same}" for comp, same in r_pivot.columns]


r_pivot['canonical_same']=r_pivot['AD_1']
r_pivot['non_canonical_same']=(r_pivot['AA_1']+r_pivot['DD_1']+r_pivot['DA_1'])
r_pivot['non_canonical_same']=(r_pivot['non_canonical_same'] > 0).astype(int)

r_pivot['canonical_not_same']=r_pivot['AD_0']
r_pivot['non_canonical_not_same']=(r_pivot['AA_0']+r_pivot['DD_0']+r_pivot['DA_0'])
r_pivot['non_canonical_not_same']=(r_pivot['non_canonical_not_same'] > 0).astype(int)

fraction_ipsinon_non=r_pivot['non_canonical_same'].mean()
fraction_contra_non=r_pivot['non_canonical_not_same'].mean()

fraction_ipsinon_can=r_pivot['canonical_same'].mean()
fraction_contra_can=r_pivot['canonical_not_same'].mean()

# Build a small 2×2 DataFrame
heatmap_data = pd.DataFrame({
    "canonical": {
        "ipsi": fraction_ipsinon_can,
        "Contra": fraction_contra_can
    },
    "non-canonical": {
        "ipsi": fraction_ipsinon_non,
        "Contra": fraction_contra_non
    }
})

plt.figure(figsize=(4,4))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".2f", cmap="viridis",
    cbar=True, linewidths=0.5
)
plt.title("Fraction of any occourance per type × Canonicality", fontsize=10)
plt.ylabel("")  # optional clean labels
plt.xlabel("")
plt.tight_layout()
plt.savefig(_out_dir / "syntpye_cannonicality_sides_anyfractionheatmap_princeton_same_sametype.svg")

plt.show()
#%%
plot_df = pd.DataFrame({
    "side": ["contra", "ipsi", "contra", "ipsi"],
    "fraction": [
        fraction_contra_can,
        fraction_ipsinon_can,
        fraction_contra_non,
        fraction_ipsinon_non
    ],
    "canonicality": [
        "canonical", "canonical",
        "non-canonical", "non-canonical"
    ]
})
import matplotlib.pyplot as plt

COLORS = {
    "canonical": "#8B4B20",     # brown
    "non-canonical": "#000000" # black
}

plt.figure(figsize=(3.2, 3.2))

for canon in ["canonical", "non-canonical"]:
    df = plot_df[plot_df["canonicality"] == canon]

    plt.plot(
        df["side"],
        df["fraction"],
        marker="o",
        linewidth=2,
        color=COLORS[canon],
        label=canon
    )

plt.ylabel("Fraction of primary types")
plt.xlabel("")
plt.ylim(0, 1)

plt.title("Canonical vs non-canonical\nIpsilateral vs contralateral", fontsize=9)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(
    _out_dir / "syntype_canonicality_sides_lineplot.svg"
)
plt.show()





#%%
syn_df_m_one_primary_filter = syn_df_m[
    (syn_df_m['primary_type_x'].isin(nodes_agg_filtered.index)) &
    (syn_df_m['primary_type_y'].isin(nodes_agg_filtered.index))
]



syn_df_m_one_primary_filter=syn_df_m_one_primary_filter.merge(nodesG[['neuron','side']],left_on='pre',right_on='neuron')
syn_df_m_one_primary_filter=syn_df_m_one_primary_filter.merge(nodesG[['neuron','side']],left_on='post',right_on='neuron')


syn_df_m_one_primary_filter=syn_df_m_one_primary_filter[syn_df_m_one_primary_filter['same_type']==0]

syn_df_m_one_primary_filter['same_side']=(syn_df_m_one_primary_filter['side_x']==syn_df_m_one_primary_filter['side_y']).astype(int)

number_of_primry_types=len(syn_df_m_one_primary_filter['primary_type_x'].unique())


syn_df_m_one_primary_filter['type']=syn_df_m_one_primary_filter['primary_type_x']


r_pivot=pd.pivot_table(syn_df_m_one_primary_filter,columns=['comp','same_side'],aggfunc='size',index='type').fillna(0)

r_pivot = (r_pivot > 0).astype(int)

aaa=syn_df_m_one_primary_filter.head(100)

r_pivot.columns = [f"{comp}_{same}" for comp, same in r_pivot.columns]


r_pivot['canonical_same']=r_pivot['AD_1']
r_pivot['non_canonical_same']=(r_pivot['AA_1']+r_pivot['DD_1']+r_pivot['DA_1'])
r_pivot['non_canonical_same']=(r_pivot['non_canonical_same'] > 0).astype(int)

r_pivot['canonical_not_same']=r_pivot['AD_0']
r_pivot['non_canonical_not_same']=(r_pivot['AA_0']+r_pivot['DD_0']+r_pivot['DA_0'])
r_pivot['non_canonical_not_same']=(r_pivot['non_canonical_not_same'] > 0).astype(int)

fraction_ipsinon_non=r_pivot['non_canonical_same'].mean()
fraction_contra_non=r_pivot['non_canonical_not_same'].mean()

fraction_ipsinon_can=r_pivot['canonical_same'].mean()
fraction_contra_can=r_pivot['canonical_not_same'].mean()


# Build a small 2×2 DataFrame
heatmap_data = pd.DataFrame({
    "canonical": {
        "ipsi": fraction_ipsinon_can,
        "Contra": fraction_contra_can
    },
    "non-canonical": {
        "ipsi": fraction_ipsinon_non,
        "Contra": fraction_contra_non
    }
})

plt.figure(figsize=(4,4))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".2f", cmap="viridis",
    cbar=True, linewidths=0.5
)
plt.title("Fraction of any occourance per type × Canonicality", fontsize=10)
plt.ylabel("")  # optional clean labels
plt.xlabel("")

plt.tight_layout()
plt.savefig(_out_dir / "syntpye_cannonicality_sides_anyfractionheatmap_princeton_same.svg")

plt.show()
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Start from your base filter ---
syn_df_m_one_primary_filter = syn_df_m[
    (syn_df_m['primary_type_x'].isin(nodes_agg_filtered.index)) &
    (syn_df_m['primary_type_y'].isin(nodes_agg_filtered.index))
]

# Merge sides
syn_df_m_one_primary_filter = syn_df_m_one_primary_filter.merge(
    nodesG[['neuron','side']], left_on='pre', right_on='neuron'
)
syn_df_m_one_primary_filter = syn_df_m_one_primary_filter.merge(
    nodesG[['neuron','side']], left_on='post', right_on='neuron',
    suffixes=('_x','_y')
)

# Flags
syn_df_m_one_primary_filter['same_side'] = (
    syn_df_m_one_primary_filter['side_x'] == syn_df_m_one_primary_filter['side_y']
).astype(int)
syn_df_m_one_primary_filter['type'] = syn_df_m_one_primary_filter['primary_type_x']

# --- Build pivot: comp × same_side × same_type ---
r_pivot = pd.pivot_table(
    syn_df_m_one_primary_filter,
    index='type',
    columns=['comp','same_side','same_type'],
    aggfunc='size'
).fillna(0)

r_pivot = (r_pivot > 0).astype(int)  # presence/absence

# Flatten column names
r_pivot.columns = [f"{comp}_{ss}_{st}" for comp, ss, st in r_pivot.columns]

# Helper: sum over existing
def sum_over_existing(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0, index=df.index)
    return df[cols].sum(axis=1)

# --- Fractions ---
# same_type = 1, ipsi (same_side=1)
frac_ipsi_same_can = r_pivot.get('AD_1_1', pd.Series(0, index=r_pivot.index)).mean()
frac_ipsi_same_non = (sum_over_existing(r_pivot, ['AA_1_1','DD_1_1','DA_1_1']) > 0).astype(int).mean()

# same_type = 1, contra (same_side=0)
frac_contra_same_can = r_pivot.get('AD_0_1', pd.Series(0, index=r_pivot.index)).mean()
frac_contra_same_non = (sum_over_existing(r_pivot, ['AA_0_1','DD_0_1','DA_0_1']) > 0).astype(int).mean()

# same_type = 0, ipsi (same_side=1)
frac_ipsi_notsame_can = r_pivot.get('AD_1_0', pd.Series(0, index=r_pivot.index)).mean()
frac_ipsi_notsame_non = (sum_over_existing(r_pivot, ['AA_1_0','DD_1_0','DA_1_0']) > 0).astype(int).mean()

# same_type = 0, contra (same_side=0)
frac_contra_notsame_can = r_pivot.get('AD_0_0', pd.Series(0, index=r_pivot.index)).mean()
frac_contra_notsame_non = (sum_over_existing(r_pivot, ['AA_0_0','DD_0_0','DA_0_0']) > 0).astype(int).mean()

# --- Build heatmap data ---
heatmap_data = pd.DataFrame({
    "Canonical": {
        "Ipsi · same type": frac_ipsi_same_can,
        "Contra · same type": frac_contra_same_can,
        "Ipsi · not same type": frac_ipsi_notsame_can,
        "Contra · not same type": frac_contra_notsame_can,
    },
    "Non-canonical": {
        "Ipsi · same type": frac_ipsi_same_non,
        "Contra · same type": frac_contra_same_non,
        "Ipsi · not same type": frac_ipsi_notsame_non,
        "Contra · not same type": frac_contra_notsame_non,
    }
})

# --- Plot ---
plt.figure(figsize=(5, 4))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".2f", cmap="viridis",
    vmin=0, vmax=1, linewidths=0.5, cbar=True
)
plt.title("Fraction of any occurrence per type\nIpsi/Contra × Same-type vs Canonicality", fontsize=10)
plt.ylabel("")
plt.xlabel("")
plt.tight_layout()
plt.savefig(_out_dir / "syntpye_cannonicality_sides_anyfractionheatmap_princeton.svg")

plt.show()
#%%
