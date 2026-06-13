
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    NEURONS_NT_BWF_FTR, NEURON_TABLE_FTR, CONNECTIONS_TABLE_FTR, OUTPUT_DIR, METHODS_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
from scipy.stats import gaussian_kde
#%%




df2=pd.read_feather(NEURONS_NT_BWF_FTR)


# Load data
nodesG = pd.read_feather(NEURON_TABLE_FTR)

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
nodesG=nodesG.merge(df2[['neuron','filopodia_fraction']],how='left',on='neuron')


#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)

#%%
connections=pd.read_feather(CONNECTIONS_TABLE_FTR)
#%%
connections=connections.merge(nodesG[['root_id','super_class']],left_on='pre',right_on='root_id',how='left')
connections=connections.merge(nodesG[['root_id','super_class']],left_on='post',right_on='root_id',how='left').dropna()
#%%
aa=connections[connections['pre']==connections['post']]
#%%
#%%
connections=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 
       'sum_syn', 'reciprocal', 
       'same_type']].dropna()

#%%

# Step 1: Identify all unique neurons from 'pre' and 'post' columns
unique_neurons = nodesG[['root_id']]
#%%
# Step 2: Count total unique partners for each neuron
partner_counts = (
    pd.concat([
        connections[['pre', 'post', 'sum_syn']],
        connections[['post', 'pre', 'sum_syn']].rename(columns={'post': 'pre', 'pre': 'post'})
    ])
    .groupby('pre')
    .agg(
        total_partners=('post', 'nunique'),
        total_synapses=('sum_syn', 'sum')
    )
)
#%%
# Step 3: Count reciprocal connections for each neuron
reciprocal_df = connections[connections['reciprocal'] == 1]

reciprocal_counts = (
    pd.concat([
        reciprocal_df[['pre', 'post', 'sum_syn']],
        reciprocal_df[['post', 'pre', 'sum_syn']].rename(columns={'post': 'pre', 'pre': 'post'})
    ])
    .groupby('pre')
    .agg(
        reciprocal_partners=('post', 'nunique'),
        reciprocal_synapses=('sum_syn', 'sum')
    )
)

#%%
# Combine the two series into a single DataFrame with each neuron as a row
final_neuron_stats_df = (
    pd.DataFrame(index=partner_counts.index)
    .join(partner_counts, how='left')
    .join(reciprocal_counts, how='left')
    .fillna(0)
    .astype(int)
    .reset_index()
    .rename(columns={'index': 'neuron'})
)
#%%
final_neuron_stats_df_d=final_neuron_stats_df.dropna()
#%%
final_neuron_stats_df['pre']=final_neuron_stats_df['pre'].astype(str)
nodesG['root_id']=nodesG['root_id'].astype(str)

#%%
final_neuron_stats_df['non_reciprocal_partners']=final_neuron_stats_df['total_partners']-final_neuron_stats_df['reciprocal_partners']
final_neuron_stats_df['non_reciprocal_synapses']=final_neuron_stats_df['total_synapses']-final_neuron_stats_df['reciprocal_synapses']
final_neuron_stats_df['non_reciprocal_ratio']=final_neuron_stats_df['non_reciprocal_synapses']/final_neuron_stats_df['non_reciprocal_partners']
final_neuron_stats_df['reciprocal_ratio']=final_neuron_stats_df['reciprocal_synapses']/final_neuron_stats_df['reciprocal_partners']
final_neuron_stats_df['ratio']=final_neuron_stats_df['reciprocal_partners']/final_neuron_stats_df['total_partners']
final_neuron_stats_df['ratio_syn']=final_neuron_stats_df['reciprocal_ratio']/final_neuron_stats_df['non_reciprocal_ratio']
final_neuron_stats_df=final_neuron_stats_df.merge(nodesG[['root_id','super_class']],left_on='pre',right_on='root_id',how='right')
#%%
final_neuron_stats_df['reciprocal_ratio']=final_neuron_stats_df['reciprocal_ratio']/2
#%%
final_neuron_stats_df['root_id']=final_neuron_stats_df['root_id'].astype(np.int64)
#%%
final_neuron_stats_df=final_neuron_stats_df.merge(df2[['neuron','filopodia_fraction']],how='left',left_on='root_id',right_on='neuron')
final_neuron_stats_df=final_neuron_stats_df.merge(nodesG[['neuron','SI']],how='left',left_on='root_id',right_on='neuron')


#%%
final_neuron_stats_df[['filopodia_fraction','ratio','SI']].corr()

#%%

g=final_neuron_stats_df.groupby(by=['super_class'])[['filopodia_fraction','ratio','SI']].corr()
g=g.reset_index()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

custom_palette = {
    "ascending": "#6EB6F6",
    "visual centrifugal": "#44733B",
    "descending": "#803D3D",
    "endocrine": "#8973B2",
    "motor": "#B48667",
    "optic": "#F4D826",
    "sensory": "#848484",
    "central": "#F9574E",
    "visual projection": "#D5A848",
}

# correlation table per super_class
g = (
    final_neuron_stats_df
    .groupby('super_class')[['filopodia_fraction', 'ratio', 'SI']]
    .corr()
    .reset_index()
)

# keep only correlations of filopodia_fraction with ratio and SI
g2 = g[g['level_1'] == 'filopodia_fraction'][['super_class', 'ratio', 'SI']]

# reshape to long form
plot_df = g2.melt(
    id_vars='super_class',
    value_vars=['ratio', 'SI'],
    var_name='x',
    value_name='corr'
)

# ---- make super_class labels match your palette keys ----
plot_df['super_class'] = (
    plot_df['super_class']
    .astype(str)
    .str.replace('_', ' ')
)

plt.figure(figsize=(6, 4))
sns.lineplot(
    data=plot_df,
    x='x',
    y='corr',
    hue='super_class',
    palette=custom_palette,
    marker='o'
)

plt.ylabel('Correlation with filopodia_fraction')
plt.xlabel('')
plt.title('Corr(filopodia_fraction, ratio/SI) by super_class')
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig5" / "reciprocal_fraction_x_bwf"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_x_reciprocal_ratio_x_bwf.svg")
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# correlation table per super_class
g = (
    final_neuron_stats_df
    .groupby('super_class')[['filopodia_fraction', 'ratio', 'SI']]
    .corr()
    .reset_index()
)

# take only corr of filopodia_fraction with ratio and SI
g2 = g[g['level_1'] == 'filopodia_fraction'][['super_class', 'ratio', 'SI']]

# make matrix: rows=super_class, cols=[ratio, SI]
hm = g2.set_index('super_class')[['ratio', 'SI']]

# optional: nicer labels (match your naming style)
hm.index = hm.index.astype(str).str.replace('_', ' ')
hm.columns = ['ratio', 'SI']

plt.figure(figsize=(5, 6))
sns.heatmap(
    hm,
    annot=True,
    fmt=".2f",
    vmin=-1, vmax=1,
    center=0,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Corr(filopodia_fraction, ratio/SI) per super_class")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
