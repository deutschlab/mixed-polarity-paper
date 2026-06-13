
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, CONNECTIONS_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
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
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG = nodesG[nodesG['super_class'].isin(['central', 'optic', 'visual_centrifugal', 'visual_projection'])]
nodesG=nodesG.dropna(subset=['axon_correct','dend_correct','primary_type'])

#%%
nodesG=nodesG[['neuron','SI','super_class']]


#%%
connections=pd.read_feather(CONNECTIONS_TABLE_FTR)
connections=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 
       'sum_syn', 'reciprocal', 
       'same_type']]
connections=connections.merge(nodesG,left_on='pre',right_on='neuron',how='left')
connections=connections.merge(nodesG,left_on='post',right_on='neuron',how='left')
#%%
connections = connections[connections['super_class_x'].isin(['central', 'optic', 'visual_centrifugal', 'visual_projection'])]
connections = connections[connections['super_class_y'].isin(['central', 'optic', 'visual_centrifugal', 'visual_projection'])]

#%%
connections=connections.query("SI_x>=0.1 and SI_y>=0.1")
#%%
#connections=connections[connections['reciprocal']==1]
#%%
AD=connections[connections['AD']>=1]
AA=connections[connections['AA']>=1]
DD=connections[connections['DD']>=1]
DA=connections[connections['DA']>=1]
#%%
types=['AD','AA','DD','DA']
d={}
n_counts={}
for type_1 in types:
    len_type=len(connections.query(f"{type_1} >= 1"))
    for type_2 in types:
            r = len(connections.query(f"{type_1} >= 1 and {type_2} >= 1")) / len_type
            d[str(type_1)+'_'+str(type_2)]=r
    n_counts[type_1]=len_type
            #%%
# Convert to DataFrame for heatmap
heatmap_df = pd.DataFrame(index=types, columns=types)

for key, value in d.items():
    t1, t2 = key.split('_')
    heatmap_df.loc[t1, t2] = value


# Prepare DataFrame for heatmap
row_labels = [f"{t} (n={n_counts[t]})" for t in types]
heatmap_df = pd.DataFrame(index=row_labels, columns=types)

for key, value in d.items():
    t1, t2 = key.split('_')
    row_label = f"{t1} (n={n_counts[t1]})"
    heatmap_df.loc[row_label, t2] = value

heatmap_df = heatmap_df.astype(float)


plt.rc('axes', titlesize=8)         # Title size
plt.rc('axes', labelsize=4)         # X and Y label size
plt.rc('xtick', labelsize=8)        # X tick label size
plt.rc('ytick', labelsize=4)        # Y tick label size
plt.rc('legend', fontsize=8)        # Legend text size
plt.rc('legend', title_fontsize=8) 
# Plot heatmap
plt.figure(figsize=(2.4,1.5))
sns.heatmap(heatmap_df, annot=True, square=True, fmt=".2f",cmap='Blues',vmax=0.25,annot_kws={"size": 4})
#plt.title('Proportion of Connections to Target Type\nGiven Query Type ≥ 1')
plt.xlabel('Conditional Type')
plt.ylabel('Given Type')
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig3" / "syn_type_prob"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "conditional_chance_syn_type_princeton.svg")
plt.show()
#%%
types=['AD','AA','DD','DA']
d={}
for type_1 in types:
        r = connections.query(f"{type_1} >= 1")[['AD','AA','DD','DA']]
        r2=((r.T/r.T.sum()).T)
        r2.mean()
        d[str(type_1)]=r2.mean()


#%%

types = ['AD', 'AA', 'DD', 'DA']
d = {}
n_counts = {}

for type_1 in types:
    r = connections.query(f"{type_1} >= 1")[['AD', 'AA', 'DD', 'DA']]
    r2 = ((r.T / r.T.sum()).T)
    d[str(type_1)] = r2.mean()
    n_counts[type_1] = len(r)

# Convert to DataFrame and add sample sizes to index
df = pd.DataFrame(d).T
df.index = [f"{ix} (n={n_counts[ix]})" for ix in df.index]

# Plot heatmap
plt.figure(figsize=(2.4,1.5))
# Convert values to formatted strings with %
annot_data = df.applymap(lambda x: f"{x:.2f}")
plt.rc('axes', titlesize=8)         # Title size
plt.rc('axes', labelsize=4)         # X and Y label size
plt.rc('xtick', labelsize=8)        # X tick label size
plt.rc('ytick', labelsize=4)        # Y tick label size
plt.rc('legend', fontsize=8)        # Legend text size
plt.rc('legend', title_fontsize=8) 
sns.heatmap(df, annot=annot_data, fmt="", square=True,cmap='Blues', vmin=0, vmax=0.25,annot_kws={"size": 4})
#plt.title('Average Percent Composition per Connection Type')
plt.xlabel('Distribution of synaptic types')
plt.ylabel('Given a connection with synaptic type')
plt.tight_layout()
plt.savefig(_out_dir / "conditional_percentage_syn_type_princeton_bign.svg")

plt.show()
#%%purity:    
dfc=connections[['pre','post','AA','AD','DD','DA','sum_syn']]
#%%
AD_pure=(dfc['AD']==dfc['sum_syn']).astype(int).sum()
AD_total=(dfc['AD']>0).astype(int).sum()
Pure_ratio_AD=AD_pure/AD_total

DD_pure=(dfc['DD']==dfc['sum_syn']).astype(int).sum()
DD_total=(dfc['DD']>0).astype(int).sum()
Pure_ratio_DD=DD_pure/DD_total

AA_pure=(dfc['AA']==dfc['sum_syn']).astype(int).sum()
AA_total=(dfc['AA']>0).astype(int).sum()
Pure_ratio_AA=AA_pure/AA_total

DA_pure=(dfc['DA']==dfc['sum_syn']).astype(int).sum()
DA_total=(dfc['DA']>0).astype(int).sum()
Pure_ratio_DA=DA_pure/DA_total
#%%
import pandas as pd

data = {
    'Pure Count': [AD_pure, DD_pure, AA_pure, DA_pure],
    'Total Count': [AD_total, DD_total, AA_total, DA_total],
    'Pure Ratio (%)': [Pure_ratio_AD*100, Pure_ratio_DD*100, Pure_ratio_AA*100, Pure_ratio_DA*100]
}

df_results = pd.DataFrame(data, index=['AD', 'DD', 'AA', 'DA'])
print(df_results)
import matplotlib.pyplot as plt

df_results['Pure Ratio (%)'].plot(kind='bar', color='skyblue')
plt.ylabel('Pure Ratio (%)')
plt.title('Proportion of Pure Connections per Type')
plt.ylim(0, 100)
plt.show()
#%%
import matplotlib.pyplot as plt

# Prepare data
labels = [
    'AD Pure', 'AD Mixed',
    'DD Pure', 'DD Mixed',
    'AA Pure', 'AA Mixed',
    'DA Pure', 'DA Mixed'
]

sizes = [
    AD_pure, AD_total - AD_pure,
    DD_pure, DD_total - DD_pure,
    AA_pure, AA_total - AA_pure,
    DA_pure, DA_total - DA_pure
]

# Normalize to percentages (optional)
grand_total = sum(sizes)
sizes_percent = [v / grand_total * 100 for v in sizes]

# Plot single ring pie chart
plt.figure(figsize=(8,8))
plt.pie(
    sizes_percent,
    labels=labels,
    autopct='%1.1f%%',
    colors=plt.cm.tab20.colors[:8],
    wedgeprops=dict(width=0.4, edgecolor='w')
)
plt.title('Pure vs Mixed Composition')
plt.show()
#%%
import matplotlib.pyplot as plt

# Custom palette
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

# Build full data
labels = [
    'AD Pure', 'AD Mixed',
    'DD Pure', 'DD Mixed',
    'AA Pure', 'AA Mixed',
    'DA Pure', 'DA Mixed'
]

sizes = [
    AD_pure, AD_total - AD_pure,
    DD_pure, DD_total - DD_pure,
    AA_pure, AA_total - AA_pure,
    DA_pure, DA_total - DA_pure
]

# Assign colors: pure gets main color, mixed gets lighter shade
colors = [
    custom_palette['AD'], '#D9B28A', 
    custom_palette['DD'], '#F9DCA7',
    custom_palette['AA'], '#C79CF0',
    custom_palette['DA'], '#DCDCDC'
]

# Normalize sizes to percentages
grand_total = sum(sizes)
sizes_percent = [v / grand_total * 100 for v in sizes]

# Plot pie chart
plt.figure(figsize=(8,8))
plt.pie(
    sizes_percent,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    wedgeprops=dict(width=0.4, edgecolor='w'),
    textprops={'fontsize': 10}
)
plt.title('Pure vs Mixed Composition (All Types)', fontsize=16)
plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Custom palette for the main types
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

# Prepare full data for all types (new order: AD, AA, DD, DA)
data = {
    'AD': [AD_pure, AD_total - AD_pure],
    'AA': [AA_pure, AA_total - AA_pure],
    'DD': [DD_pure, DD_total - DD_pure],
    'DA': [DA_pure, DA_total - DA_pure]
}

labels = ['Pure', 'Mixed']

# Create color mapping for pure/mixed for each type (new order)
colors = [
    [custom_palette['AD'], '#D9B28A'],  # AD
    [custom_palette['AA'], '#C79CF0'],  # AA
    [custom_palette['DD'], '#F9DCA7'],  # DD
    [custom_palette['DA'], '#DCDCDC']   # DA
]

# Plot
fig, axs = plt.subplots(4,1, figsize=(2,2))

for ax, (type_name, values), col in zip(axs, data.items(), colors):
    total = sum(values)
    sizes = [v / total * 100 for v in values]
    ax.pie(
        sizes,
        #labels=labels,              # ✅ outside labels
        autopct='%1.1f%%',          # ✅ inside percentages
        colors=col,
        textprops={'fontsize': 8},
        labeldistance=1.25,         # push Pure/Mixed outside
        pctdistance=0.7             # keep % inside closer to center
    )

    #ax.set_title(f"{type_name} (n={total})", fontsize=8)

pure_patch = mpatches.Patch(facecolor="#4A90E2", label="Pure")   # darker tone
mixed_patch = mpatches.Patch(facecolor="#AED6F1", label="Mixed") # lighter tone

fig.legend(handles=[pure_patch, mixed_patch],
           loc="lower center", ncol=2, fontsize=8, frameon=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0)

plt.savefig(_out_dir / "purity_pies_princeton_bign.svg")

plt.show()
#%%
