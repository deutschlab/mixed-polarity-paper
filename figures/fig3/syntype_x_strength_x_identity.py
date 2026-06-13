import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CONNECTIONS_TABLE_FTR, NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
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
connections=pd.read_feather(CONNECTIONS_TABLE_FTR)
connections=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 
       'sum_syn', 'reciprocal', 
       'same_type']]
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.dropna(subset=['axon_correct','dend_correct','primary_type'])

#%%
nodesG=nodesG[nodesG['super_class'].isin(['optic', 'central', 'visual_projection','visual_centrifugal'])]



nodesG=nodesG[['neuron','SI']]
#%%
connections=connections.merge(nodesG,left_on='pre',right_on='neuron',how='left')
connections=connections.merge(nodesG,left_on='post',right_on='neuron',how='left')
#%%
connections=connections.query("SI_x>=0.1 and SI_y>=0.1")
#%%
connections_same=connections[connections['same_type']==1]
connections_not_same=connections[connections['same_type']!=1]

#%%
ns_AD=connections_not_same[connections_not_same['AD']>=1]
s_AD=connections_same[connections_same['AD']>=1]

ns_AA=connections_not_same[connections_not_same['AA']>=1]
s_AA=connections_same[connections_same['AA']>=1]


ns_DD=connections_not_same[connections_not_same['DD']>=1]

s_DD=connections_same[connections_same['DD']>=1]



ns_DA=connections_not_same[connections_not_same['DA']>=1]

s_DA=connections_same[connections_same['DA']>=1]


#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define the correct order of categories
categories = ['ns_AD', 's_AD', 'ns_AA', 's_AA', 'ns_DD', 's_DD','ns_DA', 's_DA']

# Compute mean and CI for each category
def mean_and_ci(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    ci = 1.96 * std / np.sqrt(n) if n > 0 else 0
    ci=std
    return mean, ci

mean_ci_pairs = [
    mean_and_ci(ns_AD['AD']), mean_and_ci(s_AD['AD']),
    mean_and_ci(ns_AA['AA']), mean_and_ci(s_AA['AA']),
    mean_and_ci(ns_DD['DD']), mean_and_ci(s_DD['DD']),
    mean_and_ci(ns_DA['DA']), mean_and_ci(s_DA['DA'])

]

# Separate means and CIs
means = [pair[0] for pair in mean_ci_pairs]
cis = [pair[1] for pair in mean_ci_pairs]

# Assign colors and hatches
category_colos = [
    custom_palette['AD'], custom_palette['AD'],
    custom_palette['AA'], custom_palette['AA'],
    custom_palette['DD'], custom_palette['DD'],
    custom_palette['DA'], custom_palette['DA']

]
hatch_pattens = ['', 'xxxxx', '', 'xxxxx', '', 'xxxxx','', 'xxxxx']

# Compute count for display
counts = [len(ns_AD), len(s_AD), len(ns_AA), len(s_AA), len(ns_DD), len(s_DD), len(ns_DA), len(s_DA)]
count_labels = [f"n = {int(c/1000)}k" for c in counts]

# Create the plot
plt.figure(figsize=(2.5,2))
lower_errors = np.zeros_like(means)
upper_errors = cis

yerr = [lower_errors, upper_errors]

bars = plt.bar(
    categories,
    means,
    color=category_colos,
    yerr=yerr,
    capsize=0,  # Removes the caps
    error_kw={'elinewidth': 1, 'capthick': 0}  # Makes sure no caps are drawn
)
# Apply hatches
for bar, hatch in zip(bars, hatch_pattens):
    bar.set_hatch(hatch)

# Annotate with sample size
for bar, count_label, ci in zip(bars, count_labels, cis):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.1, count_label,
             ha='center', va='bottom', fontsize=4)

# Manual legend
legend_patches = [
    mpatches.Patch(color=custom_palette['AD'], label='AD',linewidth=0.1),
    mpatches.Patch(color=custom_palette['AA'], label='AA'),
    mpatches.Patch(color=custom_palette['DD'], label='DD'),
    mpatches.Patch(color=custom_palette['DA'], label='DA'),

    mpatches.Patch(facecolor='white', edgecolor='black', hatch='x', label='Same type')
]
plt.yticks([0,4,8,12],size=8)

plt.ylabel('Mean Value with SD')
#plt.title('Mean Values of AD, AA, and DD and DA for Same and Not Same')
plt.xticks(rotation=90)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(np.array(means) + np.array(cis)) + 1)
sns.despine(top=True,right=True)
plt.tight_layout()
#plt.savefig(r'C:\Users\user\organised_work\code\783\article\princeton\3\synapses_per_connection_princeton\syn_type_means_s_ns_princeton.svg')
plt.show()

#%%


# assumes: import numpy as np, import matplotlib.pyplot as plt
# and you already have: ns_AD, s_AD, ns_AA, s_AA, ns_DD, s_DD, ns_DA, s_DA
# custom_palette = {'AA':'#8b2be2','DD':'#F4B95A','AD':'#9F4800','DA':'#B3B3B3'}

def mean_se(a):
    a = np.asarray(a)
    n = len(a)
    if n <= 1:
        return (float(a.mean()) if n else 0.0), 0.0
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(n)
    return m, se

# Build per-type stats: (x=mean_s, xerr=se_s, y=mean_ns, yerr=se_ns)
stats = {}
stats['AD'] = {
    'x': mean_se(s_AD['AD'])[0],  'xerr': mean_se(s_AD['AD'])[1],
    'y': mean_se(ns_AD['AD'])[0], 'yerr': mean_se(ns_AD['AD'])[1]
}
stats['AA'] = {
    'x': mean_se(s_AA['AA'])[0],  'xerr': mean_se(s_AA['AA'])[1],
    'y': mean_se(ns_AA['AA'])[0], 'yerr': mean_se(ns_AA['AA'])[1]
}
stats['DD'] = {
    'x': mean_se(s_DD['DD'])[0],  'xerr': mean_se(s_DD['DD'])[1],
    'y': mean_se(ns_DD['DD'])[0], 'yerr': mean_se(ns_DD['DD'])[1]
}
stats['DA'] = {
    'x': mean_se(s_DA['DA'])[0],  'xerr': mean_se(s_DA['DA'])[1],
    'y': mean_se(ns_DA['DA'])[0], 'yerr': mean_se(ns_DA['DA'])[1]
}

types = ['AD','AA','DD','DA']
xs    = [stats[t]['x'] for t in types]
ys    = [stats[t]['y'] for t in types]
xerr  = [stats[t]['xerr'] for t in types]
yerr  = [stats[t]['yerr'] for t in types]
cols  = [custom_palette[t] for t in types]
plt.figure(figsize=(2.4, 2.2))

# scatter with error bars in both directions
for t, x, y, xe, ye, c in zip(types, xs, ys, xerr, yerr, cols):
    plt.errorbar(
        x, y,
        xerr=xe,       # horizontal error (SE of s)
        yerr=ye,       # vertical error (SE of ns)
        fmt='o', ms=0.0,
        color=c, ecolor=c,
        elinewidth=0.2, capsize=0.7
    )
    # label slightly offset
    plt.text(x + 0.03, y + 0.03, t, fontsize=6, color=c)

# diagonal y=x reference
mn = min(min(xs), min(ys))
mx = max(max(xs), max(ys))
pad = 0.1 * (mx - mn if mx > mn else 1.0)
plt.plot([1,5], [1,5], lw=0.3, ls='--', color='black')

plt.xlabel('same type (mean ± SE)', fontsize=8)
plt.ylabel('not same type (mean ± SE)', fontsize=8)
plt.xticks([1,3, 2, 4,5], fontsize=8)   # only show 0, 2, 4
plt.yticks([1,3, 2, 4,5], fontsize=8)   # only show 0, 2, 4



# optional: clean frame if seaborn is around
try:
    import seaborn as sns
    sns.despine(top=True, right=True)
except Exception:
    pass

plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig3" / "syntype_x_strength_x_identity"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "s_vs_ns_type_scatter_SE.svg")
plt.show()

#%%
aaa=s_DD.groupby(by=['DD']).size()
#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
a=allsynapses.head(1000)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# All synapses
r_all = allsynapses[['pre', 'post']].groupby(by=['pre', 'post']).size()
r_all = pd.DataFrame(r_all, columns=['number of synapses'])
rr_all = r_all.groupby(by=['number of synapses']).size()
rr_all = pd.DataFrame(rr_all, columns=['number of instances'])
rr_all['rolling sum'] = rr_all['number of instances'].cumsum()
rr_all['cdf'] = rr_all['rolling sum'] / rr_all['number of instances'].sum()

# Filtered synapses
allsynapses2 = allsynapses.query("SI_pre >= 0.1 and SI_post >= 0.1")
r_filt = allsynapses2[['pre', 'post']].groupby(by=['pre', 'post']).size()
r_filt = pd.DataFrame(r_filt, columns=['number of synapses'])
rr_filt = r_filt.groupby(by=['number of synapses']).size()
rr_filt = pd.DataFrame(rr_filt, columns=['number of instances'])
rr_filt['rolling sum'] = rr_filt['number of instances'].cumsum()
rr_filt['cdf'] = rr_filt['rolling sum'] / rr_filt['number of instances'].sum()
#%%
# Plot
plt.figure(figsize=(1.3,1))
sns.lineplot(x=rr_all.index, y=rr_all['cdf'], label='All synapses',linewidth=0.25)
sns.lineplot(x=rr_filt.index, y=rr_filt['cdf'], label='Filtered synapses (SI_pre & SI_post ≥ 0.1)',linewidth=0.25)
plt.yticks([0.5,0.6,0.7,0.8,0.9,1],size=4)
plt.tick_params(axis='both', width=0.3)

ax = plt.gca()
ax.spines['left'].set_linewidth(0.3)
ax.spines['bottom'].set_linewidth(0.3)
ax.spines['right'].set_linewidth(0.3)
ax.spines['top'].set_linewidth(0.3)  # Optional


plt.xticks([1,3,5,9,11],size=4)

plt.xlim(1, 11)
plt.xticks(range(1, 12))
plt.xlabel('Number of synapses between pre and post',size=4)
plt.ylabel('Cumulative Distribution Function (CDF)',size=4)
#plt.title('CDF of Synapse Counts Between Neuron Pairs')
plt.legend(fontsize=4)

plt.tight_layout()
plt.savefig(_out_dir / "syn_type_synapses_in_connection_cdf_princeton.svg")

plt.show()


