import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'  # or 'Helvetica' or any standard system font
from scipy.stats import gaussian_kde
#%%
df=pd.read_feather(NEURON_TABLE_FTR)
df=df.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
df['axon_correct']=df['axon_correct']*100
df['dend_correct']=df['dend_correct']*100
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}
# Classes of interest
classes = ['central', 'visual_projection', 'optic', 'visual_centrifugal']

# Filter data
dfs = df[df['super_class'].isin(classes)]

df_g=dfs['super_class'].value_counts()

# Sample 500 per class (if class has <500, take all)
dfs_sampled = dfs.groupby('super_class', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 500), random_state=42)
)

# Prepare figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Shared axis limits for consistency
xlim = (dfs['dend_correct'].min(), dfs['dend_correct'].max())
ylim = (dfs['axon_correct'].min(), dfs['axon_correct'].max())

# Plot each class separately
for ax, cls in zip(axes, classes):
    sub_sampled = dfs_sampled[dfs_sampled['super_class'] == cls]
    sub_full = dfs[dfs['super_class'] == cls]
    
    # Compute correlation on full class data
    if len(sub_full) > 1:
        r = sub_full[['axon_correct','dend_correct']].corr().iloc[0,1]
        r_text = f"r = {r:.2f}"
    else:
        r_text = "r = NA"
    
    # Scatter
    sns.scatterplot(
        data=sub_sampled,
        x='dend_correct',
        y='axon_correct',
        color=custom_palette[cls],
        alpha=0.7,
        ax=ax
    )
    
    # Reference line
    ax.plot([0, 100], [0, 100], ls='--', c='gray', lw=0.8, zorder=0)

    # Formatting
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')   # ensures x and y have same scale
    ax.set_title(f"{cls}\n{r_text}")
    ax.set_xlabel("Dendrite Correct (%)")
    ax.set_ylabel("Axon Correct (%)")

plt.suptitle("Axon vs Dendrite Correct (500 sample per class)\nCorrelations computed on all data", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
_out_dir = OUTPUT_DIR / "fig1" / "sclass_canonicality_corr"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "dend_axon_correct_corr_per_sclass.svg")
plt.show()
#%%
r=dfs[['super_class']].groupby(by='super_class').size()