
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SI_UPDATED_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import copy

nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.drop(columns=['SI'])
all_SI_df=pd.read_feather(SI_UPDATED_FTR)
all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
allowed_values = ['optic', 'central',
                  'visual_projection', 'visual_centrifugal']
nodesG = nodesG[nodesG['super_class'].isin(allowed_values)]

nodesG=nodesG.drop(columns='SI')

all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})

nodesG=nodesG.merge(all_SI_df,on='neuron',how='left')


df_f = nodesG[['super_class',
               'length_nm', 'area_nm', 'size_nm',
               'out_synapses_count', 'in_synapses_count',
               'out_partners', 'in_partners', 'leafs',
               'SI']]

df_tmp = df_f.replace([np.inf, -np.inf], pd.NA)

# 2. Rows that will be removed (have at least one NA after replacement)
mask_bad = df_tmp.isna().any(axis=1)

# 3. Save the original rows that will be removed
rows_removed = df_f[mask_bad].copy()        # full rows
removed_indices = rows_removed.index.tolist()  # just the indices if you want

# 4. Now actually clean df_f as you intended
df_f = df_tmp.dropna()

df_f = df_f.replace([float('inf'), -float('inf')], pd.NA).dropna()


df_f_f = copy.deepcopy(df_f)

df_all = df_f_f.drop(columns='super_class')

#%% Figure 2: feat x si x sclass corr

# Step 1: Drop 'SI' and show the clustermap
df_all_no_SI = df_all.drop('SI', axis=1)

# Step 1: Create clustermap to get ordering
cluster_obj = sns.clustermap(df_all_no_SI.corr(), cmap='coolwarm', annot=True)
plt.close()
# Step 2: Extract feature order
ordered_features = df_all_no_SI.columns[cluster_obj.dendrogram_col.reordered_ind]
# Apply only to columns with string data

# Extract dendrogram-based feature order

# Step 2: Compute feature–SI correlations per superclass
correlation_results = {}
for super_class in df_f_f['super_class'].unique():
    df_subset = df_f_f[df_f_f['super_class'] == super_class].drop(columns='super_class')
    correlations = df_subset.corr()['SI']
    correlation_results[super_class] = correlations

# Add correlation for all data

# Convert to DataFrame
correlation_df = pd.DataFrame(correlation_results).T
correlation_df.index.name = 'super_class'
correlation_df.reset_index(inplace=True)

# Remove 'SI' column (optional)
correlation_df = correlation_df.drop(columns='SI')

# Step 3: Melt into long format
correlation_long = correlation_df.melt(
    id_vars='super_class',
    var_name='feature',
    value_name='correlation'
)

# Reorder features by dendrogram order
correlation_long['feature'] = pd.Categorical(
    correlation_long['feature'],
    categories=ordered_features,
    ordered=True
)

# Step 4: Plot the barplot
custom_palette = {
    'visual projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual centrifugal': '#44733B',
    'all': '#7F7F7F'
}

# Step 3: Set up figure with barplot on the left, heatmap on the right
# --------------------
# Font size settings
# --------------------
title_size = 8
label_size = 8
tick_size = 8
legend_size = 8
annot_size = 4  # for heatmap numbers

# --------------------
# Figure and layout
# --------------------
fig = plt.figure(figsize=(3.6, 2.))  # Smaller figure as you said
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.05)

# --------------------
# LEFT: Horizontal Barplot
# --------------------
ax0 = fig.add_subplot(gs[0])

# Set feature order (reversed for alignment with heatmap)
correlation_long['feature'] = pd.Categorical(
    correlation_long['feature'],
    categories=ordered_features[::-1],  # top to bottom
    ordered=True
)
# Apply only to columns with string data
for col in correlation_long.columns:
    if correlation_long[col].astype(str).str.contains('_').any():
        correlation_long[col] = correlation_long[col].astype(str).str.replace('_', ' ', regex=False)

sns.barplot(
    data=correlation_long,
    y='feature',
    x='correlation',
    hue='super_class',
    palette=custom_palette,
    dodge=True,
    errorbar=None,
    linewidth=0.75,
    ax=ax0,
    orient='h'
)
or_feat=ordered_features[::-1]
or_feat=pd.DataFrame(or_feat)
or_feat.columns=['feat']
or_feat['feat']=or_feat['feat'].astype(str).str.replace('_', ' ', regex=False)
or_feat=or_feat.squeeze()
# Style barplot
ax0.set_title('Feature Correlation with SI by Superclass', fontsize=title_size)
ax0.set_xlabel('Correlation', fontsize=label_size)
ax0.set_ylabel('')
ax0.set_yticks(range(len(ordered_features)))
ax0.set_yticklabels(or_feat, fontsize=tick_size)
ax0.tick_params(axis='x', labelsize=tick_size)
ax0.tick_params(axis='y', labelsize=tick_size)
ax0.xaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

# Legend
ax0.legend(title='Super Class', fontsize=legend_size, title_fontsize=legend_size, loc='upper right')

sns.despine(ax=ax0, left=True, top=True, right=True)

# --------------------
# RIGHT: Heatmap
# --------------------
ax1 = fig.add_subplot(gs[1])
corr = df_all_no_SI[ordered_features[::-1]].corr()

# Create mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle (keep lower)

# Plot heatmap showing only lower triangle + diagonal
heatmap = sns.heatmap(
    corr,
    mask=mask,                  # 🔹 hides upper triangle
    cmap='Blues',
    annot=True,
    annot_kws={"size": annot_size},
    cbar=True,
    cbar_kws={"shrink": 0.7, "orientation": "vertical", "label": "Correlation"},
    ax=ax1,
    square=True                 # keeps cells square
)

# Get the colorbar object
cbar = heatmap.collections[0].colorbar

# ✅ Change tick label size
cbar.ax.tick_params(labelsize=4)

# ✅ Change tick formatting (e.g., 2 decimal places)
cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# ✅ Change the colorbar label font size
cbar.set_label("Correlation", fontsize=4)

# Optional: set specific ticks on the colorbar
cbar.set_ticks([0.6,0.7, 0.8,0.9, 1.0])

# Style the heatmap axes
#ax1.set_title("Feature Correlation Matrix", fontsize=title_size)
ax1.tick_params(axis='x', labelsize=4, rotation=90,size=4)
ax0.tick_params(axis='y', labelsize=tick_size, rotation=0)
ax1.set_yticklabels([])

plt.tight_layout()
ax0.set_xticks([-0.4, -0.2, 0, 0.2, 0.4],labels=[-0.4, -0.2, 0, 0.2, 0.4])

# --------------------
# Save
# --------------------
_out_dir = OUTPUT_DIR / "fig2" / "feat_x_si_x_sclass_corr"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "combined_barplot_cluster_matrix_princeton_v2.svg")
plt.show()
