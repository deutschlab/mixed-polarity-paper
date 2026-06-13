import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import os
import time
import pickle
import navis 
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SYNAPSE_TABLE_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
nodesG = pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
nodesG=nodesG[['neuron','super_class', 'primary_type','length_nm','out_synapses_count','in_synapses_count',
       'area_nm', 'size_nm', 'SI', 'leafs', 
        'out_partners', 'in_partners']]

# Define columns to exclude from PCA
cols_to_keep = ['neuron', 'super_class', 'primary_type', 'SI', 'side']
cols_to_normalize = nodesG.columns.difference(cols_to_keep)

# Drop rows with NaNs (only these rows will get PCs)
pca_input = nodesG[cols_to_normalize].dropna()

# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_input)

# PCA -> 2 components (PC1, PC2)
pca = PCA(n_components=2, svd_solver='full')
pcs = pca.fit_transform(scaled_data)

# Store PCs back on nodesG
nodesG.loc[pca_input.index, 'PC1'] = pcs[:, 0]
nodesG.loc[pca_input.index, 'PC2'] = pcs[:, 1]

# (Optional) keep the scaled values:
nodesG.loc[pca_input.index, cols_to_normalize] = scaled_data

# (Optional) check variance explained
print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)


#%% Figure 2 Panel: PCA feature loadings
# Run PCA without restricting to 1 component
# --- PCA ---
pca_full = PCA()
pca_full.fit(scaled_data)

# Eigenvalues & explained variance
print("Eigenvalues:", pca_full.explained_variance_)
print("Explained variance ratio:", pca_full.explained_variance_ratio_)

# Loadings DataFrame
loadings = pca_full.components_
loadings_df = pd.DataFrame(
    loadings,
    columns=cols_to_normalize,
    index=[f"PC{i+1}" for i in range(loadings.shape[0])]
)

print(loadings_df.T)

#%% Figure 2 Panel: signed PCA feature contributions
# --- Signed L1-normalized loadings (per PC) ---
contrib_signed = loadings_df.T.copy()
contrib_signed = contrib_signed.div(contrib_signed.abs().sum(axis=0), axis=1)  # |sum| per PC = 1

print(contrib_signed)  # features x PCs, signs preserved

# --- Quick plots ---
contrib_signed['PC1'].sort_values().plot(kind='bar', figsize=(6,4), title='Signed contrib: PC1')
_out_dir = OUTPUT_DIR / "fig2" / "pca_on_feat"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.axhline(0, color='black', linewidth=0.8); plt.tight_layout();  plt.savefig(_out_dir / "pc1signed_feat_contribute_pc1.svg",bbox_inches='tight');plt.show();

contrib_signed['PC2'].sort_values().plot(kind='bar', figsize=(6,4), title='Signed contrib: PC2')
plt.axhline(0, color='black', linewidth=0.8); plt.tight_layout();plt.savefig(_out_dir / "pc1signed_feat_contribute_pc2.svg",bbox_inches='tight'); plt.show()


# Use only rows where both PCs exist
mask = nodesG['PC1'].notna() & nodesG['PC2'].notna()

# 1) Variance-weighted score (good default)
#    Gives PC2 some influence proportional to how much variance it explains.
w1, w2 = pca.explained_variance_ratio_[:2]
nodesG.loc[mask, 'PC12_score_weighted'] = w1 * nodesG.loc[mask, 'PC1'] + w2 * nodesG.loc[mask, 'PC2']

# 2) Euclidean magnitude in PC space (radial size)
#    Larger if either PC is large; dominated by PC1 when PC1 explains much more variance.
nodesG.loc[mask, 'PC12_radius'] = ((nodesG.loc[mask, 'PC1']**2 + nodesG.loc[mask, 'PC2']**2) ** 0.5)

# 3) “Whitened” magnitude (balances PCs by their variance)
#    Useful if you don’t want PC1 to dominate just because it has higher variance.
e1, e2 = pca.explained_variance_[:2]
nodesG.loc[mask, 'PC12_whitened_radius'] = (((nodesG.loc[mask, 'PC1']**2) / e1) + ((nodesG.loc[mask, 'PC2']**2) / e2)) ** 0.5


#%% === Distribution of PC1 per super_class ===
# Filter valid PC1 values
df_pc1 = nodesG.loc[nodesG['PC1'].notna(), ['PC1', 'super_class']].copy()

# Consistent color palette (matching your style)
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(2.0, 1.4))
sns.violinplot(
    data=df_pc1,
    x='PC1',
    y='super_class',
    palette=custom_palette,
    linewidth=0.3,
    inner='box',
    scale='width',
    cut=0,
    ax=ax
)
plt.xlim(-3,15)
# --- Styling ---
ax.set_xlabel('PC1', fontsize=6)
ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=5, width=0.3)
ax.tick_params(axis='y', labelsize=5, width=0.3)
for spine in ax.spines.values():
    spine.set_linewidth(0.3)

sns.despine(ax=ax, left=True, bottom=False)
plt.tight_layout()
plt.savefig(_out_dir / "violin_plots.svg",
                    bbox_inches='tight')
plt.show()

