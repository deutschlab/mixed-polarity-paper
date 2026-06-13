# --- your setup (kept as-is) ---
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import os
import time
import pickle
import navis
import sys
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, PCA_TABLE_FTR

sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- load your data ---
nodesG=pd.read_feather(NEURON_TABLE_FTR)


# keep only the super_classes of interest
nodesG = nodesG[nodesG['super_class'].isin(
    ['central', 'optic', 'visual_projection', 'visual_centrifugal']
)]

# select columns (same as your slice)
nodesG = nodesG[['neuron','super_class','primary_type','length_nm','out_synapses_count',
                 'in_synapses_count','area_nm','size_nm','SI','leafs','out_partners','in_partners']]

# ---------------- PCA to 2 components -> output df: neuron, PC1, PC2 ----------------
# columns never used as features (guard 'side' if missing)
cols_never_features = [c for c in ['neuron', 'super_class', 'primary_type', 'SI', 'side'] if c in nodesG.columns]

# choose feature columns = all others (typically numeric in your slice)
feature_cols = nodesG.columns.difference(cols_never_features)

# drop rows with any NaNs across the feature set
pca_input = nodesG[feature_cols].dropna()
idx = pca_input.index  # remember surviving rows

# standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(pca_input)

# PCA (2 components)
pca = PCA(n_components=2, svd_solver='full')
pcs = pca.fit_transform(scaled)

# build the requested output DataFrame
pca_df = pd.DataFrame({
    'neuron': nodesG.loc[idx, 'neuron'].values,
    'PC1': pcs[:, 0],
    'PC2': pcs[:, 1]
})

print(pca_df.head())
print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)

# ---------------- Feature importance (contributions) for PC1 & PC2 ----------------
# Loadings (eigenvectors): rows=PCs, cols=features (match order to pca_input!)
features = list(pca_input.columns)
loadings_df = pd.DataFrame(pca.components_, index=['PC1','PC2'], columns=features)

# Convert to contributions: square & normalize so each PC sums to 1
contrib = (loadings_df ** 2).T
contrib = contrib.div(contrib.sum(axis=0), axis=1)

# Sort for readability
pc1_contrib = contrib['PC1'].sort_values(ascending=False)
pc2_contrib = contrib['PC2'].sort_values(ascending=False)

# Plot (horizontal bars)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

pc1_contrib.plot(kind='barh', ax=axes[0])
axes[0].invert_yaxis()
axes[0].set_xlabel('Contribution (sums to 1)')
axes[0].set_title(f'PC1 feature contributions\n(var = {pca.explained_variance_ratio_[0]:.2%})')

pc2_contrib.plot(kind='barh', ax=axes[1])
axes[1].invert_yaxis()
axes[1].set_xlabel('Contribution (sums to 1)')
axes[1].set_title(f'PC2 feature contributions\n(var = {pca.explained_variance_ratio_[1]:.2%})')

plt.show()

# (Optional) also attach PCs back to nodesG for later use:
# nodesG.loc[idx, 'PC1'] = pcs[:, 0]
# nodesG.loc[idx, 'PC2'] = pcs[:, 1]

#%%
PCA_TABLE_FTR.parent.mkdir(parents=True, exist_ok=True)
pca_df.to_feather(PCA_TABLE_FTR)