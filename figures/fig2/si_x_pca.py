import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import os
import time
import pickle
import navis 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SYNAPSE_TABLE_FTR, SI_UPDATED_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *


#%%
from sklearn.preprocessing import StandardScaler
#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%

#gg=create_links_two_neurons_shared_synapses(720575940619807397,720575940620638335,allsynapses)
#shorthen_and_open_links(gg[0])
#%%
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG=nodesG.drop(columns=['SI'])
all_SI_df=pd.read_feather(SI_UPDATED_FTR)
all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')
#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])


#%%
#nodesG['SI']=(nodesG['axon_correct']+nodesG['dend_correct'])/2
#%%
nodesG=nodesG[['neuron','super_class', 'primary_type','length_nm','out_synapses_count','in_synapses_count',
       'area_nm', 'size_nm', 'SI', 'leafs', 
        'out_partners', 'in_partners']]
#%%
#%%
#visual_projection = nodesG[nodesG['super_class']=='visual_projection']
#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
#%%
feature_names = cols_to_normalize

loadings = pd.DataFrame(
    pca.components_.T,          # transpose so rows = features
    index=feature_names,
    columns=['PC1_weight', 'PC2_weight']
)

# View weights
print(loadings.sort_values('PC1_weight', key=abs, ascending=False))
#%%
pc1_sorted = loadings['PC1_weight'].abs().sort_values(ascending=False).index

# --- Plot PC1 ---
plt.figure(figsize=(8, 6))
plt.barh(pc1_sorted, loadings.loc[pc1_sorted, 'PC1_weight'])
plt.xlabel("Weight")
plt.title("PCA Loadings – PC1")
plt.gca().invert_yaxis()
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig2" / "SI_x_PCA"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "weights.svg")
plt.show()
#%%
# Run PCA without restricting to 1 component
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

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
#%%
print(loadings_df.T)
#%%
# --- Signed L1-normalized loadings (per PC) ---
contrib_signed = loadings_df.T.copy()
contrib_signed = contrib_signed.div(contrib_signed.abs().sum(axis=0), axis=1)  # |sum| per PC = 1

print(contrib_signed)  # features x PCs, signs preserved

# --- Quick plots ---
contrib_signed['PC1'].sort_values().plot(kind='bar', figsize=(6,4), title='Signed contrib: PC1')
plt.axhline(0, color='black', linewidth=0.8); plt.tight_layout();  plt.savefig(_out_dir / "pc1signed_feat_contribute_pc1_v2.svg",bbox_inches='tight');plt.show();

contrib_signed['PC2'].sort_values().plot(kind='bar', figsize=(6,4), title='Signed contrib: PC2')
plt.axhline(0, color='black', linewidth=0.8); plt.tight_layout();plt.savefig(_out_dir / "pc1signed_feat_contribute_pc2_v2.svg",bbox_inches='tight'); plt.show()

#%%

#%%
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

#%%720575940608832324

vp=nodesG[nodesG['super_class']=='visual_projection'][['neuron','PC1','SI','primary_type']]

LC=vp[vp['neuron']==720575940608832324]
#%%
# === Iterate over super_classes and metrics; show figures (no saving, no functions) ===
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

super_classes = ['optic', 'visual_projection', 'central', 'visual_centrifugal']
metrics = ['PC1', 'PC2', 'PC12_score_weighted', 'PC12_radius', 'PC12_whitened_radius']
metrics = ['PC1', 'PC2']

# per-class arrow labels (from your examples) and count threshold
arrow_labels_map = {
    'optic': {
        'black': ['Dm12', 'Pm03', 'Li26'],
        'black': [ 'Li26'],

        'grey' : ['Sm01', 'T4c', 'Mi13'],
        'grey' : [ 'T4c',],

        'count_min': 0
    },
    'visual_projection': {
        'black': ['LT51', 'MeMe_e08', 'MeMe_e02'],
        'black': [ 'LC11'],

        'grey' : ['LPC2', 'LCe03', 'MeTu1'],
        'grey' : ['MeTu1'],

        'count_min': 0
    },
    'central': {
        'black': ['LT51', 'MeMe_e08', 'LPLC4'],
        'grey' : ['LPC2', 'LCe03', 'MeTu1'],
        'count_min': 10
    },
    'visual_centrifugal': {
        'black': ['LT51', 'MeMe_e08', 'LPLC4'],
        'grey' : ['LPC2', 'LCe03', 'MeTu1'],
        'count_min': 0
    }
}
full_info=[]
#%%
# fixed color scaling to match your plots
vmin, vmax, vcenter = -1.5,3, (-1.5+3)/2
cmap = plt.cm.coolwarm
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

for sc in super_classes:
    cfg = arrow_labels_map.get(sc, {'black': [], 'grey': [], 'count_min': 0})
    df_sc = nodesG[nodesG['super_class'] == sc].copy()
    if df_sc.empty:
        print(f"[{sc}] no rows. Skipping.")
        continue

    for metric in metrics:
        if metric not in df_sc.columns:
            print(f"[{sc} | {metric}] column missing. Skipping.")
            continue

        # group (counts over all rows in class; means ignore NaNs)
        summary = df_sc.groupby('primary_type').agg(
            count=('neuron', 'count'),
            mean_SI=('SI', 'mean'),
            mean_metric=(metric, 'mean'),
        ).reset_index()

        # optional min count filter per class
        summary = summary[summary['count'] >= cfg.get('count_min', 0)]
        if summary.empty:
            print(f"[{sc} | {metric}] empty after filtering. Skipping.")
            continue

        # sort by SI and build colors from the chosen metric
        summary_sorted = summary.sort_values(by='mean_SI', ascending=True).reset_index(drop=True)
        summary_sorted['bar_height'] = 0.1
        bar_colors = [tuple(c) for c in cmap(norm(summary_sorted['mean_metric']))]

        # ---- main bars ----
        fig, ax1 = plt.subplots(figsize=(8, 1.5))
        sns.barplot(
            x='primary_type',
            y='bar_height',
            data=summary_sorted,
            palette=bar_colors,
            edgecolor='black',
            ax=ax1,
            linewidth=0.1,
        )

        # axis styling
        ax1.set_ylim(0, 0.1)
        ax1.set_ylabel("")
        ax1.set_yticks([])
        ax1.set_xlabel("", fontsize=4)
        ax1.tick_params(axis='x', labelsize=4, rotation=90)

        # ---- neuron count overlay (log) ----
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.plot(summary_sorted['primary_type'], summary_sorted['count'],
                 linestyle='-', color='black', linewidth=0.3, zorder=10)
        ax2.scatter(summary_sorted['primary_type'], summary_sorted['count'],
                    color='black', s=0.05, edgecolors='black', zorder=11, linewidth=0.05)
        ax2.set_ylabel("Neuron Count (log scale)", fontsize=4)
        ax2.tick_params(axis='y', labelsize=4)

        # ---- SI-only top inset ----
        ax3 = ax1.inset_axes([0, 1.04, 1, 0.20], sharex=ax1)
        ax3.plot(summary_sorted.index, summary_sorted['mean_SI'],
                 color='black', linewidth=0.2, marker='o', markersize=0.2)
        ax3.set_xlim(-0.5, len(summary_sorted) - 0.5)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([0, 0.5, 1])
        ax3.tick_params(axis='y', labelsize=4)
        ax3.set_ylabel("SI", fontsize=4, labelpad=1)
        ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # ---- colorbar for selected metric ----
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(
            sm, ax=ax1, pad=0.02,
            boundaries=np.linspace(-1.5, 3, 51),  # stabilize color interpolation
            extend='both'
        )
        cbar.set_ticks([-1.5,0,1.5,3])   # or use np.arange(-5,6,1) for every integer
        cbar.set_label(f"Mean {metric}", fontsize=4)
        cbar.ax.tick_params(labelsize=4)


        # --- arrows under selected labels ---
        arrow_labels_black = cfg.get('black', [])
        arrow_labels_grey  = cfg.get('grey', [])
        arrow_labels_all = set(arrow_labels_black + arrow_labels_grey)

        arrow_y_pos = -0.06
        for i, ptype in enumerate(summary_sorted['primary_type']):
            if ptype in arrow_labels_all:
                col = bar_colors[i]
                ax1.annotate(
                    '',
                    xy=(i-0.4, arrow_y_pos - 0.25),
                    xytext=(i-0.4, arrow_y_pos - 0.37),
                    xycoords=('data', 'axes fraction'),
                    annotation_clip=False,
                    clip_on=False,
                    arrowprops=dict(
                        arrowstyle='simple',
                        facecolor=col,
                        edgecolor='black',
                        linewidth=0.3,
                        shrinkA=0, shrinkB=0,
                        mutation_scale=7
                    ),
                    zorder=60
                )

        # --- custom numeric x labels with guide lines ---
        labels = [(pt if pt in arrow_labels_all else str(i))
                  for i, pt in enumerate(summary_sorted['primary_type'], start=1)]

        N = len(summary_sorted)
        ax1.set_xticks(range(N))
        ax1.set_xticklabels([])
        ax1.tick_params(axis='x', length=0)

        y_top, y_bottom = -0.08, -0.28
        gap = 0.05
        xax = ax1.get_xaxis_transform()

        for i, lbl in enumerate(labels):
            y_end  = y_top if (i % 2 == 0) else y_bottom
            y_text = y_end - gap
            ax1.plot([i, i], [0.0, y_end],
             transform=xax, clip_on=False, zorder=50,
             linewidth=0.2, color='black')
            ax1.text(i, y_text, lbl,
                     ha='center', va='top', rotation=90,
                     fontsize=3, fontweight='normal',
                     transform=xax, clip_on=False, zorder=60)

        plt.subplots_adjust(bottom=0.36)
        plt.title(f"{sc} — colored by mean {metric}", fontsize=6, y=1.12)
        full_info.append([summary_sorted,[sc,metric] ])

        plt.tight_layout()
        _out_dir_bars = _out_dir / "colored_bars"
        _out_dir_bars.mkdir(parents=True, exist_ok=True)
        fig.savefig(_out_dir_bars / f"ptypes_{sc}_{metric}_princeton_fixed.svg",
                    bbox_inches='tight')
        plt.show()
        #%%
import pandas as pd
rows = []
for item in full_info:
    # each item looks like: [DataFrame, [super_class, metric]]
    df, meta = item
    super_class, metric = meta

    # add index numbers
    df = df.reset_index(drop=True).copy()
    df['index'] = df.index + 1
    df['super_class'] = super_class
    df['metric'] = metric

    # keep only the columns you want
    tmp = df[['index', 'primary_type', 'super_class', 'metric']]
    rows.append(tmp)

final_df = pd.concat(rows, ignore_index=True)
print(final_df.head())

#%%
final_df['super_class']=final_df['super_class'].str.replace("_", " ")
#%%
final_df.to_feather(str(_out_dir / "full_info.ftr"))
#%%
final_df.to_clipboard(index=False)   # copies as tab-separated text
#%%

#%% === Scatter plot of PC1 × PC2 colored by super_class ===
#%% === Distribution of PC1 per super_class ===
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.savefig(_out_dir / "violin_plots_fixed.svg",
                    bbox_inches='tight')
plt.show()
#%%
fig, ax = plt.subplots(figsize=(8.0, 2.8))
sns.kdeplot(
    data=df_pc1,
    x='PC1',hue='super_class',
    palette=custom_palette,fill=False,    common_norm=True,
    linewidth=0.8,
    cut=0,
    ax=ax
)
plt.xlim(-2,5)
plt.savefig(_out_dir / "kde_plots_fixed.svg",
                    bbox_inches='tight')
plt.show()
#%%
#%% === Rolling mean of PC1 vs SI per super_class ===
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% === Rolling mean of PC1 vs SI per super_class + CDF of SI ===
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% === Rolling mean of PC1 vs SI + CDF for grouped super_classes ===
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Clean data
df_roll = nodesG.dropna(subset=['SI', 'PC1', 'super_class']).copy()

# Updated color palette
custom_palette = {
    'visual projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual centrifugal': '#44733B'
}

# Define groupings
groupings = {
    'optic_visual_projection': ['optic', 'visual_projection'],
    'central_visual_centrifugal': ['central', 'visual_centrifugal']
}

# Function to plot grouped classes
def plot_grouped_classes(df, classes, palette, title, save_path):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for sc in classes:
        df_sc = df[df['super_class'] == sc].sort_values('SI')
        if df_sc.empty:
            continue

        # Rolling window: n/10 (min 20 points)
        window = max(int(len(df_sc) / 10), 20)
        df_sc['PC1_roll'] = df_sc['PC1'].rolling(window=window, center=True).mean()

        # --- Plot rolling PC1 mean ---
        sns.lineplot(
            data=df_sc,
            x='SI',
            y='PC1_roll',
            color=palette[sc.replace('_', ' ')],
            linewidth=0.8,
            label=sc.replace('_', ' '),
            ax=ax
        )

        # --- Add CDF on secondary axis ---
        ax2 = ax.twinx()
        sorted_si = np.sort(df_sc['SI'].values)
        cdf = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
        ax2.plot(sorted_si, cdf, color=palette[sc.replace('_', ' ')], linewidth=0.3, alpha=0.5)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelsize=5, width=0.3, colors='gray')
        ax2.set_ylabel('CDF(SI)', fontsize=5, color='gray')

    # --- Formatting ---
    if 'central' in classes or 'visual_centrifugal' in classes:
        ax.set_xlim(0, 0.7)
    else:
        ax.set_xlim(0, 0.5)
    ax.set_xlabel('SI', fontsize=6)
    ax.set_ylabel('Rolling mean PC1', fontsize=6)
    ax.tick_params(axis='both', labelsize=5, width=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)

    ax.legend(fontsize=5, frameon=False)
    ax.set_title(title, fontsize=7)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# --- Create the two figures ---
plot_grouped_classes(
    df_roll,
    ['optic', 'visual_projection'],
    custom_palette,
    title='Optic + Visual Projection',
    save_path=str(_out_dir / "optic_visual_projection_PC1_SI_CDF_v2.svg")
)

plot_grouped_classes(
    df_roll,
    ['central', 'visual_centrifugal'],
    custom_palette,
    title='Central + Visual Centrifugal',
    save_path=str(_out_dir / "central_visual_centrifugal_PC1_SI_CDF_v2.svg")
)
#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('white')
sns.set_context('paper', font_scale=0.8)

# ---------- FIGURE 1: optic + visual projection (stacked vertically) ----------
fig, axes = plt.subplots(2, 1, figsize=(3.2, 4.8), sharex=True)  # stacked vertically

for ax, sc, title, ylim in zip(
    axes,
    ['optic', 'visual_projection'],
    ['Optic', 'Visual Projection'],
    [(-1.5, 3), (0, 5)]  # separate y-limits for each
):
    df_sc = df_roll[df_roll['super_class'] == sc].sort_values('SI')
    if df_sc.empty:
        ax.set_visible(False)
        continue

    # Rolling mean
    window = min(int(len(df_sc) / 30)**0.5, 400)
    window = min(int(np.sqrt(len(df_sc))), 400)
    df_sc['PC1_roll'] = df_sc['PC1'].rolling(window=window, center=True).mean()

    # Rolling PC1 mean
    sns.lineplot(
        data=df_sc,
        x='SI',
        y='PC1_roll',
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.8,
        ax=ax
    )

    # CDF on secondary axis (kept but subtle)
    ax2 = ax.twinx()
    sorted_si = np.sort(df_sc['SI'].values)
    cdf = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
    ax2.plot(sorted_si, cdf,
             color=custom_palette[sc.replace('_', ' ')],
             linewidth=0.3, alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=5, width=0.3, colors='gray')
    ax2.set_ylabel('CDF(SI)', fontsize=5, color='gray')

    # Formatting
    # --- change only here ---
    if sc == 'visual_projection':
        ax.set_xlim(0, 0.75)
    else:
        ax.set_xlim(0, 0.5)
    # -------------------------
    ax.set_ylim(ylim)
    ax.set_ylabel('Rolling mean PC1', fontsize=6)
    ax.tick_params(axis='both', labelsize=5, width=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)
    ax.set_title(title, fontsize=7)

# Shared x-axis label for the bottom plot
axes[-1].set_xlabel('SI', fontsize=6)

plt.tight_layout()
plt.savefig(
    _out_dir / "optic_visual_projection_PC1_SI_CDF_vertical_v2.svg",
    bbox_inches='tight'
)
plt.show()


# ---------- FIGURE 2: central + visual centrifugal (stacked vertically) ----------
fig, axes = plt.subplots(2, 1, figsize=(3.2, 4.8), sharex=True)

for ax, sc, title, ylim in zip(
    axes,
    ['central', 'visual_centrifugal'],
    ['Central', 'Visual Centrifugal'],
    [(0, 5), (-1, 20)]  # different y-limits per subplot
):
    df_sc = df_roll[df_roll['super_class'] == sc].sort_values('SI')
    if df_sc.empty:
        ax.set_visible(False)
        continue

    # Rolling mean
    window = min(int(np.sqrt(len(df_sc))), 400)
    df_sc['PC1_roll'] = df_sc['PC1'].rolling(window=window, center=True).mean()

    sns.lineplot(
        data=df_sc,
        x='SI',
        y='PC1_roll',
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.8,
        ax=ax
    )

    # CDF secondary axis (kept same for consistency)
    ax2 = ax.twinx()
    sorted_si = np.sort(df_sc['SI'].values)
    cdf = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
    ax2.plot(sorted_si, cdf,
             color=custom_palette[sc.replace('_', ' ')],
             linewidth=0.3, alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=5, width=0.3, colors='gray')
    ax2.set_ylabel('CDF(SI)', fontsize=5, color='gray')

    # Formatting
    ax.set_xlim(0, 0.7)
    ax.set_ylim(ylim)
    ax.set_ylabel('Rolling mean PC1', fontsize=6)
    ax.tick_params(axis='both', labelsize=5, width=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)
    ax.set_title(title, fontsize=7)

axes[-1].set_xlabel('SI', fontsize=6)

plt.tight_layout()
plt.savefig(
    _out_dir / "central_visual_centrifugal_PC1_SI_CDF_vertical_v2.svg",
    bbox_inches='tight'
)
plt.show()


#%%with corrs

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('white')
sns.set_context('paper', font_scale=0.8)

# ---------- FIGURE 1: optic + visual projection (stacked vertically) ----------
fig, axes = plt.subplots(2, 1, figsize=(3.2, 4.8), sharex=True)  # stacked vertically

for ax, sc, title, ylim in zip(
    axes,
    ['optic', 'visual_projection'],
    ['Optic', 'Visual Projection'],
    [(-1.1, 2), (-0.2, 3)]  # separate y-limits for each
):
    df_sc = df_roll[df_roll['super_class'] == sc].sort_values('SI')
    if df_sc.empty:
        ax.set_visible(False)
        continue

    # Rolling mean
    window = min(int(len(df_sc) / 30), 400)
    df_sc['PC1_roll'] = df_sc['PC1'].rolling(window=window, center=True).mean()

    # --- Correlation between SI and PC1 ---
    corr = df_sc[['SI', 'PC1']].corr().iloc[0, 1]
    corr_text = f"r = {corr:.2f}"
    ax.text(0.02, 0.9, corr_text, transform=ax.transAxes,
            fontsize=6, color='gray', ha='left', va='center')

    # Rolling PC1 mean
    sns.lineplot(
        data=df_sc,
        x='SI',
        y='PC1_roll',
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.8,
        ax=ax
    )

    # CDF on secondary axis
    ax2 = ax.twinx()
    sorted_si = np.sort(df_sc['SI'].values)
    cdf = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
    ax2.plot(sorted_si, cdf,
             color=custom_palette[sc.replace('_', ' ')],
             linewidth=0.3, alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=5, width=0.3, colors='gray')
    ax2.set_ylabel('CDF(SI)', fontsize=5, color='gray')

    # Formatting
    if sc == 'visual_projection':
        ax.set_xlim(0, 0.75)
    else:
        ax.set_xlim(0, 0.5)
    ax.set_ylim(ylim)
    ax.set_ylabel('Rolling mean PC1', fontsize=6)
    ax.tick_params(axis='both', labelsize=5, width=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)
    ax.set_title(title, fontsize=7)

# Shared x-axis label
axes[-1].set_xlabel('SI', fontsize=6)
axes[-1].set_xlim(0,0.5)

plt.tight_layout()
plt.savefig(
    _out_dir / "optic_visual_projection_PC1_SI_CDF_vertical_v2.svg",
    bbox_inches='tight'
)
plt.show()


# ---------- FIGURE 2: central + visual centrifugal (stacked vertically) ----------
fig, axes = plt.subplots(2, 1, figsize=(3.2, 4.8), sharex=True)

for ax, sc, title, ylim in zip(
    axes,
    ['central', 'visual_centrifugal'],
    ['Central', 'Visual Centrifugal'],
    [(0, 5), (-1, 25)]
):
    df_sc = df_roll[df_roll['super_class'] == sc].sort_values('SI')
    if df_sc.empty:
        ax.set_visible(False)
        continue

    window = min(int(len(df_sc) / 30), 400)
    df_sc['PC1_roll'] = df_sc['PC1'].rolling(window=window, center=True).mean()

    # --- Correlation between SI and PC1 ---
    corr = df_sc[['SI', 'PC1']].corr().iloc[0, 1]
    corr_text = f"r = {corr:.2f}"
    ax.text(0.02, 0.9, corr_text, transform=ax.transAxes,
            fontsize=6, color='gray', ha='left', va='center')

    sns.lineplot(
        data=df_sc,
        x='SI',
        y='PC1_roll',
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.8,
        ax=ax
    )

    ax2 = ax.twinx()
    sorted_si = np.sort(df_sc['SI'].values)
    cdf = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
    ax2.plot(sorted_si, cdf,
             color=custom_palette[sc.replace('_', ' ')],
             linewidth=0.3, alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=5, width=0.3, colors='gray')
    ax2.set_ylabel('CDF(SI)', fontsize=5, color='gray')

    ax.set_xlim(0, 0.7)
    ax.set_ylim(ylim)
    ax.set_ylabel('Rolling mean PC1', fontsize=6)
    ax.tick_params(axis='both', labelsize=5, width=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)
    ax.set_title(title, fontsize=7)

axes[-1].set_xlabel('SI', fontsize=6)
axes[-1].set_xlim(0,0.65)

plt.tight_layout()
plt.savefig(
    _out_dir / "central_visual_centrifugal_PC1_SI_CDF_vertical_v2.svg",
    bbox_inches='tight'
)
plt.show()


#%%
# ---------- FIGURE 1: optic + visual projection (stacked vertically) ----------
fig, axes = plt.subplots(2, 1, figsize=(3.2, 4.8), sharex=True)

for ax, sc, title, ylim in zip(
    axes,
    ['optic', 'visual_projection'],
    ['Optic', 'Visual Projection'],
    [(-1.1, 2), (-0.2, 3)]
):
    df_sc = df_roll[df_roll['super_class'] == sc].sort_values('SI')
    if df_sc.empty:
        ax.set_visible(False)
        continue

    # ---------- Rolling mean + 95% CI (y-axis: PC1) ----------
    window = min(int(len(df_sc) / 30), 400)

    df_sc = df_sc.copy()
    roll = df_sc['PC1'].rolling(window=window, center=True)

    df_sc['PC1_roll'] = roll.mean()
    df_sc['PC1_roll_sd'] = roll.std()
    df_sc['PC1_roll_ci'] = 1.96 * df_sc['PC1_roll_sd'] / np.sqrt(window)

    # optional: smooth CI slightly
    df_sc['PC1_roll_ci'] = (
        df_sc['PC1_roll_ci']
        .rolling(3, center=True)
        .mean()
    )

    # ---------- Correlation ----------
    corr = df_sc[['SI', 'PC1']].corr().iloc[0, 1]
    ax.text(0.02, 0.9, f"r = {corr:.2f}",
            transform=ax.transAxes,
            fontsize=6, color='gray',
            ha='left', va='center')

    # ---------- Mean line ----------
    ax.plot(
        df_sc['SI'],
        df_sc['PC1_roll'],
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.8
    )

    # ---------- 95% CI band ----------
    ax.fill_between(
        df_sc['SI'],
        df_sc['PC1_roll'] - df_sc['PC1_roll_ci'],
        df_sc['PC1_roll'] + df_sc['PC1_roll_ci'],
        color=custom_palette[sc.replace('_', ' ')],
        alpha=0.25,
        linewidth=0
    )

    # ---------- CDF ----------
    ax2 = ax.twinx()
    sorted_si = np.sort(df_sc['SI'].values)
    cdf = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
    ax2.plot(
        sorted_si, cdf,
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.3, alpha=0.5
    )
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=5, width=0.3, colors='gray')
    ax2.set_ylabel('CDF(SI)', fontsize=5, color='gray')

    # ---------- Formatting ----------
    ax.set_xlim(0, 0.75 if sc == 'visual_projection' else 0.5)
    ax.set_ylim(ylim)
    ax.set_ylabel('Rolling mean PC1', fontsize=6)
    ax.tick_params(axis='both', labelsize=5, width=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)

    ax.set_title(title, fontsize=7)

axes[-1].set_xlabel('SI', fontsize=6)
axes[-1].set_xlim(0, 0.5)

plt.tight_layout()
plt.savefig(
    _out_dir / "optic_visual_projection_PC1_SI_CDF_vertical_v2.svg",
    bbox_inches='tight'
)
plt.show()


# ---------- FIGURE 2: central + visual centrifugal (stacked vertically) ----------
fig, axes = plt.subplots(2, 1, figsize=(3.2, 4.8), sharex=True)

for ax, sc, title, ylim in zip(
    axes,
    ['central', 'visual_centrifugal'],
    ['Central', 'Visual Centrifugal'],
    [(0, 5), (-1, 25)]
):
    df_sc = df_roll[df_roll['super_class'] == sc].sort_values('SI')
    if df_sc.empty:
        ax.set_visible(False)
        continue

    # ---------- Rolling mean + 95% CI ----------
    window = min(int(len(df_sc) / 30), 400)

    df_sc = df_sc.copy()
    roll = df_sc['PC1'].rolling(window=window, center=True)

    df_sc['PC1_roll'] = roll.mean()
    df_sc['PC1_roll_sd'] = roll.std()
    df_sc['PC1_roll_ci'] = 1.96 * df_sc['PC1_roll_sd'] / np.sqrt(window)

    df_sc['PC1_roll_ci'] = (
        df_sc['PC1_roll_ci']
        .rolling(3, center=True)
        .mean()
    )

    # ---------- Correlation ----------
    corr = df_sc[['SI', 'PC1']].corr().iloc[0, 1]
    ax.text(0.02, 0.9, f"r = {corr:.2f}",
            transform=ax.transAxes,
            fontsize=6, color='gray',
            ha='left', va='center')

    # ---------- Mean line ----------
    ax.plot(
        df_sc['SI'],
        df_sc['PC1_roll'],
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.8
    )

    # ---------- 95% CI band ----------
    ax.fill_between(
        df_sc['SI'],
        df_sc['PC1_roll'] - df_sc['PC1_roll_ci'],
        df_sc['PC1_roll'] + df_sc['PC1_roll_ci'],
        color=custom_palette[sc.replace('_', ' ')],
        alpha=0.25,
        linewidth=0
    )

    # ---------- CDF ----------
    ax2 = ax.twinx()
    sorted_si = np.sort(df_sc['SI'].values)
    cdf = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
    ax2.plot(
        sorted_si, cdf,
        color=custom_palette[sc.replace('_', ' ')],
        linewidth=0.3, alpha=0.5
    )
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=5, width=0.3, colors='gray')
    ax2.set_ylabel('CDF(SI)', fontsize=5, color='gray')

    # ---------- Formatting ----------
    ax.set_xlim(0, 0.7)
    ax.set_ylim(ylim)
    ax.set_ylabel('Rolling mean PC1', fontsize=6)
    ax.tick_params(axis='both', labelsize=5, width=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)

    ax.set_title(title, fontsize=7)

axes[-1].set_xlabel('SI', fontsize=6)
axes[-1].set_xlim(0, 0.65)

plt.tight_layout()
plt.savefig(
    _out_dir / "central_visual_centrifugal_PC1_SI_CDF_vertical_v2.svg",
    bbox_inches='tight'
)
plt.show()

#%%
nodesG[nodesG['neuron']==720575940644651208][['SI','primary_type']]