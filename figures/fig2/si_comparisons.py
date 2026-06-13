
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SYNAPSE_TABLE_FTR, NEURON_TABLE_NONP_FTR, SYNAPSE_TABLE_NONP_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
from math import ceil
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
import pickle
#%%

nodesP=pd.read_feather(NEURON_TABLE_FTR)
nodesG=pd.read_feather(NEURON_TABLE_NONP_FTR)
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class'])

#%%

# --- Keep only needed cols ---
nodesP = nodesP[['neuron','SI','super_class']].rename(columns={'SI':'SI_princeton'})
nodesG = nodesG[['neuron','SI','super_class']]

# --- Merge ---
df = pd.merge(nodesP, nodesG, on=['neuron','super_class'], how='inner')
#%%
# --- Overall correlation ---
r_all = df[['SI_princeton','SI']].corr().iloc[0,1]
print(f"Overall correlation: {r_all:.3f}")

# --- Per-class correlations, sorted ---
corrs = (
    df.groupby('super_class')
      .apply(lambda g: g['SI_princeton'].corr(g['SI']))
      .sort_values()
)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 0.4*len(corrs)+2))
corrs.plot(kind='barh', ax=ax, xlim=(0,1), title='Correlation by super_class')
ax.set_xlabel('Pearson r')

# Annotate r values
for i, (cls, val) in enumerate(corrs.items()):
    ax.text(val + 0.02, i, f"{val:.2f}", va='center')

plt.tight_layout()
plt.show()
    #%%
    
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt

# --- Combined binned scatter ---
bin_w = 0.05
bins = np.arange(0, 1 + bin_w, bin_w)

df['bin'] = pd.cut(df['SI_princeton'], bins=bins, include_lowest=True)
binned = df.groupby('bin').agg(
    x_center=('SI_princeton', 'mean'),
    y_mean=('SI', 'mean'),
    count=('SI', 'size')
).dropna()

plt.figure(figsize=(6,4))
sizes = 100 * binned['count'] / binned['count'].max()
plt.scatter(binned['x_center'], binned['y_mean'], s=sizes, alpha=0.7)
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlim(0,1); plt.ylim(0,1)
plt.xlabel('SI_princeton (binned 0.05)')
plt.ylabel('Mean SI')
r_val = round(df['SI_princeton'].corr(df['SI']),2)

plt.title(f'Binned SI (overall) r = {r_val}')
plt.grid(True, ls=':', lw=0.5)
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig2" / "SI_comparisons"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_corr_all.svg")

plt.show()

    
    
    #%%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd

# --- Bin setup ---
bin_w = 0.05
bins = np.arange(0, 1 + bin_w, bin_w)

custom_palette = {
    "ascending": "#6EB6F6",
    "visual_centrifugal": "#44733B",
    "descending": "#803D3D",
    "endocrine": "#8973B2",
    "motor": "#B48667",
    "optic": "#F4D826",
    "sensory": "#848484",
    "central": "#F9574E",
    "visual_projection": "#D5A848",
}

# --- Desired order ---
upper4 = ["central", "visual_projection", "optic", "visual_centrifugal"]
all_classes = sorted(df['super_class'].unique())
rest = sorted([c for c in all_classes if c not in upper4])
ordered = upper4 + rest  # 9 total
# --- Figure & GridSpec ---
fig = plt.figure(figsize=(15, 7))  # made slightly taller
gs = GridSpec(
    2, 20, figure=fig,
    wspace=0.9,   # more horizontal space
    hspace=0.5    # more vertical space
)

axes = {}

# Top row: 4 plots spanning 5 cols each
col_width = 20 // 4
for i, cls in enumerate(upper4):
    axes[cls] = fig.add_subplot(gs[0, i*col_width:(i+1)*col_width])

# Bottom row: 5 plots spanning 4 cols each
col_width = 20 // 5
for j, cls in enumerate(rest):
    axes[cls] = fig.add_subplot(gs[1, j*col_width:(j+1)*col_width])

# --- Plotting loop unchanged ---
for superclass, ax in axes.items():
    sub = df[df['super_class'] == superclass].copy()
    r_val = sub['SI_princeton'].corr(sub['SI'])

    sub['bin'] = pd.cut(sub['SI_princeton'], bins=bins, include_lowest=True)
    binned = sub.groupby('bin', dropna=True).agg(
        x_center=('SI_princeton', 'mean'),
        y_mean=('SI', 'mean'),
        count=('SI', 'size')
    ).dropna()

    sizes = (100 * binned['count'] / binned['count'].max()).values if len(binned) else [30]
    color = custom_palette.get(superclass, 'tab:blue')

    ax.scatter(binned['x_center'], binned['y_mean'], s=sizes, alpha=0.7, color=color)
    ax.plot([0, 1], [0, 1], '--', lw=1, color='black')

    title_r = "N/A" if r_val is None or np.isnan(r_val) else f"{r_val:.2f}"
    ax.set_title(f"{superclass}\nr={title_r}", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('SI_princeton (binned 0.05)')
    ax.set_ylabel('Mean SI')

sns.despine(top=True, right=True)
fig.suptitle('Binned SI by super_class (titles show RAW r)', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title
plt.savefig(_out_dir / "SI_corr_per_sclass.svg")
plt.show()

    
    
    
    
#%%

#%%


#%%



sns.kdeplot(nodesG['SI'],label='old')
sns.kdeplot(nodesP['SI_princeton'],label='princeton')
plt.legend()
plt.show()

        #%%
syn_dfp=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_NONP_FTR)

#%%

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
#%%
import sys
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# ------------------------------------------------------------
# --- Config -------------------------------------------------
# ------------------------------------------------------------
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

# ------------------------------------------------------------
# --- Load data ----------------------------------------------
# ------------------------------------------------------------
nodesP = pd.read_feather(NEURON_TABLE_FTR)
nodesG = pd.read_feather(NEURON_TABLE_NONP_FTR)
nodesG = nodesG.dropna(subset=['dend_correct', 'axon_correct', 'super_class'])

nodesP = nodesP[['neuron', 'SI', 'super_class']].rename(columns={'SI': 'SI_princeton'})
nodesG = nodesG[['neuron', 'SI', 'super_class']]
df = pd.merge(nodesP, nodesG, on=['neuron', 'super_class'], how='inner')
#%%
# ------------------------------------------------------------
# --- Helper: consistent size scaling ------------------------
# ------------------------------------------------------------
def size_from_count(count, ref_max=40000, min_area=8, max_area=300, use_sqrt=True):
    """Convert neuron counts to marker area (pt²) consistently."""
    x = np.asarray(count, dtype=float)
    if use_sqrt:
        x = np.sqrt(x)
        ref = np.sqrt(ref_max)
    else:
        ref = float(ref_max)
    area = (x / ref) * max_area
    return np.clip(area, min_area, max_area)

# ------------------------------------------------------------
# --- Setup --------------------------------------------------
# ------------------------------------------------------------
bin_w = 0.05
bins = np.arange(0, 1 + bin_w, bin_w)

custom_palette = {
    "ascending": "#6EB6F6",
    "visual_centrifugal": "#44733B",
    "descending": "#803D3D",
    "endocrine": "#8973B2",
    "motor": "#B48667",
    "optic": "#F4D826",
    "sensory": "#848484",
    "central": "#F9574E",
    "visual_projection": "#D5A848",
}

upper4 = ["central", "visual_projection", "optic", "visual_centrifugal"]
all_classes = sorted(df['super_class'].unique())
rest = sorted([c for c in all_classes if c not in upper4])
ordered = upper4 + rest

ref_max = max(40000, df.groupby(pd.cut(df['SI_princeton'], bins=bins)).size().max())

# ------------------------------------------------------------
# --- Figure layout: horizontal layout ------------------------
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 4.2))  # wide and short
gs = GridSpec(2, 22, figure=fig, wspace=0.6, hspace=0.4)

# Left: overall correlation panel
ax_general = fig.add_subplot(gs[:, :5])

# Right: per-super_class panels
axes = {}
col_width_top = 17 // 4
for i, cls in enumerate(upper4):
    axes[cls] = fig.add_subplot(gs[0, 5 + i * col_width_top:5 + (i + 1) * col_width_top])

col_width_bottom = 17 // 5
for j, cls in enumerate(rest):
    axes[cls] = fig.add_subplot(gs[1, 5 + j * col_width_bottom:5 + (j + 1) * col_width_bottom])

# ------------------------------------------------------------
# --- (A) General panel (left) -------------------------------
# ------------------------------------------------------------
df['bin'] = pd.cut(df['SI_princeton'], bins=bins, include_lowest=True)
binned_all = df.groupby('bin').agg(
    x_center=('SI_princeton', 'mean'),
    y_mean=('SI', 'mean'),
    count=('SI', 'size')
).dropna()

sizes_all = size_from_count(binned_all['count'], ref_max=ref_max)
ax_general.scatter(
    binned_all['x_center'],
    binned_all['y_mean'],
    s=sizes_all,
    alpha=0.7,
    color='#4A90E2'  # blue for general panel
)
ax_general.plot([0, 1], [0, 1], '--', lw=1, color='black')
ax_general.set_xlim(0, 1)
ax_general.set_ylim(0, 1)
ax_general.set_xlabel('Segregation Index (SI_princeton)')
ax_general.set_ylabel('Mean SI')
ax_general.set_xticks([0, 0.25, 0.5, 0.75, 1])
r_all = round(df['SI_princeton'].corr(df['SI']), 2)
ax_general.set_title(f'Overall correlation\nr = {r_all}', fontsize=11)

# ------------------------------------------------------------
# --- (B) Per-super_class panels (right) ---------------------
# ------------------------------------------------------------
for i, (superclass, ax) in enumerate(axes.items()):
    sub = df[df['super_class'] == superclass].copy()
    r_val = sub['SI_princeton'].corr(sub['SI'])

    sub['bin'] = pd.cut(sub['SI_princeton'], bins=bins, include_lowest=True)
    binned = sub.groupby('bin', dropna=True).agg(
        x_center=('SI_princeton', 'mean'),
        y_mean=('SI', 'mean'),
        count=('SI', 'size')
    ).dropna()

    sizes = size_from_count(binned['count'].values, ref_max=ref_max) if len(binned) else [30]
    color = custom_palette.get(superclass, 'tab:blue')

    ax.scatter(binned['x_center'], binned['y_mean'], s=sizes, alpha=0.7, color=color)
    ax.plot([0, 1], [0, 1], '--', lw=1, color='black')

    title_r = "N/A" if r_val is None or np.isnan(r_val) else f"{r_val:.2f}"
    ax.set_title(f"{superclass}\nr={title_r}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    # Remove y labels/ticks for all but leftmost in each row
    if (i not in [0, 4]):  # keep first of each row
        ax.set_yticklabels([])
        ax.set_ylabel('')
        ax.tick_params(axis='y', length=0)

    # Remove x labels for all except bottom row
    if superclass in upper4:
        ax.set_xticklabels([])

# ------------------------------------------------------------
# --- Shared x label for all panels --------------------------
# ------------------------------------------------------------
fig.text(0.6, 0.04, 'Segregation Index (SI_princeton)', ha='center', fontsize=10)

# ------------------------------------------------------------
# --- Shared legend (absolute neuron counts, extended) -------
# ------------------------------------------------------------
legend_counts = [100, 500, 1000, 5000, 10000, 20000, 40000]
handles = [
    plt.scatter([], [], s=size_from_count(c, ref_max=ref_max),
                color='gray', alpha=0.5, label=f'{c:,} neurons')
    for c in legend_counts
]
fig.legend(
    handles=handles,
    title='Bin count (neurons)',
    frameon=False,
    fontsize=7,
    title_fontsize=7,
    loc='upper left',
    bbox_to_anchor=(0.06, 0.97)
)

# ------------------------------------------------------------
# --- Final layout -------------------------------------------
# ------------------------------------------------------------
sns.despine(top=True, right=True)
fig.suptitle('SI vs. SI_princeton: Overall and per super_class', y=0.995, fontsize=11)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # leave space for shared x label
plt.savefig(
    _out_dir / "SI_corr_combined_general_left_blue_clean.svg",
    format='svg'
)
plt.show()
