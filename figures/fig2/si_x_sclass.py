import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import scikit_posthocs as sp
from scipy.stats import gaussian_kde
from scipy.stats import f_oneway, kruskal, levene, shapiro
import matplotlib as mpl
from matplotlib.collections import PolyCollection
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
# Data
nodesG = pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class'])

# Define the super_classes where primary_type MUST be present
target_classes = ["visual_centrifugal", "central", "visual_projection", "optic"]

# Drop NaN primary_type only within those classes
nodesG = nodesG[~(
    nodesG['super_class'].isin(target_classes) & 
    nodesG['primary_type'].isna()
)]

nodesG[nodesG['neuron']==720575940642312136]['SI']
#syn_df = pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\article\synapses_783_article.ftr')

#%%
# ------- Fig 1e Super_class X SI -------

# Stats & sorted order (by mean, descending as in your script)
summary_stats = (
    nodesG.groupby('super_class')['SI']
    .agg(mean='mean', std='std', count='count')
    .assign(se=lambda df: df['std'] / np.sqrt(df['count']))
    .reset_index()
    .sort_values(by='mean', ascending=False)
)
# --- split into left vs right groups ---
left_classes  = ["visual_centrifugal", "central", "visual_projection", "optic"]
right_classes = [c for c in summary_stats['super_class'] if c not in left_classes]

# Sort each block by mean SI (descending)
summary_means = summary_stats.set_index('super_class')['mean']
left_sorted  = sorted(left_classes,  key=lambda c: summary_means[c], reverse=True)
right_sorted = sorted(right_classes, key=lambda c: summary_means[c], reverse=True)

# Final order = left block first, then right block
order = left_sorted + [" "] + right_sorted

# Your explicit palette by super_class (names must match your column values exactly)
custom_palette = {
    "ascending": "#6EB6F6",
    "visual_centrifugal": "#44733B",
    "descending": "#803D3D",
    "endocrine": "#8973B2"," ":'white',
    "motor": "#B48667",
    "optic": "#F4D826",
    "sensory": "#848484",
    "central": "#F9574E",
    "visual_projection": "#D5A848",'sensory_ascending':'black'
}

# Build the palette *for this plot* from the dict (avoids order/position errors)
palette_for_plot = {cls: custom_palette[cls] for cls in order if cls in custom_palette}

group_sizes = nodesG['super_class'].value_counts()

plt.figure(figsize=(2.7, 2))
ax = sns.violinplot(
    data=nodesG,
    x='super_class',
    y='SI',
    order=order,
    palette=palette_for_plot,
    cut=0,
    density_norm='width',     # instead of scale
    linewidth=0.5,
    inner=None,               # no default marks; we add custom
    bw_method=0.17            # new style for bandwidth
)
ax.set_ylim(0, 1.2)

# --- Right group outline-only (match by tick labels, not PolyCollection order) ---
violin_bodies = [pc for pc in ax.findobj(PolyCollection) if pc.get_paths()]
tick_labels = [lbl.get_text() for lbl in ax.get_xticklabels() if lbl.get_text().strip()]

for body, category in zip(violin_bodies, tick_labels):
    if category in right_sorted:  # outline-only group
        body.set_facecolor('none')
        body.set_edgecolor(palette_for_plot.get(category, 'black'))
        body.set_linewidth(0.8)

# --- Add quartile lines + median dot (both groups, skip spacer) ---
xpos = {cat: i for i, cat in enumerate(order)}
tick_halfwidth = 0.08
median_dot_size = 8

for cat in order:
    if cat == " ":
        continue
    y = nodesG.loc[nodesG['super_class'] == cat, 'SI'].dropna().values
    if y.size == 0:
        continue
    x = xpos[cat]
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    ymin, ymax = y.min(), y.max()

    # quartile ticks
    ax.hlines(q1, x - tick_halfwidth, x + tick_halfwidth, color='black', linewidth=1, zorder=4)
    ax.hlines(q3, x - tick_halfwidth, x + tick_halfwidth, color='black', linewidth=1, zorder=4)
    # median white dot
    ax.scatter(x, med, s=median_dot_size, facecolors='white', edgecolors='black', zorder=5)

# --- Counts above violins (skip spacer) ---
for idx, category in enumerate(order):
    if category == " ":
        continue
    max_y = nodesG.loc[nodesG['super_class'] == category, 'SI'].max()
    count = int(group_sizes.get(category, 0))
    ax.text(idx, max_y + 0.015, f'{count}',
            ha='center', va='bottom', fontsize=4, color='black')

# --- Clean up xticks (hide spacer) ---
tick_labels_clean = [lbl if lbl != " " else "" for lbl in order]
plt.xticks(np.arange(len(order)), tick_labels_clean, rotation=90, size=4)

plt.xlabel('Super Class', size=8)
plt.ylabel('SI', size=8)
plt.title("")
plt.yticks(np.arange(0, 1.15, 0.25), size=8)
sns.despine(top=True, right=True)
plt.tight_layout()

_out_dir = OUTPUT_DIR / "fig2" / "SI_x_sclass"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_sclass_violin_princeton.svg")
plt.show()

# Filter only the 4 main superclasses
main_classes = ['visual_centrifugal', 'optic', 'central', 'visual_projection']
anova_df = nodesG[nodesG['super_class'].isin(main_classes)][['super_class', 'SI']].dropna()

# Normality and homogeneity tests
normality_pvals = {}
for sc, group in anova_df.groupby('super_class'):
    if len(group['SI']) >= 3:
        stat, p = shapiro(group['SI'].sample(min(5000, len(group['SI'])), random_state=0))
        normality_pvals[sc] = p

group_values = [g['SI'].values for _, g in anova_df.groupby('super_class')]
levene_stat, levene_p = levene(*group_values)

print("Homogeneity (Levene) p =", levene_p)
print("Normality (median group p) =", np.median(list(normality_pvals.values())))

# Since likely non-normal, do Kruskal–Wallis
stat, p = kruskal(*group_values)
print(f"Kruskal–Wallis: H = {stat:.3f}, p = {p:.3e}")

# Post hoc Dunn test
if p < 0.05:
    posthoc = sp.posthoc_dunn(anova_df, val_col='SI', group_col='super_class', p_adjust='bonferroni')
    print(posthoc)
