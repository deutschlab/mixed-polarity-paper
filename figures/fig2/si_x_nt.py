
from nglui.statebuilder import ChainedStateBuilder

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from fafbseg import flywire
import navis
import pandas as pd

import time
import datetime
import itertools
import os
import pickle
import networkx as nx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import pingouin as pg
import scikit_posthocs as sp
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import f_oneway, levene, mannwhitneyu, shapiro
import random
client = CAVEclient('flywire_fafb_production')
import seaborn as sns
import gc

nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])

nodesG=nodesG.dropna(subset=['nt_type'])
allowed_values = ['optic', 'central', 'visual_projection','visual_centrifugal']

# Filter the DataFrame
nodesG = nodesG[nodesG['super_class'].isin(allowed_values)]

nodesG['nt_type'] = nodesG['nt_type'].replace({
    "acetylcholine": "ACH",
    "gaba": "GABA",
    "glutamate": "GLUT",
    "serotonin": "SER",
    "dopamine": "DA",
    "octopamine": "OCT"
})

plt.rc('axes', titlesize=8)         # Title size
plt.rc('axes', labelsize=8)         # X and Y label size
plt.rc('xtick', labelsize=4)        # X tick label size
plt.rc('ytick', labelsize=8)        # Y tick label size
plt.rc('legend', fontsize=8)        # Legend text size
plt.rc('legend', title_fontsize=8) 

#%% Figure 2 Panel: SI X nt
custom_palette = {
    "ACH": "#95A3CE",
    "GABA": "#D5A848",
    "GLUT": "#86A859",
    "SER": "#8C6295",
    "DA": "#B87969",
    "OCT": "#725C98",
    " ": "white"  # blank group
}

# Define left and right groups explicitly
left_nts = ["ACH", "GABA", "GLUT"]
right_nts = ["SER", "DA", "OCT"]

# Full order with spacer
# Compute median SI for all neurotransmitters
median_sis = nodesG.groupby('nt_type')['SI'].median()

# Sort each group internally
left_sorted = sorted(left_nts, key=lambda x: median_sis.get(x, 0), reverse=True)
right_sorted = sorted(right_nts, key=lambda x: median_sis.get(x, 0), reverse=True)

# Final order with spacer
order = left_sorted + [" "] + right_sorted
# Map palette for plot
palette_for_plot = {k: custom_palette[k] for k in order if k in custom_palette}

# Count per group
group_sizes = nodesG['nt_type'].value_counts()

plt.figure(figsize=(2.7, 2))
ax = sns.violinplot(
    data=nodesG,
    x='nt_type',
    y='SI',
    order=order,
    palette=palette_for_plot,
    cut=0,
    density_norm='width',
    linewidth=0.5,
    inner=None,
    bw_method=0.17
)
ax.set_ylim(0, 1.2)

# --- Outline-only for right group (using tick labels match) ---
from matplotlib.collections import PolyCollection
violin_bodies = [pc for pc in ax.findobj(PolyCollection) if pc.get_paths()]
tick_labels = [lbl.get_text() for lbl in ax.get_xticklabels() if lbl.get_text().strip()]

for body, category in zip(violin_bodies, tick_labels):
    if category in right_nts:
        body.set_facecolor('none')
        body.set_edgecolor(palette_for_plot.get(category, 'black'))
        body.set_linewidth(0.8)

# --- Quartile lines + median dot (no vertical line) ---
xpos = {cat: i for i, cat in enumerate(order)}
tick_halfwidth = 0.08
median_dot_size = 8

for cat in order:
    if cat == " ":
        continue
    y = nodesG.loc[nodesG['nt_type'] == cat, 'SI'].dropna().values
    if y.size == 0:
        continue
    x = xpos[cat]
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    # Vertical line removed
    ax.hlines(q1, x - tick_halfwidth, x + tick_halfwidth, color='black', linewidth=1, zorder=4)
    ax.hlines(q3, x - tick_halfwidth, x + tick_halfwidth, color='black', linewidth=1, zorder=4)
    ax.scatter(x, med, s=median_dot_size, facecolors='white', edgecolors='black', zorder=5)

# --- Counts above violins (skip spacer) ---
for idx, category in enumerate(order):
    if category == " ":
        continue
    max_y = nodesG.loc[nodesG['nt_type'] == category, 'SI'].max()
    count = int(group_sizes.get(category, 0))
    ax.text(idx, max_y + 0.03, f'{count}', ha='center', va='bottom', fontsize=4, color='black')

# Clean up
tick_labels_clean = [lbl if lbl != " " else "" for lbl in order]
plt.xticks(np.arange(len(order)), tick_labels_clean, rotation=90, size=4)
plt.xlabel('Neurotransmitter Type', size=8)
plt.ylabel('SI', size=8)
plt.yticks(np.arange(0, 1.15, 0.25), size=8)
sns.despine(top=True, right=True)
plt.tight_layout()

# Save
_out_dir = OUTPUT_DIR / "fig2" / "SI_x_nt"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_nt_violin_princeton.svg")
plt.show()

# Statistical tests: NT-type SI comparison


# --- Filter only main neurotransmitters ---
main_nt = ["ACH", "GABA", "GLUT"]
df_nt = nodesG[nodesG["nt_type"].isin(main_nt)][["nt_type", "SI"]].dropna()

# --- Normality test (Shapiro per group) ---
norm_p = df_nt.groupby("nt_type")["SI"].apply(lambda x: shapiro(x.sample(min(5000, len(x)), random_state=0)).pvalue)
print("Normality (Shapiro per group):")
print(norm_p)
print(f"Median normality p = {np.median(norm_p):.3e}\n")

# --- Homogeneity of variance ---
lev_p = levene(*(df_nt.loc[df_nt["nt_type"] == g, "SI"] for g in main_nt)).pvalue
print(f"Levene’s test for equal variances: p = {lev_p:.3e}")

# --- If parametric assumptions hold ---
if (lev_p > 0.05) and all(norm_p > 0.05):
    print("\n✅ Assumptions met → Performing one-way ANOVA:")
    f_stat, p_val = f_oneway(*(df_nt.loc[df_nt["nt_type"] == g, "SI"] for g in main_nt))
    print(f"One-way ANOVA: F = {f_stat:.3f}, p = {p_val:.3e}")

    # --- Post-hoc Tukey HSD ---
    tukey = pg.pairwise_tukey(data=df_nt, dv="SI", between="nt_type")
    print("\nPost-hoc Tukey HSD results:")
    print(tukey[["A", "B", "p-tukey", "cohen-d"]])

else:
    # --- Nonparametric path (if not normal or equal variances) ---
    print("\n❌ Nonparametric distributions → Using Kruskal–Wallis test:")
    from scipy.stats import kruskal
    h_stat, p_val = kruskal(*(df_nt.loc[df_nt["nt_type"] == g, "SI"] for g in main_nt))
    print(f"Kruskal–Wallis: H = {h_stat:.3f}, p = {p_val:.3e}")

    # --- Post-hoc Dunn’s test (Bonferroni) ---
    import scikit_posthocs as sp
    posthoc = sp.posthoc_dunn(df_nt, val_col="SI", group_col="nt_type", p_adjust="bonferroni")
    print("\nPost-hoc Dunn test (Bonferroni corrected):")
    print(posthoc)


# Kruskal–Wallis effect size (eta squared)
H = 15070.808
n_total = len(df_nt)
eta_sq = (H - len(main_nt) + 1) / (n_total - len(main_nt))
print(f"η² (eta squared) = {eta_sq:.3f}")

# --- Pairwise effect sizes (rank-biserial correlations) ---
# Using the same data frame 'df_nt'

pairs = list(itertools.combinations(main_nt, 2))
pairwise_effects = []

for a, b in pairs:
    data_a = df_nt.loc[df_nt["nt_type"] == a, "SI"].values
    data_b = df_nt.loc[df_nt["nt_type"] == b, "SI"].values
    U, p = mannwhitneyu(data_a, data_b, alternative="two-sided")
    n1, n2 = len(data_a), len(data_b)
    # rank-biserial correlation r = 1 - (2U)/(n1*n2)
    r = 1 - (2 * U) / (n1 * n2)
    pairwise_effects.append([a, b, r, p])

# Convert to DataFrame
effect_df = pd.DataFrame(pairwise_effects, columns=["Group 1", "Group 2", "Rank-biserial r", "p-value"])
print("\nPairwise effect sizes (rank-biserial correlations):")
print(effect_df)

