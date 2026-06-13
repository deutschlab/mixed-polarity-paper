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

# Define a custom color palette for the super_class
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}
#%%
df=pd.read_feather(NEURON_TABLE_FTR)
#%%
df=df.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
df['axon_correct']=df['axon_correct']*100
df['dend_correct']=df['dend_correct']*100
#%%

# Create long-form DataFrame for plotting
df_long = pd.melt(
    df,
    id_vars=['neuron', 'super_class'],
    value_vars=['axon_correct', 'dend_correct'],
    var_name='compartment',
    value_name='correct_value'
)

# Define custom color palette for the compartments
custom_palette = {
    'axon_correct': '#92278f',
    'dend_correct': '#fbb042'
}

# Define order of super_classes
order = ['central', 'visual_projection', 'optic', 'visual_centrifugal',
         'ascending', 'descending', 'sensory', 'endocrine', 'motor']



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define groupings
group_1 = ['optic', 'central', 'visual_projection', 'visual_centrifugal']
group_2 = [sc for sc in df_long['super_class'].unique() if sc not in group_1]

# Common settings
palette = ['#92278f', '#fbb042']
hatched_classes = ['ascending', 'descending', 'sensory', 'endocrine', 'motor']
hatch_pattern = '//////'
hue_levels = df_long['compartment'].unique().tolist()

# ====================
# First group: visual types
# ====================
df_vis = df_long[df_long['super_class'].isin(group_1)]
order_vis = ['central','optic','visual_projection','visual_centrifugal']

plt.figure(figsize=(2.8,1.8))
ax = sns.violinplot(
    data=df_vis,
    x='super_class',
    y='correct_value',
    hue='compartment',
    palette=palette,
    order=order_vis,
    inner=None,             # disable default quartile lines
    linewidth=0.4,
    cut=0
)

# Hatching
patch_index = 0
for hue in hue_levels:
    for super_class in order_vis:
        if patch_index >= len(ax.patches):
            break
        if super_class in hatched_classes:
            patch = ax.patches[patch_index]
            patch.set_hatch(hatch_pattern)
            patch.set_edgecolor('white')
            patch.set_linewidth(0)
        patch_index += 1

# --- Custom quartile lines + median dots (aligned) ---
xpos = {cat: i for i, cat in enumerate(order_vis)}
num_hues = len(hue_levels)
tick_halfwidth = 0.08
median_dot_size = 9

# Approximate hue offsets (two violins per category)
violin_width = 0.8  # same as sns.violinplot default; change if you pass a custom width
offsets = [violin_width * (j - (num_hues - 1) / 2) / num_hues for j in range(num_hues)]
for i, cat in enumerate(order_vis):
    for j, hue in enumerate(hue_levels):
        subset = df_vis.loc[
            (df_vis['super_class'] == cat) &
            (df_vis['compartment'] == hue),
            'correct_value'
        ].dropna()
        if subset.empty:
            continue
        q1, med, q3 = np.percentile(subset, [25, 50, 75])
        x = xpos[cat] + offsets[j]
        ax.hlines(q1, x - tick_halfwidth, x + tick_halfwidth,
                  color='black', linewidth=1.2, zorder=4)
        ax.hlines(q3, x - tick_halfwidth, x + tick_halfwidth,
                  color='black', linewidth=1.2, zorder=4)
        ax.scatter(x, med, s=median_dot_size,
                   facecolors='white', edgecolors='black', zorder=5)

plt.title("Axon and Dendrite Correct — Visual Super Classes")
plt.xlabel("Super Class")
plt.ylabel("Cannonical Percent")
plt.ylim(0, 100)
plt.legend(title="Compartment")
plt.tight_layout()
sns.despine()
plt.xlabel("")
plt.title("")
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=4)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('legend', title_fontsize=8)

new_labels = [
    lbl.get_text().replace('_', ' ').title()
    for lbl in ax.get_xticklabels()
]
handles, labels = ax.get_legend_handles_labels()
new_labels_legend = [lbl.replace('_', ' ').title() for lbl in labels]
new_labels_legend[1] = 'Dendrite Correct'
ax.legend(handles, new_labels_legend, title="Compartment")
ax.set_xticklabels(new_labels)
_out_dir = OUTPUT_DIR / "fig1" / "canonicality_violin_intrinsic_sclass"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "Axon_Dend_Sclass_violin_visual.svg")
plt.show()
#%%

#%%
print("=== VISUAL GROUP SAMPLE COUNTS ===")
for cat in order_vis:
    for hue in hue_levels:
        n = df_vis.loc[
            (df_vis['super_class'] == cat) &
            (df_vis['compartment'] == hue),
            'correct_value'
        ].dropna().shape[0]
        print(f"{cat:<22} | {hue:<15} | n = {n}")

#%%



allowed_values = ['optic', 'central', 'visual_projection','visual_centrifugal']

# Filter the DataFrame
df = df[df['super_class'].isin(allowed_values)]
#%%


from scipy.stats import levene

# Define the classes of interest
classes = ['central', 'visual_projection', 'optic', 'visual_centrifugal']

# Create groups for axon_correct
axon_groups = [df[df['super_class'] == c]['axon_correct'].dropna() for c in classes]

# Create groups for dend_correct
dend_groups = [df[df['super_class'] == c]['dend_correct'].dropna() for c in classes]

print([len(g) for g in axon_groups])  # sample sizes per class
print([len(g) for g in dend_groups])  # sample sizes per class
#%%
from scipy.stats import levene

# Levene test for axon_correct
stat_ax, p_ax = levene(*axon_groups)
print("Levene test (axon_correct):", stat_ax, p_ax)

# Levene test for dend_correct
stat_dend, p_dend = levene(*dend_groups)
print("Levene test (dend_correct):", stat_dend, p_dend)
#%%
import pingouin as pg

# Welch ANOVA for axon_correct
welch_ax = pg.welch_anova(dv='axon_correct', between='super_class', data=df)
print("Welch ANOVA (axon_correct):\n", welch_ax)

# Welch ANOVA for dend_correct
welch_dend = pg.welch_anova(dv='dend_correct', between='super_class', data=df)
print("Welch ANOVA (dend_correct):\n", welch_dend)

#%%
import pingouin as pg

# Axon correct pairwise (Welch-style)
posthoc_ax = pg.pairwise_ttests(
    dv='axon_correct',
    between='super_class',
    data=df,
    padjust='bonf',        # multiple comparisons correction (Bonferroni)
    parametric=True,
    effsize='hedges',      # Hedges' g effect size
    correction=False       # use Welch correction (no pooled var)
)
print(posthoc_ax[['A','B','T','dof','p-corr','hedges']])

# Dend correct pairwise (Welch-style)
posthoc_dend = pg.pairwise_ttests(
    dv='dend_correct',
    between='super_class',
    data=df,
    padjust='bonf',
    parametric=True,
    effsize='hedges',
    correction=False
)
print(posthoc_dend[['A','B','T','dof','p-corr','hedges']])
#%%
desc_ax = df.groupby('super_class')['axon_correct'].agg(['mean','std','count'])
desc_dend = df.groupby('super_class')['dend_correct'].agg(['mean','std','count'])

print("Axon Correct:\n", desc_ax.loc[['central','optic','visual_projection','visual_centrifugal']])
print("\nDendrite Correct:\n", desc_dend.loc[['central','optic','visual_projection','visual_centrifugal']])
#%%interaction


import pingouin as pg

# Data must be long-form (you already have df_long)
aov_interaction = pg.mixed_anova(
    dv='correct_value',
    within='compartment',
    between='super_class',
    subject='neuron',
    data=df_long[df_long['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
)
print(aov_interaction)
#%%




#%%rest of sclasses







