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
#%%
df=df.dropna(subset=['dend_correct','axon_correct','super_class'
                     ])

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

#%%
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
# Second group: all others
# ====================
df_rest = df_long[df_long['super_class'].isin(group_2)]
order_rest = [sc for sc in group_2 if sc in df_rest['super_class'].unique()]

plt.figure(figsize=(2.8,1.8))
ax = sns.violinplot(
    data=df_rest,
    x='super_class',
    y='correct_value',
    hue='compartment',
    palette=palette,
    order=order_rest,
    inner=None,             # disable default quartile lines
    linewidth=0.4,
    cut=0
)

# Hatching
patch_index = 0
for hue in hue_levels:
    for super_class in order_rest:
        if patch_index >= len(ax.patches):
            break
        if super_class in hatched_classes:
            patch = ax.patches[patch_index]
            patch.set_hatch(hatch_pattern)
            patch.set_edgecolor('white')
            patch.set_linewidth(0)
        patch_index += 1

# --- Custom quartile lines + median dots (aligned) ---
xpos = {cat: i for i, cat in enumerate(order_rest)}
num_hues = len(hue_levels)
tick_halfwidth = 0.08
median_dot_size = 9
violin_width = 0.8  # same as sns.violinplot default; change if you pass a custom width
offsets = [violin_width * (j - (num_hues - 1) / 2) / num_hues for j in range(num_hues)]
for i, cat in enumerate(order_rest):
    for j, hue in enumerate(hue_levels):
        subset = df_rest.loc[
            (df_rest['super_class'] == cat) &
            (df_rest['compartment'] == hue),
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

plt.title("Axon and Dendrite Correct — Non-Visual Super Classes")
plt.xlabel("Super Class")
plt.ylabel("Cannonical Percent")
plt.ylim(0, 100)
plt.legend(title="Compartment")
plt.tight_layout()
sns.despine()
plt.title("")
plt.xlabel("")
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
_out_dir = OUTPUT_DIR / "fig1" / "canonicality_violin_non_intrinsic_sclass"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "Axon_Dend_Sclass_violin_nonvisual.svg")
plt.show()

#%%
# ============================
# Print sample counts for both visual and non-visual groups
# ============================


print("\n=== NON-intrinsic GROUP SAMPLE COUNTS ===")
for cat in order_rest:
    for hue in hue_levels:
        n = df_rest.loc[
            (df_rest['super_class'] == cat) &
            (df_rest['compartment'] == hue),
            'correct_value'
        ].dropna().shape[0]
        print(f"{cat:<22} | {hue:<15} | n = {n}")

#%%
