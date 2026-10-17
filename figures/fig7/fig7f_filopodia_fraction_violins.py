# Fig 7F (left half: intrinsic super-classes) and Supp 7-S4 C (right half: the other super-classes): per-neuron fraction of boutons with a postsynaptic terminal.
# Ported from the authors' original script fig7f_fig7S4c.py.
# Changes: paths via config.py; figure saves switched on and placed before plt.show().
# Each violin value is one neuron and compartment (the mean of 'filopodia' over its boutons), with no minimum bouton count.
# The 'All' violins use every compartment row with no SI filter; the axon and dendrite violins keep SI >= 0.1.
# Boutons with no compartment are left out of the violins but count in the bouton-level means that order the super-classes.
# Positions 3 and 4 of the non-intrinsic order are typed in ('sensory', 'ascending'); on these tables that is already the computed order.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BOUTON_TABLE_FTR, OUTPUT_DIR

_out_dir = OUTPUT_DIR / "fig7" / "fig7f_filopodia_fraction_violins"
_out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Matplotlib config
# --------------------------------------------------
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

# --------------------------------------------------
# Imports
# --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
#%%
# --------------------------------------------------
# Load data
# --------------------------------------------------
bouton_table = pd.read_feather(
    BOUTON_TABLE_FTR
)

# --------------------------------------------------
# Define intrinsic superclasses
# --------------------------------------------------
intrinsic_superclasses = [
    'central',
    'optic',
    'visual_projection',
    'visual_centrifugal'
]

bouton_table['is_intrinsic'] = bouton_table['super_class'].isin(intrinsic_superclasses)
#%%
bouton_table.dtypes
#%%
# --------------------------------------------------
# Per-neuron filopodia fraction
# --------------------------------------------------
neuron_frac = (
    bouton_table
    .groupby(['neuron', 'super_class', 'comp'], as_index=False)
    .agg(
        filo_frac=('filopodia', 'mean'),
        mean_SI=('SI', 'mean')
    )
)

# --------------------------------------------------
# Order super_classes by "All" filopodia fraction
# --------------------------------------------------
mean_all = (
    bouton_table
    .groupby('super_class')['filopodia']
    .mean()
    .sort_values(ascending=False)
)

intrinsic_order = [sc for sc in mean_all.index if sc in intrinsic_superclasses]
non_intrinsic_order = [sc for sc in mean_all.index if sc not in intrinsic_superclasses]
non_intrinsic_order[3]='sensory'
non_intrinsic_order[4]='ascending'


superclass_order = intrinsic_order + non_intrinsic_order

# --------------------------------------------------
# Build violin data
# --------------------------------------------------
# Intrinsic: 3 conditions
df_intrinsic = []
df_intrinsic.append(
    neuron_frac[
        neuron_frac['super_class'].isin(intrinsic_superclasses)
    ].assign(condition='All')
)
df_intrinsic.append(
    neuron_frac[
        (neuron_frac['super_class'].isin(intrinsic_superclasses)) &
        (neuron_frac['mean_SI'] >= 0.1) &
        (neuron_frac['comp'] == 'A')
    ].assign(condition='SI ≥ 0.1 - Axon')
)
df_intrinsic.append(
    neuron_frac[
        (neuron_frac['super_class'].isin(intrinsic_superclasses)) &
        (neuron_frac['mean_SI'] >= 0.1) &
        (neuron_frac['comp'] == 'D')
    ].assign(condition='SI ≥ 0.1 - Dendrite')
)
df_intrinsic = pd.concat(df_intrinsic, ignore_index=True)

# Non-intrinsic: single "All"
df_non_intrinsic = neuron_frac[
    ~neuron_frac['super_class'].isin(intrinsic_superclasses)
]

# --------------------------------------------------
# Plot config
# --------------------------------------------------
condition_order = [
    'All',
    'SI ≥ 0.1 - Axon',
    'SI ≥ 0.1 - Dendrite'
]

condition_colors = {
    'All': 'blue',
    'SI ≥ 0.1 - Axon': '#8B2BE2',
    'SI ≥ 0.1 - Dendrite': '#F4B95A'
}


# --------------------------------------------------
# Create figure with two subplots
# --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(26, 4.5),
    sharey=True,
    gridspec_kw={'width_ratios': [3, 2]}  # intrinsic usually has more categories
)

violin_width = 1.0
tick_halfwidth = 0.06
median_dot_size = 30

# ==================================================
# LEFT PANEL — INTRINSIC
# ==================================================

sns.violinplot(
    data=df_intrinsic,
    x='super_class',
    y='filo_frac',
    hue='condition',
    order=intrinsic_order,
    hue_order=condition_order,
    palette=condition_colors,
    inner=None,
    cut=0,
    linewidth=0,
    width=violin_width,
    ax=ax1
)

# transparency
for pc in ax1.collections:
    pc.set_alpha(0.8)

# quartiles + medians
xpos1 = {sc: i for i, sc in enumerate(intrinsic_order)}

offsets_intrinsic = {
    'All': -violin_width * 0.33,
    'SI ≥ 0.1 - Axon': 0.0,
    'SI ≥ 0.1 - Dendrite': violin_width * 0.33
}

for sc in intrinsic_order:
    for cond in condition_order:
        subset = df_intrinsic.loc[
            (df_intrinsic['super_class'] == sc) &
            (df_intrinsic['condition'] == cond),
            'filo_frac'
        ].dropna()

        if subset.empty:
            continue

        q1, med, q3 = np.percentile(subset, [25, 50, 75])
        x = xpos1[sc] + offsets_intrinsic[cond]

        ax1.hlines([q1, q3],
                   x - tick_halfwidth,
                   x + tick_halfwidth,
                   color='black',
                   linewidth=1.2,
                   zorder=4)

        ax1.scatter(x, med,
                    s=median_dot_size,
                    facecolors='white',
                    edgecolors='black',
                    zorder=5)

ax1.set_title("Intrinsic Super Classes")
ax1.set_xlabel("")
ax1.set_ylabel("Per-neuron Filopodia Fraction")

for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

# remove auto legend (we'll add global one)
ax1.get_legend().remove()

# ==================================================
# RIGHT PANEL — NON-INTRINSIC
# ==================================================

sns.violinplot(
    data=df_non_intrinsic,
    x='super_class',
    y='filo_frac',
    order=non_intrinsic_order,
    color=condition_colors['All'],
    inner=None,
    cut=0,
    linewidth=0,
    width=0.6,
    ax=ax2
)

for pc in ax2.collections:
    pc.set_alpha(0.8)

xpos2 = {sc: i for i, sc in enumerate(non_intrinsic_order)}

for sc in non_intrinsic_order:
    subset = df_non_intrinsic.loc[
        df_non_intrinsic['super_class'] == sc,
        'filo_frac'
    ].dropna()

    if subset.empty:
        continue

    q1, med, q3 = np.percentile(subset, [25, 50, 75])
    x = xpos2[sc]

    ax2.hlines([q1, q3],
               x - tick_halfwidth,
               x + tick_halfwidth,
               color='black',
               linewidth=1.2,
               zorder=4)

    ax2.scatter(x, med,
                s=median_dot_size,
                facecolors='white',
                edgecolors='black',
                zorder=5)

ax2.set_title("Non-Intrinsic Super Classes")
ax2.set_xlabel("")
ax2.set_ylabel("")

for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)

# ==================================================
# Shared Y-axis range
# ==================================================
ax1.set_ylim(0.0, 1.0)

# ==================================================
# Global legend
# ==================================================
legend_handles = [
    Patch(facecolor=condition_colors[k],
          edgecolor='black',
          label=k,
          alpha=0.8)
    for k in condition_order
]

fig.legend(handles=legend_handles,
           frameon=False,
           loc='upper right')

# ==================================================
# Title
# ==================================================
fig.suptitle(
    "Bouton Filopodia Fractions – Intrinsic vs Non-Intrinsic Super Classes",
    fontsize=16,
    y=0.98
)


plt.tight_layout()
plt.savefig(
    _out_dir / 'super_fractions_split2_new(2).svg',
    format='svg'
)
plt.show()


#%%

summary = (
    bouton_table
    .groupby('is_intrinsic')
    .agg(
        boutons=('cluster', 'size'),
        unique_neurons=('neuron', 'nunique')
    )
    .rename(index={True: 'Intrinsic', False: 'Non-intrinsic'})
)

print(summary)