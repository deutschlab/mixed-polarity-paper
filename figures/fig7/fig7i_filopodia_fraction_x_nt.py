# Fig 7I: per-neuron fraction of boutons with a postsynaptic terminal, by neurotransmitter.
# Ported from the authors' original script fig7i.py.
# Paths come from config.py; every figure is saved before plt.show(); svg.fonttype='none' and Arial (repository convention).
# It writes NEURONS_NT_BWF_FTR, which figures/fig7/filopodia_x_mirror_neurons.py and figures/fig5/reciprocal_fraction_x_bwf.py read.
# Its neuron column is pandas' nullable Int64.
# Its neuron set is every bouton neuron with a neurotransmitter in NEURON_DATA_FTR, all super-classes, no SI filter.
# Each neuron's fraction pools its axonal and dendritic boutons, with no minimum bouton count.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BOUTON_FILOPODIA_TABLE_FTR, BOUTON_TABLE_FTR, NEURONS_NT_BWF_FTR, NEURON_DATA_FTR, OUTPUT_DIR

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "fig7" / "fig7i_filopodia_fraction_x_nt"
_out_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# --------------------------------------------------
# Load data
# --------------------------------------------------
bouton_table = pd.read_feather(
    BOUTON_TABLE_FTR
)

#%%

# bouton_table = b_table.copy()
#%%
# Fill null primary_type with last 9 digits of neuron
bouton_table['primary_type'] = bouton_table['primary_type'].fillna(
    bouton_table['neuron'].astype(str).str[-9:]
)
#%%
bouton_table.dtypes
#%%
neuron_data = pd.read_feather(NEURON_DATA_FTR)
neuron_data = neuron_data.rename(columns={'root_id':'neuron'})
neuron_data = neuron_data[['neuron','nt_type']]
neuron_data.dtypes

#%%
neuron_data[neuron_data['nt_type']=='DA']

#%%
bouton_table = bouton_table.merge(neuron_data, on = 'neuron', how = 'left')
#%%

sum(bouton_table['nt_type'].isna())

#%%


filopodia= pd.read_feather(BOUTON_FILOPODIA_TABLE_FTR)

#%%
# filopodia = bwf_table.copy()
#%%
filopodia.dtypes
#%%
filopodia = filopodia.rename(columns={'count':'fil_count'})
#%%
# merge fil_count onto bouton_table
bouton_table = bouton_table.merge(
    filopodia[['neuron', 'bwf', 'fil_count']],
    left_on=['neuron', 'cluster'],
    right_on=['neuron', 'bwf'],
    how='left'
)

# drop helper column
bouton_table = bouton_table.drop(columns='bwf')

# ensure fil_count is only present when filopodia == True
bouton_table.loc[~bouton_table['filopodia'], 'fil_count'] = None
#%%%
# filopodia True but missing fil_count (should be rare or zero)
missing = bouton_table.loc[
    bouton_table['filopodia'] & bouton_table['fil_count'].isna()
]

print("Missing fil_count rows:", len(missing))

#%%
del filopodia

#%%

bouton_table['nt_type'].unique() 

#%%
import pandas as pd
import seaborn as sns

# ----------------------------------------------------
# 1. Prepare data
# ----------------------------------------------------

# Drop rows without nt_type
df = bouton_table.dropna(subset=['nt_type']).copy()

# Compute filopodia fraction per neuron
neuron_frac = (
    df
    .groupby(['neuron', 'nt_type'], as_index=False)
    .agg(
        filopodia_fraction=('filopodia', 'mean')
    )
)
#%%
neuron_frac.to_feather(NEURONS_NT_BWF_FTR)



#%%
nt_counts = (
    neuron_frac
    .groupby('nt_type')
    .size()
    .sort_values(ascending=False)
)

#%%
# ----------------------------------------------------
# 2. Neurotransmitter colors
# ----------------------------------------------------

nt_colors = {
    "ACH": "#95A3CE",
    "GABA": "#D5A848",
    "GLUT": "#86A859",
    "SER": "#8C6295",
    "DA": "#B87969",
    "OCT": "#725C98",
}

nt_labels = {
    "ACH": "ACH",
    "GABA": "GABA",
    "GLUT": "GLUT",
    "SER": "SER",
    "DA": "DA",
    "OCT": "OCT",
}

# Keep only nt_types we have colors for
neuron_frac = neuron_frac[neuron_frac['nt_type'].isin(nt_colors)]

# ----------------------------------------------------
# 3. Sort nt_types by decreasing mean filopodia fraction
# ----------------------------------------------------

nt_order = (
    neuron_frac
    .groupby('nt_type')['filopodia_fraction']
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)


# ----------------------------------------------------
# 4. Violin plot: filopodia fraction by nt_type
# ----------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

# --- Base violinplot ---
sns.violinplot(
    data=neuron_frac,
    x='nt_type',
    y='filopodia_fraction',
    order=nt_order,
    palette=nt_colors,
    inner=None,          # IMPORTANT: we draw quartiles manually
    linewidth=0,
    cut=0,
    ax=ax
)

# ----------------------------------------------------
# Custom quartile lines + median dots (template-compatible)
# ----------------------------------------------------

order_rest = nt_order
hue_levels = [None]   # single "hue" to preserve template logic

xpos = {cat: i for i, cat in enumerate(order_rest)}
num_hues = len(hue_levels)

violin_width = 0.8
tick_halfwidth = 0.08
median_dot_size = 40

offsets = [
    violin_width * (j - (num_hues - 1) / 2) / num_hues
    for j in range(num_hues)
]

for i, cat in enumerate(order_rest):
    for j, _ in enumerate(hue_levels):

        subset = neuron_frac.loc[
            neuron_frac['nt_type'] == cat,
            'filopodia_fraction'
        ].dropna()

        if subset.empty:
            continue

        q1, med, q3 = np.percentile(subset, [25, 50, 75])
        x = xpos[cat] + offsets[j]

        # Quartile ticks
        ax.hlines(
            q1, x - tick_halfwidth, x + tick_halfwidth,
            color='black', linewidth=1.2, zorder=4
        )
        ax.hlines(
            q3, x - tick_halfwidth, x + tick_halfwidth,
            color='black', linewidth=1.2, zorder=4
        )

        # Median dot
        ax.scatter(
            x, med,
            s=median_dot_size,
            facecolors='white',
            edgecolors='black',
            zorder=5
        )

# ----------------------------------------------------
# Styling
# ----------------------------------------------------

ax.set_xlabel('')
ax.set_ylabel('Filopodia fraction')

ax.set_xticklabels(
    [nt_labels[n] for n in order_rest],
    fontsize=11
)

# minimalist axes (matches your preference)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(_out_dir / 'nt_type_frac_new2.svg', format='svg')
plt.show()
