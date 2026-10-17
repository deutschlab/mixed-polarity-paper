# Fig 7E: neurotransmitter pies, 'Neuron presynaptic to bouton' (left, from FILOPODIA_FILTERED_FTR) against
# 'Any presynaptic neuron (control)' (right: every synapse whose presynaptic neuron has boutons).
# Ported from the authors' original script fig7e.py.
# Changes: paths via config.py; figure saves switched on and placed before plt.show().
# Cannot run until SYNAPSES_NO_SELF_FTR (t2.ftr) and FILOPODIA_FILTERED_FTR are supplied; neither is in this repository.
# Both pies count table rows (synapses), not neurons. The left pie keeps every row of FILOPODIA_FILTERED_FTR,
# whatever its presynaptic neuron; the right pie only synapses whose presynaptic neuron is in BOUTON_TABLE_FTR.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BOUTON_TABLE_FTR, FILOPODIA_FILTERED_FTR, NEURON_DATA_FTR, OUTPUT_DIR, SYNAPSES_NO_SELF_FTR

_out_dir = OUTPUT_DIR / "fig7" / "fig7e_nt_presynaptic_pies"
_out_dir.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
neuron_data = pd.read_feather(NEURON_DATA_FTR)
neuron_data['neuron']=neuron_data['root_id']
neuron_data = neuron_data[['neuron','nt_type']]
neuron_data.dtypes
#%%

syn = pd.read_feather(SYNAPSES_NO_SELF_FTR)
#%%
a = syn.head()
bouton_table = pd.read_feather(
    BOUTON_TABLE_FTR
)
#%%
# bouton_table = b_table.copy()
bouton_table.dtypes
#%%
neurons = bouton_table['neuron']
#%%
# 1. Number of unique neurons
n_unique_neurons = neurons.nunique()
print("Number of unique neurons:", n_unique_neurons)

# 2. Keep only rows in syn where 'pre' is in neurons
syn = syn[syn['pre'].isin(neurons)]

#%%
syn = syn.merge(
    neuron_data,
    left_on='pre',
    right_on='neuron',
    how='inner'
)


#%%

filopodia= pd.read_feather(FILOPODIA_FILTERED_FTR)
#%%

filopodia = filopodia.merge(neuron_data, left_on = 'pre',right_on = 'neuron',how = 'left')



#%% Combined pie charts with custom colors and percentages

import matplotlib.pyplot as plt
import numpy as np

# Custom palette
custom_palette = {
   "ACH": "#95A3CE",
   "GABA": "#D5A848",
   "GLUT": "#86A859",
   "SER": "#8C6295",
   "DA": "#B87969",
   "OCT": "#725C98",
}

# --- Prepare data ---
nt_counts1 = syn.loc[syn['nt_type'].notna(), 'nt_type'].value_counts()
nt_percent1 = nt_counts1 / nt_counts1.sum() * 100

nt_counts2 = filopodia.loc[filopodia['nt_type'].notna(), 'nt_type'].value_counts()
nt_percent2 = nt_counts2 / nt_counts2.sum() * 100

# --- Figure with 2 subplots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Main title
fig.suptitle("Distribution of Neurotransmitter type", fontsize=16, y=0.99)
# # --- Pie 1 ---
# colors1 = [custom_palette.get(nt, "#333333") for nt in nt_counts1.index]
# wedges1, _ = axes[0].pie(nt_counts1.values, startangle=90, colors=colors1)
# axes[0].set_title("Any presynaptic neuron (control)")
# axes[0].axis('equal')

# # --- Pie 2 ---
# colors2 = [custom_palette.get(nt, "#333333") for nt in nt_counts2.index]
# wedges2, _ = axes[1].pie(nt_counts2.values, startangle=90, colors=colors2)
# axes[1].set_title("Neuron presynaptic to bouton")
# axes[1].axis('equal')

# --- Pie 1 (LEFT) : Neuron presynaptic to bouton ---
colors_left = [custom_palette.get(nt, "#333333") for nt in nt_counts2.index]
wedges_left, _ = axes[0].pie(nt_counts2.values, startangle=90, colors=colors_left)
axes[0].set_title("Neuron presynaptic to bouton")
axes[0].axis('equal')


# --- Pie 2 (RIGHT) : Any presynaptic neuron (control) ---
colors_right = [custom_palette.get(nt, "#333333") for nt in nt_counts1.index]
wedges_right, _ = axes[1].pie(nt_counts1.values, startangle=90, colors=colors_right)
axes[1].set_title("Any presynaptic neuron (control)")
axes[1].axis('equal')


# --- Legend with percentages ---
legend_labels1 = [f"{nt} ({pct:.1f}%)" for nt, pct in zip(nt_counts1.index, nt_percent1)]
legend_labels2 = [f"{nt} ({pct:.1f}%)" for nt, pct in zip(nt_counts2.index, nt_percent2)]

# Combine labels for legend (unique)
all_nts = list(dict.fromkeys(list(nt_counts1.index) + list(nt_counts2.index)))

# Colors in legend
legend_colors = [custom_palette.get(nt, "#333333") for nt in all_nts]

fig.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
             for color in legend_colors],
    labels=[f"{nt} ({nt_percent2.get(nt, 0):.1f}%, {nt_percent1.get(nt, 0):.1f}%)"
            for nt in all_nts],
    title="nt_type",
    loc='center right',
    frameon=False
)

plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for legend
plt.savefig(_out_dir / 'pie_plot_nt_post_frac_and_bwf_3_newest.svg', format='svg')
plt.show()

