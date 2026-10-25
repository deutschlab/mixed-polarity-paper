# Fig 7K: share of postsynaptic terminals on boutons that are 'hand shake', other reciprocal, or not reciprocal,
# by super-class of the neuron that owns the bouton (the postsynaptic neuron of these synapses).
# Ported from the authors' original script fig7k.py.
# Paths come from config.py; every figure is saved before plt.show(); svg.fonttype='none' and Arial (repository convention).
# Reads super_class from NEURON_TABLE_FTR. The original read a copy of that table whose super_class is the same for all 138,289 neurons (only its nt_type column differs, and it is not used).
# The bars pool terminals (synapses) across neurons and connections.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FILOPODIA_RECIPROCAL_FTR, NEURON_TABLE_FTR, OUTPUT_DIR

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "fig7" / "fig7k_reciprocal_composition"
_out_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%%
filopodia= pd.read_feather(FILOPODIA_RECIPROCAL_FTR)

#%%
neuron_data = pd.read_feather(NEURON_TABLE_FTR)
neuron_data['post']= neuron_data['neuron']
neuron_data= neuron_data[['post','super_class']]
filopodia = filopodia.merge(neuron_data, on='post',how='left')
#%%
filopodia['post'].nunique()
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = filopodia.copy()

# --------------------------------------------------
# 1. Classify reciprocal category
# --------------------------------------------------
df["recip_category"] = np.select(
    [
        df["local_reciprocal"],
        df["global_reciprocal"] & ~df["local_reciprocal"]
    ],
    [
        "Local reciprocal",
        "Global reciprocal only"
    ],
    default="Non-reciprocal"
)

# --------------------------------------------------
# 2. Count per super_class
# --------------------------------------------------
counts = (
    df
    .groupby(["super_class", "recip_category"])
    .size()
    .unstack(fill_value=0)
)

# Ensure all expected columns exist (important!)
for col in ["Local reciprocal", "Global reciprocal only", "Non-reciprocal"]:
    if col not in counts.columns:
        counts[col] = 0

counts = counts[
    ["Local reciprocal", "Global reciprocal only", "Non-reciprocal"]
]

# --------------------------------------------------
# 3. Convert to percentages
# --------------------------------------------------
percentages = counts.div(counts.sum(axis=1), axis=0) * 100
# --------------------------------------------------
# 3. Sort by GLOBAL reciprocal fraction (descending)
# --------------------------------------------------
percentages["global_fraction"] = (
    percentages["Local reciprocal"] +
    percentages["Global reciprocal only"]
)

percentages = percentages.sort_values("global_fraction", ascending=False)

percentages = percentages.drop(columns=["global_fraction"])
# --------------------------------------------------
# 4. Plot stacked bar chart
# --------------------------------------------------
#%%
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(percentages))
bottom = np.zeros(len(percentages))

colors = {
    "Local reciprocal": "#4CAF50",        # green
    "Global reciprocal only": "#FFC107",  # yellow/amber
    "Non-reciprocal": "#B0BEC5"           # gray
}

legend_labels = {
    "Local reciprocal": "'Hand Shake'",
    "Global reciprocal only": "Reciprocal not 'Hand Shake'",
    "Non-reciprocal": "Non Reciprocal"
}



for category in percentages.columns:
    ax.bar(
        x,
        percentages[category].values,
        bottom=bottom,
        label=legend_labels[category],
        color=colors[category]
    )
    bottom += percentages[category].values

# --------------------------------------------------
# 5. Styling
# --------------------------------------------------
ax.set_xticks(x)
ax.set_xticklabels(percentages.index, rotation=45, ha="right")

ax.set_ylabel("Percentage of synapses (%)")
ax.set_ylim(0, 100)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=True)

plt.title("Reciprocal composition by super class")
plt.tight_layout()

# IMPORTANT: save before show (avoids empty file issues)
plt.savefig(
    _out_dir / 'super_classes_reciprocal_new.svg',
    format="svg"
)

plt.show()