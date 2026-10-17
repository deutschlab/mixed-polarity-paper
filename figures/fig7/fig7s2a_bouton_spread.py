# Supp 7-S2 A: mean RMS against mean centroid distance of each neuron's boutons.
# Ported from the authors' original script fig7S2a.py.
# Changes: paths via config.py; figure saves switched on and placed before plt.show().
# It plots every neuron of BOUTON_INERTIA_FTR, before any cell type is excluded; it reads CONSOLIDATED_CELL_TYPES_CSV but does not use it.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BOUTON_INERTIA_FTR, CONSOLIDATED_CELL_TYPES_CSV, OUTPUT_DIR

_out_dir = OUTPUT_DIR / "fig7" / "fig7s2a_bouton_spread"
_out_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
neuron_quality=pd.read_feather(BOUTON_INERTIA_FTR)
#%%
neuron_quality.dtypes
#%%
neuron_types = pd.read_csv(CONSOLIDATED_CELL_TYPES_CSV)

#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(
    neuron_quality['mean_rms'],
    neuron_quality['mean_centroid_dist'],
    alpha=0.4
)

ax.set_xlabel('Mean RMS')
ax.set_ylabel('Mean Centroid Distance (Separation)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(_out_dir / 'inertia_all.svg', format='svg')
plt.show()
