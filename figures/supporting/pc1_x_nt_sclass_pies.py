# Supporting numbers, not a panel: neurotransmitter and super-class make-up of neurons with PC1 < 0.5 vs >= 0.5
# (SI >= 0.1). Backs the text's "simple neurons (PC1 < 0.5), which are primarily optic neurons".
# Ported from Amit's Fig3/Fig_3/syntype X PC1/PC1_ber_nt_sclass/PC1_ber_nt_sclass.py.
# Changes: paths via config.py and the repository preamble (which also sets editable Arial text in the SVGs);
# Amit's colour module and binomial_methods imports removed (nothing from them is used); nothing else.
# Uses the neuron table's SI column for SI >= 0.1. In the delivered table that is the older SI; a neuron table
# rebuilt by step 05 holds the corrected SI (SI_updated.ftr), which selects a different set of neurons.
# The neurotransmitter is nt_type (the FlyWire prediction); neurons without one are left out of those pies.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, PCA_TABLE_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "supporting/pc1_x_nt_sclass_pies"
_out_dir.mkdir(parents=True, exist_ok=True)
#%%
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import os
import time
import pickle
import navis 
import sys
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
from methods_all import *
custom_pallete_nt = {
    "ACH": "#95A3CE",
    "GABA": "#D5A848",
    "GLUT": "#86A859",
    "SER": "#8C6295",
    "DA": "#B87969",
    "OCT": "#725C98",
}
custom_palette_sclass  = { "ascending": "#6EB6F6", "visual centrifugal": "#44733B", "descending": "#803D3D", "endocrine": "#8973B2", "motor": "#B48667", "optic": "#F4D826", "sensory": "#848484", "central": "#F9574E", "visual projection": "#D5A848", }



#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%

#%%
nodesG=nodesG[['neuron','super_class','SI','nt_type' ]]
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]

#%%

pcadf=pd.read_feather(
    PCA_TABLE_FTR
)
#%%

nodesG=nodesG.merge(pcadf,on='neuron',how='left')
#%%




# nicer labels for display/colors
nodesG['super_class_plot'] = nodesG['super_class'].str.replace('_', ' ', regex=False)

def autopct_with_counts(values):
    total = sum(values)
    def inner(pct):
        count = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n(n={count})' if count > 0 else ''
    return inner

def pie_from_series(ax, series, title, palette=None):
    counts = series.value_counts()

    labels = counts.index.tolist()
    values = counts.values

    if palette is not None:
        colors = [palette.get(x, '#CCCCCC') for x in labels]
    else:
        colors = None

    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=autopct_with_counts(values),
        startangle=90,
        counterclock=False,
        textprops={'fontsize': 10}
    )
    ax.set_title(title, fontsize=12)
    ax.axis('equal')

def plot_4_pies(df, fig_title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # PC1 < 0.5
    df_low = df[df['PC1'] < 0.5]
    pie_from_series(
        axes[0, 0],
        df_low['nt_type'],
        f'PC1 < 0.5\nnt_type (n={len(df_low)})',
        palette=custom_pallete_nt
    )
    pie_from_series(
        axes[0, 1],
        df_low['super_class_plot'],
        f'PC1 < 0.5\nsuper_class (n={len(df_low)})',
        palette=custom_palette_sclass
    )

    # PC1 >= 0.5
    df_high = df[df['PC1'] >= 0.5]
    pie_from_series(
        axes[1, 0],
        df_high['nt_type'],
        f'PC1 >= 0.5\nnt_type (n={len(df_high)})',
        palette=custom_pallete_nt
    )
    pie_from_series(
        axes[1, 1],
        df_high['super_class_plot'],
        f'PC1 >= 0.5\nsuper_class (n={len(df_high)})',
        palette=custom_palette_sclass
    )

    fig.suptitle(fig_title, fontsize=16)
    plt.tight_layout()
    plt.savefig(_out_dir / "pie_nt_pc.svg")

    plt.show()

# with SI >= 0.1
nodesG_si = nodesG[nodesG['SI'] >= 0.1].copy()
plot_4_pies(nodesG_si, 'Distribution by PC1 groups | SI >= 0.1')
# without SI filter
#plot_4_pies(nodesG.copy(), 'Distribution by PC1 groups | no SI filter')


