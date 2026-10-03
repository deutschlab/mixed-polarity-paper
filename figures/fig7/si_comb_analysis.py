# Fig 7J: mean SI with and without the incoming synapses that sit on filopodia.
# Reads the three tables written by figures/fig7/si_sim_baseline.py, si_sim_no_filopodia.py
# and si_sim_sampleout.py; run those first.
# Ported from the authors' original script Fig 7/filapodia_extraction/SI_sim/SI_comb_analysis.py.
# Changes: paths via config.py, unused imports removed, and one crash fixed in the
# "conditions on x-axis" cell: it looked up 'Sample-out' in LABELS, which holds only
# 'Baseline' and 'No filopodia', and raised before drawing. That cell now picks its columns
# by name. It only shows a figure; the saved 7J panel (sim_results.svg) is drawn earlier
# and does not change.
# Also added: svg.fonttype='none' and Arial (repository convention). The original saved SVGs drew text as
# DejaVu Sans glyph paths, so text and spacing differ from his files; plotted values do not.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    METHODS_DIR, NEURON_TABLE_FTR, FILOPODIA_NODES_BASELINE_FTR,
    FILOPODIA_NODES_NO_FILOPODIA_FTR, FILOPODIA_NODES_SAMPLEOUT_FTR, OUTPUT_DIR,
)
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

_out_dir = OUTPUT_DIR / "fig7" / "si_comb_analysis"
_out_dir.mkdir(parents=True, exist_ok=True)
#%%
nodesGb=pd.read_feather(FILOPODIA_NODES_BASELINE_FTR)
nodesGf=pd.read_feather(FILOPODIA_NODES_NO_FILOPODIA_FTR)
nodesGs=pd.read_feather(FILOPODIA_NODES_SAMPLEOUT_FTR)
#%%
nodesGb=nodesGb[['neuron', 'baseline_SI', 'axon_correct', 'dend_correct']]
nodesGb.columns=['neuron', 'baseline_SI', 'baseline_axon_correct', 'baseline_dend_correct']

nodesGf=nodesGf[['neuron', 'no_filopodia_SI', 'axon_correct', 'dend_correct']]
nodesGf.columns=['neuron', 'no_filopodia_SI', 'no_filopodia_axon_correct', 'no_filopodia_dend_correct']


nodesGs=nodesGs[['neuron', 'sampleout_SI', 'axon_correct', 'dend_correct']]
nodesGs.columns=['neuron', 'sampleout_SI', 'sampleout_axon_correct', 'sampleout_dend_correct']

#%%
df=nodesGb.merge(nodesGf,on='neuron',how='left')

#%%

#%%

#%%
df=nodesGb.merge(nodesGf,on='neuron',how='left')
df=df.merge(nodesGs,on='neuron',how='left')
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
df=df.merge(nodesG[['neuron','super_class']])
#%%
df=df[df['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]

#%%

df[['baseline_SI','no_filopodia_SI',  'sampleout_SI',]].mean()
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df[['baseline_SI','no_filopodia_SI']].mean()

#%%
# =========================
# column definitions
# =========================

SI_COLS = [
    'baseline_SI',
    'no_filopodia_SI',
]

AXON_COLS = [
    'baseline_axon_correct',
    'no_filopodia_axon_correct',
]

DEND_COLS = [
    'baseline_dend_correct',
    'no_filopodia_dend_correct',
]

LABELS = ['Baseline', 'No filopodia']


#%%
#%%
# ---- Mean SI (global) with SE ----

means = df[SI_COLS].mean()
ses = df[SI_COLS].std() / np.sqrt(df[SI_COLS].count())

plt.figure(figsize=(4, 3))
plt.bar(LABELS, means, yerr=ses, capsize=4)
plt.ylabel("Mean SI")
plt.title("Mean SI (global)")
plt.tight_layout()
plt.ylim(0.16,0.19)
plt.savefig(_out_dir / "sim_results.svg")
plt.show()
#%%

paired_df = df[['baseline_SI', 'no_filopodia_SI']].dropna()

from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(paired_df['baseline_SI'],
                          paired_df['no_filopodia_SI'])

print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.3e}")
# Drop NaNs just in case
paired_df = df[['baseline_SI', 'no_filopodia_SI']].dropna()

diff = paired_df['baseline_SI'] - paired_df['no_filopodia_SI']

cohen_d = diff.mean() / diff.std(ddof=1)

print(f"Cohen's d (paired) = {cohen_d:.3f}")
#%%
import numpy as np
from scipy import stats

mean_diff = diff.mean()
se_diff = stats.sem(diff)
ci_low, ci_high = stats.t.interval(
    0.95, len(diff)-1, loc=mean_diff, scale=se_diff
)

print(f"Mean difference = {mean_diff:.4f}")
print(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

#%%
# ---- Mean SI per superclass with SE ----

grouped = df.groupby('super_class')

means_sc = grouped[SI_COLS].mean()
ses_sc = grouped[SI_COLS].std() / np.sqrt(grouped[SI_COLS].count())

ax = means_sc.plot(
    kind='bar',
    yerr=ses_sc,
    figsize=(6, 4),
    capsize=4
)

plt.ylabel("Mean SI")
plt.title("Mean SI per superclass")
plt.legend(LABELS, frameon=False)
plt.tight_layout()
plt.show()
#%%
#%%
#%%
#%%
# ============================================================
# AXON vs DENDRITE CORRECTNESS — LINES + SE
# ONE FIGURE • ONE COLUMN • SUBPLOTS PER SUPERCLASS
# ============================================================

groups = list(df.groupby('super_class'))
n_sc = len(groups)

fig, axes = plt.subplots(
    n_sc, 1,
    figsize=(5, 3.2 * n_sc),
    sharex=True,
    sharey=True
)

if n_sc == 1:
    axes = [axes]

x = np.arange(2)  # Axon, Dendrite

for ax, (sc, df_sc) in zip(axes, groups):

    means_axon = df_sc[AXON_COLS].mean()
    ses_axon   = df_sc[AXON_COLS].std() / np.sqrt(df_sc[AXON_COLS].count())

    means_dend = df_sc[DEND_COLS].mean()
    ses_dend   = df_sc[DEND_COLS].std() / np.sqrt(df_sc[DEND_COLS].count())

    for i, label in enumerate(LABELS):
        y = [means_axon[i], means_dend[i]]
        yerr = [ses_axon[i], ses_dend[i]]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker='o',
            linewidth=2,
            capsize=4,
            label=label if ax is axes[0] else None
        )

    ax.set_title(sc)
    ax.set_ylim(0.60, 0.90)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(['Axon', 'Dendrite'])
axes[0].set_ylabel("Mean correctness")

fig.legend(LABELS, frameon=False, loc='upper right',)
#fig.suptitle("Axon vs Dendrite correctness per superclass", y=1.02)

plt.tight_layout()
plt.show()


#%%
#%%
#%%
# ============================================================
# AXON vs DENDRITE AS LINES
# CONDITIONS ON X-AXIS (REORDERED)
# ============================================================
#%%
# ============================================================
# AXON vs DENDRITE AS LINES
# CONDITIONS ON X-AXIS (REORDERED)
# AXON = AA COLOR | DENDRITE = DD COLOR
# ============================================================

# custom palette (fixed)
custom_palette = {
    'AA': '#8B2BE2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

groups = list(df.groupby('super_class'))
n_sc = len(groups)

fig, axes = plt.subplots(
    n_sc, 1,
    figsize=(5, 3.2 * n_sc),
    sharex=True,
    sharey=True
)

if n_sc == 1:
    axes = [axes]

# desired order
ORDER = ['Baseline', 'Sample-out', 'No filopodia']
ORDER_COLS = {'Baseline': 'baseline', 'Sample-out': 'sampleout', 'No filopodia': 'no_filopodia'}
AXON_COLS_ORD = [f'{ORDER_COLS[o]}_axon_correct' for o in ORDER]
DEND_COLS_ORD = [f'{ORDER_COLS[o]}_dend_correct' for o in ORDER]

x = np.arange(len(ORDER))

for ax, (sc, df_sc) in zip(axes, groups):

    means_axon = df_sc[AXON_COLS_ORD].median()
    ses_axon   = df_sc[AXON_COLS_ORD].std() / np.sqrt(df_sc[AXON_COLS_ORD].count())

    means_dend = df_sc[DEND_COLS_ORD].median()
    ses_dend   = df_sc[DEND_COLS_ORD].std() / np.sqrt(df_sc[DEND_COLS_ORD].count())

    # ---- Axon (AA) ----
    ax.errorbar(
        x,
        means_axon.values,
        yerr=ses_axon.values,
        marker='o',
        linewidth=2,
        capsize=4,
        color=custom_palette['AA'],
        label='Axon'
    )

    # ---- Dendrite (DD) ----
    ax.errorbar(
        x,
        means_dend.values,
        yerr=ses_dend.values,
        marker='o',
        linewidth=2,
        capsize=4,
        color=custom_palette['DD'],
        label='Dendrite'
    )

    ax.set_title(sc)
    ax.set_ylim(0.60, 1)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(ORDER)
axes[0].set_ylabel("Mean correctness")

fig.legend(['Axon', 'Dendrite'], frameon=False, loc='upper right')

plt.tight_layout()
plt.show()


#%%
# ============================================================
# AXON vs DENDRITE CORRECTNESS — ONE FIGURE, SUBPLOTS PER SUPERCLASS
# ============================================================

groups = list(df.groupby('super_class'))
n_sc = len(groups)

# layout: 3 columns, as many rows as needed
n_cols = 3
n_rows = int(np.ceil(n_sc / n_cols))

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5 * n_cols, 3.5 * n_rows),
    sharey=True
)

axes = np.array(axes).reshape(-1)  # flatten for easy indexing

x = np.arange(2)  # Axon, Dendrite
width = 0.25

for ax, (sc, df_sc) in zip(axes, groups):

    means_axon = df_sc[AXON_COLS].mean()
    ses_axon   = df_sc[AXON_COLS].std() / np.sqrt(df_sc[AXON_COLS].count())

    means_dend = df_sc[DEND_COLS].mean()
    ses_dend   = df_sc[DEND_COLS].std() / np.sqrt(df_sc[DEND_COLS].count())

    for i, label in enumerate(LABELS):
        ax.bar(
            x + (i - 1) * width,
            [means_axon[i], means_dend[i]],
            yerr=[ses_axon[i], ses_dend[i]],
            width=width,
            capsize=3,
            label=label if ax is axes[0] else None
        )

    ax.set_xticks(x)
    ax.set_xticklabels(['Axon', 'Dendrite'])
    ax.set_ylim(0, 1)
    ax.set_title(sc)

# remove unused axes
for ax in axes[len(groups):]:
    ax.axis('off')

axes[0].set_ylabel("Mean correctness")
fig.legend(LABELS, frameon=False, loc='upper center', ncol=len(LABELS))
fig.suptitle("Axon vs Dendrite correctness per superclass", y=1.02)

plt.tight_layout()
plt.show()
