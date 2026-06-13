# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:19:44 2025

@author: user
"""

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.patches as mpatches

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FULL_RECI_CONNECTIONS_FTR, CONNECTIONS_TABLE_FTR, NEURON_TABLE_FTR,
    OUTPUT_DIR, METHODS_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

from scipy.stats import gaussian_kde


#%%
final_df=pd.read_feather(FULL_RECI_CONNECTIONS_FTR)

connections=pd.read_feather(CONNECTIONS_TABLE_FTR)

#%%
connections=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 
       'sum_syn', 'reciprocal', 
       'same_type']]


#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%

#%%
connections=connections.merge(nodesG[['root_id','SI','super_class']],left_on='pre',right_on='root_id',how='left')
connections=connections.merge(nodesG[['root_id','SI','super_class']],left_on='post',right_on='root_id',how='left')
#%%

#%%
connections = connections[connections['super_class_x'].isin(['central', 'optic', 'visual_centrifugal', 'visual_projection'])]
connections = connections[connections['super_class_y'].isin(['central', 'optic', 'visual_centrifugal', 'visual_projection'])]

connections=connections[(connections['SI_x']>=0.1)&(connections['SI_y']>=0.1)]


#%%
AD_r=connections[(connections['AD']>=1)&(connections['reciprocal']==1)]
AD_nr=connections[(connections['AD']>=1)&(connections['reciprocal']==0)]


AA_r=connections[(connections['AA']>=1)&(connections['reciprocal']==1)]
AA_nr=connections[(connections['AA']>=1)&(connections['reciprocal']==0)]

DD_r=connections[(connections['DD']>=1)&(connections['reciprocal']==1)]
DD_nr=connections[(connections['DD']>=1)&(connections['reciprocal']==0)]

DA_r=connections[(connections['DA']>=1)&(connections['reciprocal']==1)]

DA_nr=connections[(connections['DA']>=1)&(connections['reciprocal']==0)]
#%%
import matplotlib.pyplot as plt

# Define your palette
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

# Calculate medians
medians = {
    'AD': {
        'r': AD_r['AD'].mean(),
        'nr': AD_nr['AD'].mean()
    },
    'AA': {
        'r': AA_r['AA'].mean(),
        'nr': AA_nr['AA'].mean()
    },
    'DD': {
        'r': DD_r['DD'].mean(),
        'nr': DD_nr['DD'].mean()
    },
    'DA': {
        'r': DA_r['DA'].mean(),
        'nr': DA_nr['DA'].mean()
    }
}

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# X positions: 0 for 'r', 1 for 'nr'
x_positions = {'r': 0, 'nr': 1}

# Plot each synapse type
for syn_type in ['AD', 'AA', 'DD', 'DA']:
    y_values = [medians[syn_type]['r'], medians[syn_type]['nr']]
    x_vals = [x_positions['r'], x_positions['nr']]
    ax.plot(
        x_vals, 
        y_values, 
        marker='o', 
        label=syn_type, 
        color=custom_palette[syn_type]
    )

# Customize plot
ax.set_xticks([0, 1])
ax.set_yticks(np.array(range(0,10)))

ax.set_xticklabels(['r', 'nr'])
ax.set_ylabel('Median value')
ax.set_title('Median Synapse Strength by Reciprocity')
ax.legend(title='Synapse Type')

plt.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np

# Define your palette
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

# Create filtered datasets for same_type=1 and same_type=0
# (Filtering AD_r, AD_nr, etc., into same_type=1 and 0 groups)
filtered_data = {}

for syn_type in ['AD', 'AA', 'DD', 'DA']:
    filtered_data[syn_type] = {
        1: {  # same_type = 1
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 1)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 1)][syn_type].mean()
        },
        0: {  # same_type = 0
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 0)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 0)][syn_type].mean()
        }
    }

# Create the figure
fig, ax = plt.subplots(figsize=(10, 7))

# X positions
x_positions = {'r': 0, 'nr': 1}

# Plot each synapse type with same_type = 0 and 1 separately
for syn_type in ['AD', 'AA', 'DD', 'DA']:
    for same_type_val in [1, 0]:
        means = [filtered_data[syn_type][same_type_val]['r'], filtered_data[syn_type][same_type_val]['nr']]
        x_vals = [x_positions['r'], x_positions['nr']]
        
        # Line style: solid for same_type=1, dashed for same_type=0
        linestyle = '-' if same_type_val == 1 else '--'
        
        ax.plot(
            x_vals,
            means,
            marker='o',
            label=f"{syn_type} (same_type={same_type_val})",
            color=custom_palette[syn_type],
            linestyle=linestyle
        )

# Customize plot
ax.set_xticks([0, 1])
ax.set_xticklabels(['r', 'nr'])
ax.set_yticks(np.array(range(0, 10)))
ax.set_ylabel('Mean value')
ax.set_title('Mean Synapse Strength by Reciprocity and Same Type')
ax.legend(title='Synapse Type + Same Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np

# Define your palette
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

# Create filtered datasets for same_type=1 and same_type=0
filtered_data = {}

for syn_type in ['AD', 'AA', 'DD', 'DA']:
    filtered_data[syn_type] = {
        1: {  # same_type = 1
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 1)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 1)][syn_type].mean()
        },
        0: {  # same_type = 0
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 0)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 0)][syn_type].mean()
        }
    }

# Create side-by-side figures
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# X positions
x_positions = {'r': 0, 'nr': 1}

# Titles for each panel
same_type_titles = {0: 'Not same Type', 1: 'Same Type'}

# Loop: Not same type first (0), then Same type (1)
for idx, same_type_val in enumerate([0, 1]):
    ax = axes[idx]
    
    for syn_type in ['AD', 'AA', 'DD', 'DA']:
        means = [filtered_data[syn_type][same_type_val]['r'], filtered_data[syn_type][same_type_val]['nr']]
        x_vals = [x_positions['r'], x_positions['nr']]
        
        ax.plot(
            x_vals,
            means,
            marker='o',
            label=syn_type,
            color=custom_palette[syn_type]
        )
    
    # Customize individual axes
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['r', 'nr'])
    ax.set_yticks(np.array(range(1, 10)))  # start y-ticks from 1
    ax.set_title(same_type_titles[same_type_val])
    
    # Remove X label
    # (no ax.set_xlabel here)
    
    # Remove upper and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if idx == 0:
        ax.set_ylabel('Mean value')
    ax.legend(title='Synapse Type')

# Main title
fig.suptitle('Mean Synapse synaptic type and reciprocity', fontsize=16)

plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig5" / "syn_type_strength_identity"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "synapses_reci_syntype_ptype_princeton.svg")
plt.show()
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Define your palette
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

# Brighten the palette
def brighten_color(color, amount=0.35):
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return c + (white - c) * amount

custom_palette_dark = {k: brighten_color(v) for k, v in custom_palette.items()}

# Prepare data
filtered_data = {}
for syn_type in ['AD', 'AA', 'DD', 'DA']:
    filtered_data[syn_type] = {
        1: {
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 1)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 1)][syn_type].mean()
        },
        0: {
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 0)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 0)][syn_type].mean()
        }
    }

# Create figure
fig, axes = plt.subplots(
    2, 1, figsize=(1, 3),
    gridspec_kw={'height_ratios': [9, 9]},
    sharex=True
)

x_positions = {'nr': 0, 'r': 1}
same_type_titles = {0: 'Not same Type', 1: 'Same Type'}

for idx, same_type_val in enumerate([0, 1]):
    ax = axes[idx]

    for syn_type in ['AD', 'AA', 'DD', 'DA']:
        means = [
            filtered_data[syn_type][same_type_val]['nr'],
            filtered_data[syn_type][same_type_val]['r']
        ]
        x_vals = [x_positions['nr'], x_positions['r']]

        color = custom_palette[syn_type] if same_type_val == 0 else custom_palette_dark[syn_type]

        ax.plot(
            x_vals,
            means,
            marker='o',
            label=syn_type,
            color=color,
            linewidth=1,
            markersize=1
        )

    # Axis formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['nr', 'r'], fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    if same_type_val == 0:
        ax.set_ylim(0, 11)
        ax.set_yticks(np.arange(0, 12, 2))
    else:
        ax.set_ylim(0, 4)
        ax.set_yticks(np.arange(0, 5))

    ax.set_title(same_type_titles[same_type_val], fontsize=8)
    ax.set_ylabel('Mean value', fontsize=8)

    # Add top-left synapse labels (no counts)
    x_text = -0.1
    y_start = ax.get_ylim()[1] - 0.5
    y_spacing = 0.45

    for i, syn_type in enumerate(['AD', 'AA', 'DD', 'DA']):
        color = custom_palette[syn_type] if same_type_val == 0 else custom_palette_dark[syn_type]
        ax.text(
            x_text,
            y_start - i * y_spacing,
            syn_type,
            fontsize=8,
            color=color,
            ha='left',
            va='top'
        )

# Save
plt.tight_layout()
plt.savefig(
    _out_dir / "synapses_reci_syntype_ptype_princeton.svg"
)
plt.show()
#%%



#%%


# -*- coding: utf-8 -*-
"""
Conditional synapse-count analysis (given syn_type exists, syn_count >= 1)
Design:
- reciprocal: 0/1
- same_type: 0/1
- syn_type: AD/AA/DA/DD
DV:
- syn_count (integer >=1, conditional on existence)

Recommended test:
- Negative Binomial GLM: syn_count ~ reciprocal * same_type * syn_type
Outputs:
- Descriptive plots (your style)
- Model summary + Wald term tests
- Model-predicted means plot
"""

#%% ---------------------------
# Imports / global settings
# ---------------------------
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import sys
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *  # keep your logic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

#%% ---------------------------
# Load data
# ---------------------------
final_df = pd.read_feather(FULL_RECI_CONNECTIONS_FTR
)

connections = pd.read_feather(CONNECTIONS_TABLE_FTR
)

nodesG = pd.read_feather(NEURON_TABLE_FTR
)

#%% ---------------------------
# Keep only needed columns
# ---------------------------
connections = connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD',
                           'sum_syn', 'reciprocal', 'same_type']].copy()

#%% ---------------------------
# Merge neuron metadata (pre/post)
# ---------------------------
connections = connections.merge(
    nodesG[['root_id', 'SI', 'super_class']],
    left_on='pre', right_on='root_id', how='left'
)
connections = connections.merge(
    nodesG[['root_id', 'SI', 'super_class']],
    left_on='post', right_on='root_id', how='left'
)

#%% ---------------------------
# Filter (your exact filters)
# ---------------------------
allowed_classes = ['central', 'optic', 'visual_centrifugal', 'visual_projection']

connections = connections[connections['super_class_x'].isin(allowed_classes)]
connections = connections[connections['super_class_y'].isin(allowed_classes)]

connections = connections[(connections['SI_x'] >= 0.1) & (connections['SI_y'] >= 0.1)].copy()

# Ensure types
connections['reciprocal'] = connections['reciprocal'].astype(int)
connections['same_type'] = connections['same_type'].astype(int)

#%% ---------------------------
# Custom palette
# ---------------------------
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

syn_cols = ['AD', 'AA', 'DD', 'DA']  # keep your plotting order

#%% ============================================================
# PART A — DESCRIPTIVE PLOTS (your original idea, cleaned labels)
# ============================================================

#%% --- A1: mean synapse count by reciprocity (conditional on >=1)
means_by_recip = {}

for syn_type in syn_cols:
    means_by_recip[syn_type] = {
        'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1)][syn_type].mean(),
        'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0)][syn_type].mean()
    }

fig, ax = plt.subplots(figsize=(7.5, 5))

x_positions = {'r': 0, 'nr': 1}

for syn_type in syn_cols:
    y_values = [means_by_recip[syn_type]['r'], means_by_recip[syn_type]['nr']]
    x_vals = [x_positions['r'], x_positions['nr']]
    ax.plot(
        x_vals,
        y_values,
        marker='o',
        label=syn_type,
        color=custom_palette[syn_type]
    )

ax.set_xticks([0, 1])
ax.set_xticklabels(['r', 'nr'])
ax.set_ylabel('Mean synapse count (given ≥1)')
ax.set_title('Mean synapse count by reciprocity (conditional on existence)')
ax.legend(title='Synapse type', frameon=False)
plt.tight_layout()
plt.show()

#%% --- A2: mean synapse count by reciprocity + same_type (conditional on >=1)
filtered_data = {}

for syn_type in syn_cols:
    filtered_data[syn_type] = {
        1: {  # same_type=1
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 1)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 1)][syn_type].mean()
        },
        0: {  # same_type=0
            'r': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 1) & (connections['same_type'] == 0)][syn_type].mean(),
            'nr': connections[(connections[syn_type] >= 1) & (connections['reciprocal'] == 0) & (connections['same_type'] == 0)][syn_type].mean()
        }
    }

fig, ax = plt.subplots(figsize=(9, 5.5))

for syn_type in syn_cols:
    for same_type_val in [1, 0]:
        y_values = [filtered_data[syn_type][same_type_val]['r'],
                    filtered_data[syn_type][same_type_val]['nr']]
        x_vals = [x_positions['r'], x_positions['nr']]
        linestyle = '-' if same_type_val == 1 else '--'

        ax.plot(
            x_vals,
            y_values,
            marker='o',
            color=custom_palette[syn_type],
            linestyle=linestyle,
            label=f"{syn_type} (same_type={same_type_val})"
        )

ax.set_xticks([0, 1])
ax.set_xticklabels(['r', 'nr'])
ax.set_ylabel('Mean synapse count (given ≥1)')
ax.set_title('Mean synapse count by reciprocity and same-type (conditional on existence)')
ax.legend(title='Type + same_type', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.show()

#%% ============================================================
# PART B — STATISTICAL TESTING (recommended)
# Negative Binomial GLM on positive counts only (syn_count >= 1)
# ============================================================

#%% ---------------------------
# Build long-format dataset (one row per connection × syn_type)
# ---------------------------
long_df = connections.melt(
    id_vars=['pre', 'post', 'reciprocal', 'same_type'],
    value_vars=syn_cols,
    var_name='syn_type',
    value_name='syn_count'
).copy()

# Conditional analysis: only where synapse type exists
long_df = long_df[long_df['syn_count'] >= 1].copy()

# categorical setup
long_df['syn_type'] = pd.Categorical(long_df['syn_type'], categories=syn_cols, ordered=True)
long_df['reciprocal'] = long_df['reciprocal'].astype(int)
long_df['same_type'] = long_df['same_type'].astype(int)
long_df['syn_count'] = long_df['syn_count'].astype(int)

# quick cell counts check (important for imbalance)
cell_counts = long_df.groupby(['syn_type', 'reciprocal', 'same_type']).size().reset_index(name='n_rows')
print("\n--- Cell counts (rows per syn_type × reciprocal × same_type) ---")
print(cell_counts)

print("\n--- syn_count summary ---")
print(long_df['syn_count'].describe())

#%% ---------------------------
# Negative Binomial GLM (full factorial)
# ---------------------------
formula = "syn_count ~ reciprocal * same_type * syn_type"

nb_model = smf.glm(
    formula=formula,
    data=long_df,
    family=sm.families.NegativeBinomial()
).fit()

print("\n==================== NB GLM SUMMARY ====================")
print(nb_model.summary())

# Wald tests per term (ANOVA-like table)
print("\n==================== WALD TEST TERMS ====================")
print(nb_model.wald_test_terms())

#%% ---------------------------
# Model-predicted means (response scale)
# ---------------------------
import itertools

pred_grid = pd.DataFrame(
    list(itertools.product([0, 1], [0, 1], syn_cols)),
    columns=['reciprocal', 'same_type', 'syn_type']
)

pred_grid['syn_type'] = pd.Categorical(pred_grid['syn_type'], categories=syn_cols, ordered=True)

pred_grid['pred_mean'] = nb_model.predict(pred_grid)

print("\n--- Predicted means (conditional, NB GLM) ---")
print(pred_grid.sort_values(['syn_type', 'same_type', 'reciprocal']))

#%% ---------------------------
# Plot predicted means (publish-style, matches stats model)
# ---------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

x_positions_int = {0: 0, 1: 1}  # 0=nr, 1=r

for syn_type in syn_cols:
    for st in [1, 0]:
        sub = pred_grid[(pred_grid['syn_type'] == syn_type) & (pred_grid['same_type'] == st)].copy()
        sub = sub.sort_values('reciprocal')

        linestyle = '-' if st == 1 else '--'

        ax.plot(
            [x_positions_int[r] for r in sub['reciprocal']],
            sub['pred_mean'].values,
            marker='o',
            color=custom_palette[syn_type],
            linestyle=linestyle,
            label=f"{syn_type} (same_type={st})"
        )

ax.set_xticks([0, 1])
ax.set_xticklabels(['nr', 'r'])
ax.set_ylabel('Predicted mean synapse count (given ≥1)')
ax.set_title('Negative Binomial GLM predicted means (conditional on existence)')
ax.legend(title='Type + same_type', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.show()

#%% ============================================================
# PART C — OPTIONAL: simple post-hoc within each syn_type
# (tests reciprocity, same_type, and their interaction per category)
# ============================================================

posthoc_rows = []

for syn_type in syn_cols:
    df_s = long_df[long_df['syn_type'] == syn_type].copy()

    m = smf.glm(
        formula="syn_count ~ reciprocal * same_type",
        data=df_s,
        family=sm.families.NegativeBinomial()
    ).fit()

    posthoc_rows.append({
        "syn_type": syn_type,
        "coef_reciprocal": m.params.get("reciprocal", np.nan),
        "p_reciprocal": m.pvalues.get("reciprocal", np.nan),
        "coef_same_type": m.params.get("same_type", np.nan),
        "p_same_type": m.pvalues.get("same_type", np.nan),
        "coef_interaction": m.params.get("reciprocal:same_type", np.nan),
        "p_interaction": m.pvalues.get("reciprocal:same_type", np.nan),
        "n_rows": len(df_s)
    })

posthoc_df = pd.DataFrame(posthoc_rows)

print("\n==================== POST-HOC (per syn_type) ====================")
print(posthoc_df)

# NOTE:
# If you use post-hoc p-values for claims, apply multiple-testing correction (FDR).
#%%
for i in posthoc_df:
    print(posthoc_df[i])
    #%%
#%% ---------------------------
# BEST VISUAL: 2 panels (same_type split), model-predicted means
# ---------------------------

import matplotlib.pyplot as plt

# Ensure order
syn_cols = ['AD', 'AA', 'DD', 'DA']

custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

panel_map = {
    0: "non-same type (same_type=0)",
    1: "same type (same_type=1)"
}

for ax, st in zip(axes, [0, 1]):
    for syn_type in syn_cols:
        sub = pred_grid[(pred_grid['same_type'] == st) & (pred_grid['syn_type'] == syn_type)].copy()
        sub = sub.sort_values('reciprocal')

        x = ['nr', 'r']  # 0 -> nr, 1 -> r
        y = sub['pred_mean'].values

        ax.plot(
            x, y,
            marker='o',
            linewidth=2,
            color=custom_palette[syn_type],
            label=syn_type
        )

    ax.set_title(panel_map[st])
    ax.set_xlabel('Reciprocity')

axes[0].set_ylabel('Predicted mean synapse count (given ≥1)')

# single legend (right side)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Synapse type', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

plt.tight_layout()
plt.show()

#%%
#%% ---------------------------
# P-values for reciprocity effect within each syn_type and same_type
# ---------------------------

import statsmodels.formula.api as smf
import statsmodels.api as sm

pvals = {}

for st in [0, 1]:
    for syn_type in syn_cols:
        df_sub = long_df[(long_df['same_type'] == st) & (long_df['syn_type'] == syn_type)].copy()

        # NB GLM: syn_count ~ reciprocal
        m = smf.glm(
            formula="syn_count ~ reciprocal",
            data=df_sub,
            family=sm.families.NegativeBinomial()
        ).fit()

        pvals[(st, syn_type)] = float(m.pvalues['reciprocal'])

pvals

#%%
def p_to_stars(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"
#%%
#%% ---------------------------
# 2-panel predicted means + significance markers
# ---------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

panel_title = {
    0: "non-same type (same_type=0)",
    1: "same type (same_type=1)"
}

for ax, st in zip(axes, [0, 1]):

    for syn_type in syn_cols:
        sub = pred_grid[(pred_grid['same_type'] == st) & (pred_grid['syn_type'] == syn_type)].copy()
        sub = sub.sort_values('reciprocal')

        x = np.array([0, 1])  # 0=nr, 1=r
        y = sub['pred_mean'].values

        ax.plot(
            x, y,
            marker='o',
            linewidth=2,
            color=custom_palette[syn_type],
            label=syn_type
        )

        # ----- significance marker for r vs nr within this panel/type -----
        star = p_to_stars(pvals[(st, syn_type)])

        # place star above the midpoint between the two points
        y_star = max(y) * 1.08  # a bit above the higher point
        ax.text(
            0.5, y_star,
            star,
            ha='center',
            va='bottom',
            fontsize=11,
            color=custom_palette[syn_type]
        )

        # optional: draw a small bracket line
        ax.plot([0, 0, 1, 1], [y_star*0.98, y_star, y_star, y_star*0.98],
                linewidth=1, color=custom_palette[syn_type])

    ax.set_title(panel_title[st])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-reciprocal', 'Reciprocal'])
    ax.set_xlabel('Reciprocity')

axes[0].set_ylabel('Predicted mean synapse count (given ≥1)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Synapse type', loc='center left',
           bbox_to_anchor=(1.02, 0.5), frameon=False)

plt.tight_layout()
plt.show()
