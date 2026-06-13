# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 00:25:52 2025

@author: user
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FULL_RECI_CONNECTIONS_FTR, NEURON_TABLE_FTR, CLASSIFICATION_CSV,
    OUTPUT_DIR, METHODS_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

from scipy.stats import gaussian_kde

_out_dir = OUTPUT_DIR / "fig5" / "syn_type_x_reci_x_dominance_x_sides"
_out_dir.mkdir(parents=True, exist_ok=True)
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}


#%%
final_df=pd.read_feather(FULL_RECI_CONNECTIONS_FTR)
#%%
r = final_df.query(f"`Node B` == {720575940611514261} and `Node A` == {720575940608199748}")

#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)
#%%
nodesoG=pd.read_csv(CLASSIFICATION_CSV)
nodesoG=nodesoG[['root_id','side']]
nodesoG.columns=['neuron','side']
nodesG=nodesG.merge(nodesoG,on='neuron',how='left')

#%%
final_df=final_df.merge(nodesG[['root_id','super_class','side']],left_on='Node A',right_on='root_id',how='left')
final_df=final_df.merge(nodesG[['root_id','super_class','side']],left_on='Node B',right_on='root_id',how='left')

final_df=final_df.dropna(subset=['super_class_x','super_class_y'])


final_df['same_type']=(final_df['primary_type_A']==final_df['primary_type_B']).astype(int)
final_df['same_side']=(final_df['side_x']==final_df['side_y']).astype(int)

final_df=final_df[(final_df['SI_A']>=0.1)&(final_df['SI_B']>=0.1)]


#%%                  ------------------------------------- non diagonal----------------------
d_non_diagonal={}
for type_1 in ['AD','DA','AA','DD']:
   for type_2 in ['AD','DA','AA','DD']: 
         
        if type_1==type_2:
            pass
        else:
            c1a=final_df[(final_df[f'node_A_pre_{type_1}']>final_df['dim_A->B']/2)&(final_df[f'node_B_pre_{type_2}']>final_df['dim_B->A']/2)][[f'node_A_pre_{type_1}','dim_A->B',f'node_B_pre_{type_2}','dim_B->A','same_type','same_side']]
            c1b=final_df[(final_df[f'node_B_pre_{type_1}']>final_df['dim_B->A']/2)&(final_df[f'node_A_pre_{type_2}']>final_df['dim_A->B']/2)][[f'node_B_pre_{type_1}','dim_B->A',f'node_A_pre_{type_2}','dim_A->B','same_type','same_side']]
            
            c1a.columns=[f'{type_1}',f'{type_1}->{type_2}_dim',f'{type_2}',f'{type_2}->{type_1}_dim','same_type','same_side']
            c1b.columns=[f'{type_1}',f'{type_1}->{type_2}_dim',f'{type_2}',f'{type_2}->{type_1}_dim','same_type','same_side']
            c1=pd.concat([c1a,c1b])
            
            c1_same=c1[c1['same_type']==1]
            c1_same_side=c1_same[c1_same['same_side']==1]
            c1_not_same_side=c1_same[c1_same['same_side']==0]



            total_same_side=len(c1_same_side)
            total_not_same_side=len(c1_not_same_side)
            prop_same_side=total_same_side/len(final_df[final_df['same_type']==1])
            prop_not_same_side=total_not_same_side/len(final_df[final_df['same_type']==1])

            d_non_diagonal[str(type_1)+'_'+str(type_2)]=[total_same_side,total_not_same_side,prop_same_side,prop_not_same_side]


#%%   -------------------------------------diagonal----------------------
d_diagonal={}
for type_ in ['AD','DA','AA','DD']:
    df_v=final_df[
        (final_df[f'node_A_pre_{type_}'] > final_df['dim_A->B'] / 2) &
        (final_df[f'node_B_pre_{type_}'] > final_df['dim_B->A'] / 2)
    ][[f'node_A_pre_{type_}', 'dim_A->B', f'node_B_pre_{type_}', 'dim_B->A', 'same_type','same_side']]
    
        
    df_same = df_v[df_v['same_type'] == 1]
    df_same_side = df_same[df_same['same_side'] == 1]
    df_not_same_side = df_same[df_same['same_side'] == 0]

    total_same_side = len(df_same_side)
    total_not_same_side = len(df_not_same_side)
        
        
        
    prop_same_side = total_same_side / len(final_df[final_df['same_type'] == 1])
    prop_not_same_side = total_not_same_side / len(final_df[final_df['same_type'] == 1])
      
    d_diagonal[type_]=[total_same_side,total_not_same_side,prop_same_side,prop_not_same_side]
#%%# Define types
# Define types
types = ['AD', 'DA', 'AA', 'DD']

# Initialize empty DataFrames
same_total = pd.DataFrame(0, index=types, columns=types)
not_same_total = pd.DataFrame(0, index=types, columns=types)
same_prop = pd.DataFrame(0.0, index=types, columns=types)
not_same_prop = pd.DataFrame(0.0, index=types, columns=types)

# Fill diagonal values from d_diagonal
for t in types:
    same_total.loc[t, t] = d_diagonal[t][0]
    not_same_total.loc[t, t] = d_diagonal[t][1]
    same_prop.loc[t, t] = d_diagonal[t][2] * 100
    not_same_prop.loc[t, t] = d_diagonal[t][3] * 100

# Fill off-diagonal values from d_non_diagonal
for key, values in d_non_diagonal.items():
    from_type, to_type = key.split('_')
    same_total.loc[from_type, to_type] = values[0]
    not_same_total.loc[from_type, to_type] = values[1]
    same_prop.loc[from_type, to_type] = values[2] * 100
    not_same_prop.loc[from_type, to_type] = values[3] * 100



#%%
def sum_lower_triangle(df):
    mask = np.triu(np.ones_like(df, dtype=bool), k=1)  # mask upper triangle
    df_lower = df.mask(mask)  # mask upper triangle values to NaN
    total = df_lower.sum().sum()  # sum remaining values (lower + diagonal)
    return total

# Apply to your matrices
same_prop_lower_sum = sum_lower_triangle(same_prop)
not_same_prop_lower_sum = sum_lower_triangle(not_same_prop)

print(f"Total % in SAME TYPE (lower triangle including diag): {same_prop_lower_sum:.1f}%")
print(f"Total % in NOT SAME TYPE (lower triangle including diag): {not_same_prop_lower_sum:.1f}%")
def sum_lower_triangle_counts(df):
    mask = np.triu(np.ones_like(df, dtype=bool), k=1)  # mask upper triangle
    df_lower = df.mask(mask)  # set upper triangle to NaN
    total = df_lower.sum().sum()  # sum remaining (lower + diag)
    return total

# Apply it to your count matrices
same_total_lower_sum = sum_lower_triangle_counts(same_total)
not_same_total_lower_sum = sum_lower_triangle_counts(not_same_total)

print(f"Total SAME TYPE connections (lower triangle including diag): {same_total_lower_sum}")
print(f"Total NOT SAME TYPE connections (lower triangle including diag): {not_same_total_lower_sum}")
#%%

# New heatmap plotting function with n=xxx in the title
def plot_heatmap_lower_with_diag(data, title, total_value, cmap='Blues'):
    plt.figure(figsize=(8,6))
    mask = np.triu(np.ones_like(data, dtype=bool), k=1)
    sns.heatmap(
        data, 
        mask=mask, 
        annot=True, 
        fmt='.1f' if np.issubdtype(data.values.dtype, np.floating) else 'd',
        cmap=cmap, 
        cbar=True,
        cbar_kws={"location": "left"}
    )
    plt.title(f"{title}\nn={total_value:.0f}")
    plt.tight_layout()
    plt.show()

# For TOTALS
plot_heatmap_lower_with_diag(same_total, 'Same Type same side', same_total_lower_sum)
plot_heatmap_lower_with_diag(not_same_total, 'Same Type not same side', not_same_total_lower_sum)
#%%
# For PERCENTAGES
plot_heatmap_lower_with_diag(same_prop, 'Same Type same side', same_total_lower_sum)
plot_heatmap_lower_with_diag(not_same_prop, 'Same Type not same side', not_same_total_lower_sum)

#%%
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(same_prop, dtype=bool), k=1)
sns.heatmap(
    same_prop, 
    mask=mask, 
    annot=True, 
    fmt='.1f' if np.issubdtype(same_prop.values.dtype, np.floating) else 'd',
    cmap='Blues', 
    cbar=True,
    cbar_kws={"location": "left"},vmin=0,vmax=50,
    ax=ax
)
ax.set_title(f"Same Type Percentage\nn={same_total_lower_sum:.0f}")
fig.tight_layout()
fig.savefig(_out_dir / "same_type_percentage_princeton.svg", format='svg')
#%%
# --- Plot and Save Not Same Type ---
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(not_same_prop, dtype=bool), k=1)
sns.heatmap(
    not_same_prop, 
    mask=mask, 
    annot=True, 
    fmt='.1f' if np.issubdtype(not_same_prop.values.dtype, np.floating) else 'd',
    cmap='Blues', 
    cbar=True,
    cbar_kws={"location": "left"},
    ax=ax,vmin=0,vmax=50
)
ax.set_title(f"Not Same Type Percentage\nn={not_same_total_lower_sum:.0f}")
fig.tight_layout()
fig.savefig(_out_dir / "not_same_type_percentage_princeton.svg", format='svg')
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make a copy to avoid modifying original
same_prop_pct = same_prop.copy()

# Mask for lower triangle including diagonal
lower_mask = np.tril(np.ones_like(same_prop, dtype=bool))

# Get total sum of lower triangle (visible part of plot)
visible_total = same_prop.where(lower_mask).sum().sum()

# Normalize visible values to sum to 100
same_prop_pct = same_prop.where(~lower_mask, same_prop / visible_total * 100)

# Now plot
fig, ax = plt.subplots(figsize=(1.8, 1.8))
mask = np.triu(np.ones_like(same_prop_pct, dtype=bool), k=1)
sns.heatmap(
    same_prop_pct,
    mask=mask,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar=True,
    cbar_kws={"location": "left"},
    vmin=0, vmax=100,    annot_kws={"size": 4}
,
    ax=ax
)
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=4)

# Set axis label sizes
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)

ax.set_title(f"Same Side Percentage (Total=100%)\nn={same_total_lower_sum:.0f}", fontsize=8)
fig.tight_layout()
fig.savefig(_out_dir / "same_type_percentage_same_side_totalnorm_princeton.svg", format='svg')

#%%
# Make a copy to avoid modifying original
not_same_prop_pct = not_same_prop.copy()

# Mask for lower triangle including diagonal
lower_mask = np.tril(np.ones_like(not_same_prop, dtype=bool))

# Get total of visible cells
visible_total = not_same_prop.where(lower_mask).sum().sum()

# Normalize visible values to sum to 100
not_same_prop_pct = not_same_prop.where(~lower_mask, not_same_prop / visible_total * 100)

# Plot
fig, ax = plt.subplots(figsize=(1.8, 1.8))
mask = np.triu(np.ones_like(not_same_prop_pct, dtype=bool), k=1)
sns.heatmap(
    not_same_prop_pct,
    mask=mask,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar=True,
    cbar_kws={"location": "left"},
    vmin=0, vmax=100,
    ax=ax,  annot_kws={"size": 4}
)

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=4)

# Set axis label sizes
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)



ax.set_title(f"Not Same Side Percentage (Total=100%)\nn={not_same_total_lower_sum:.0f}", fontsize=8)
fig.tight_layout()
fig.savefig(_out_dir / "not_same__side_type_percentage_totalnorm_princeton.svg", format='svg')


#%%
# Create a set to store all selected indices
selected_indices = set()

# --- First, collect indices from non-diagonal ---
for type_1 in ['AD', 'DA', 'AA', 'DD']:
    for type_2 in ['AD', 'DA', 'AA', 'DD']:
        if type_1 != type_2:
            # Find rows satisfying non-diagonal condition
            idx1 = final_df[
                (final_df[f'Node_A_pre_{type_1}'] > final_df['dim_A->B']/2) &
                (final_df[f'Node_B_pre_{type_2}'] > final_df['dim_B->A']/2)
            ].index

            idx2 = final_df[
                (final_df[f'Node_B_pre_{type_1}'] > final_df['dim_B->A']/2) &
                (final_df[f'Node_A_pre_{type_2}'] > final_df['dim_A->B']/2)
            ].index

            # Add both
            selected_indices.update(idx1)
            selected_indices.update(idx2)

# --- Then, collect indices from diagonal ---
for type_ in ['AD', 'DA', 'AA', 'DD']:
    idx_diag = final_df[
        (final_df[f'Node_A_pre_{type_}'] > final_df['dim_A->B']/2) &
        (final_df[f'Node_B_pre_{type_}'] > final_df['dim_B->A']/2)
    ].index

    selected_indices.update(idx_diag)

# --- Now find the rows NOT selected ---
all_indices = set(final_df.index)
unselected_indices = all_indices - selected_indices

# Final dataframe of unselected rows
final_df_unselected = final_df.loc[list(unselected_indices)].reset_index(drop=True)

print(f"Number of rows NOT in any of the selected categories: {len(final_df_unselected)}")
#%%side analysis