import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FULL_RECI_CONNECTIONS_FTR, CONNECTIONS_TABLE_FTR, CONNECTIONS_TABLE_NONP_FTR,
    NEURON_TABLE_FTR, NEURON_TABLE_NONP_FTR, OUTPUT_DIR, METHODS_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
from scipy.stats import gaussian_kde
#%%
custom_palette = {
    'AA': '#8b2be2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}
#%%
final_df=pd.read_feather(FULL_RECI_CONNECTIONS_FTR)
#%%new calculation for AD-DA proportions
'''

df_AB=final_df[(final_df['node_A_pre_AD']>0)&(final_df['node_B_pre_DA']>0)][['node_A_pre_AD','node_B_pre_DA']]

df_BA=final_df[(final_df['node_A_pre_DA']>0)&(final_df['node_B_pre_AD']>0)][['node_A_pre_DA','node_B_pre_AD']]

df_AB.columns=['pre_AD','pre_DA']
df_BA.columns=['pre_DA','pre_AD']


df=pd.concat([df_AB,df_BA])

df['sym_ratio'] = (df['pre_AD'] - df['pre_DA']) / (df['pre_AD'] + df['pre_DA'])

df['per']=df['pre_AD'] /df['pre_DA']

df['per'].mean()

sns.histplot(df['per'],bins=750)
plt.xlabel('AD / DA ratio')
plt.xlim(0,10)
'''
#%%
connections=pd.read_feather(CONNECTIONS_TABLE_FTR)
connections=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 
       'sum_syn', 'reciprocal', 
       'same_type']]
#%%

nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)

connections=connections.merge(nodesG[['root_id','super_class','SI']],left_on='pre',right_on='root_id',how='left')
connections=connections.merge(nodesG[['root_id','super_class','SI']],left_on='post',right_on='root_id',how='left').dropna()
#%%
connections=connections[(connections['SI_x']>=0.1)&(connections['SI_y']>=0.1)]
#%%
from scipy.stats import chi2_contingency

# Observed counts
a = len(connections[(connections['same_type'] == 1) & (connections['reciprocal'] == 1)])
b = len(connections[(connections['same_type'] == 1) & (connections['reciprocal'] == 0)])
c = len(connections[(connections['same_type'] == 0) & (connections['reciprocal'] == 1)])
d = len(connections[(connections['same_type'] == 0) & (connections['reciprocal'] == 0)])

# Create contingency table
contingency_table = np.array([[a, b], [c, d]])
observed_df = pd.DataFrame(contingency_table, columns=["Reciprocal", "Non-Reciprocal"], index=["Same", "Not Same"])


chi2, p, dof, expected = chi2_contingency(contingency_table)
expected_df = pd.DataFrame(expected, columns=["Reciprocal", "Non-Reciprocal"], index=["Same", "Not Same"])


# Calculate proportions for observed and expected counts per row
observed_proportions = observed_df.div(observed_df.sum(axis=1), axis=0)
expected_proportions = expected_df.div(expected_df.sum(axis=1), axis=0)

# Prepare data for plotting
labels = [
    'Reciprocal (Same)',
    'Non-Reciprocal (Same)',
    'Reciprocal (Not Same)',
    'Non-Reciprocal (Not Same)'
]

observed_values = [
    observed_proportions.loc['Same', 'Reciprocal'],
    observed_proportions.loc['Same', 'Non-Reciprocal'],
    observed_proportions.loc['Not Same', 'Reciprocal'],
    observed_proportions.loc['Not Same', 'Non-Reciprocal']
]

expected_values = [
    expected_proportions.loc['Same', 'Reciprocal'],
    expected_proportions.loc['Same', 'Non-Reciprocal'],
    expected_proportions.loc['Not Same', 'Reciprocal'],
    expected_proportions.loc['Not Same', 'Non-Reciprocal']
]

# Adjust x-axis positions to reduce gaps between bars
x = np.array([0, 0.3, 0.75, 1.05])  # Reduced gap between left and right groups
bar_width = 0.25  # Slightly narrower bars to fit closer together

# Recommended colors
observed_color = '#17a2b8'   # Teal for observed bars
expected_color = '#f39c12'   # Orange for expected bars
mixed_color_alpha = 0.5      # Transparency to show overlapping areas

fig, ax = plt.subplots(figsize=(8, 6))

# Plot expected bars first (base layer)
ax.bar(x, expected_values, bar_width, label='Expected', color=expected_color, alpha=mixed_color_alpha)

# Plot observed bars directly on top of expected bars (to show differences)
ax.bar(x, observed_values, bar_width, label='Observed', color=observed_color, alpha=mixed_color_alpha)

# Formatting
ax.set_xlabel('Category')
ax.set_ylabel('Proportion')
ax.set_title('Observed vs. Expected Proportions (Grouped and Overlapping)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20)
ax.legend()

plt.tight_layout()
plt.show()

# Print Chi-Square Results
print("Chi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of Freedom: {dof}")

#%%
import matplotlib.pyplot as plt

# Normalize observed_df to get proportions
proportions_observed = observed_df.div(observed_df.sum(axis=1), axis=0)
proportions_observed = proportions_observed.reset_index()

proportions_observed = proportions_observed[proportions_observed['index'] == 'Same']
proportions_observed.index = proportions_observed['index']
proportions_observed = proportions_observed.drop(columns=['index'])

# Transpose the original observed DataFrame to switch rows and columns
observed_df_transposed = observed_df.T

# Calculate proportions for the transposed DataFrame
proportions_observed_transposed = observed_df_transposed.div(observed_df_transposed.sum(axis=1), axis=0)
proportions_observed_transposed = proportions_observed_transposed[['Same']]

# Create subplots with separate y-axes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot first subplot with its own y-axis
proportions_observed.T.plot(kind='bar', color=['#1f77b4'], ax=axes[0])
axes[0].set_title('Proportion of Reciprocal vs Non-Reciprocal in Same vs Not Same')
axes[0].set_ylabel('Proportion')

# Plot second subplot with its own y-axis
proportions_observed_transposed.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], ax=axes[1])
axes[1].set_title('Proportion of Same vs Not Same in Reciprocal vs Non-Reciprocal')
axes[1].set_ylabel('Proportion')
axes[1].set_xlabel('Reciprocal vs Non-Reciprocal')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[0].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

# Adjust layout
plt.tight_layout()
plt.show()

# Print Chi-Square Results
print("Chi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of Freedom: {dof}")
#%%


# Reciprocal Connections
reci = connections[connections['reciprocal'] == 1]
same_reci = reci[reci['same_type'] == 1]
same_reci_value_graph = len(same_reci) / len(reci)

# Non-Reciprocal Connections
non_reci = connections[connections['reciprocal'] == 0]
same_non_reci = non_reci[non_reci['same_type'] == 1]
same_non_reci_value_graph = len(same_non_reci) / len(non_reci)

# Plot
sns.barplot(x=['Same / Reciprocal', 'Same / Non-Reciprocal'], 
            y=[same_reci_value_graph, same_non_reci_value_graph])

plt.ylabel('Proportion of Same Type')
plt.title('Proportion of Same Type in Reciprocal vs. Non-Reciprocal Connections')
plt.show()


nstype=connections[connections['same_type']==0]
nstyper=nstype[nstype['reciprocal']==1]
nsame_value_graph=len(nstyper)/len(nstype)

#same tpye
stype=connections[connections['same_type']==1]
styper=stype[stype['reciprocal']==1]

same_value_graph=len(styper)/len(stype)

#result
sns.barplot(x=['All Connections', 'Same Type Connections'], 
            y=[nsame_value_graph, same_value_graph])

plt.ylabel('Reciprocity Value')
plt.title('Reciprocity in Network Connections')
plt.show()

#%%

import os
import matplotlib.pyplot as plt
import seaborn as sns

_out_dir = OUTPUT_DIR / "fig5" / "chi_reci_identity"
_out_dir.mkdir(parents=True, exist_ok=True)

# Compute proportions
reci = connections[connections['reciprocal'] == 1]
same_reci = reci[reci['same_type'] == 1]
same_reci_value_graph = len(same_reci) / len(reci)

non_reci = connections[connections['reciprocal'] == 0]
same_non_reci = non_reci[non_reci['same_type'] == 1]
same_non_reci_value_graph = len(same_non_reci) / len(non_reci)

nstype = connections[connections['same_type'] == 0]
nstyper = nstype[nstype['reciprocal'] == 1]
nsame_value_graph = len(nstyper) / len(nstype)

stype = connections[connections['same_type'] == 1]
styper = stype[stype['reciprocal'] == 1]
same_value_graph = len(styper) / len(stype)

fig, axes = plt.subplots(1, 2, figsize=(9, 5))

# Define bar colors
bar_colors = ['#1f77b4', '#ff7f0e']  # Blue and Orange

# Right plot: P(Same-Type | Reci / Non-Reci)
sns.barplot(
    ax=axes[1], 
    x=['Reciprocal', 'Non-Reciprocal'], 
    y=[same_reci_value_graph, same_non_reci_value_graph],
    palette=bar_colors
)
axes[1].set_ylabel('Proportion of Same-Type')
axes[1].set_title('Proportions of Same-Type')
axes[1].set_xticklabels(['Pr(Same|Reciprocal)', 'Pr(Same|Non-Reciprocal)'])

# Left plot: P(Reciprocal | Same-Type / Not Same-Type)
sns.barplot(
    ax=axes[0], 
    x=['Not Same', 'Same'], 
    y=[nsame_value_graph, same_value_graph],
    palette=bar_colors
)
axes[0].set_ylabel('Reciprocity Value')
axes[0].set_title('Graph Reciprocity')
axes[0].set_xticklabels(['Pr(Reciprocal|Not Same)', 'Pr(Reciprocal|Same)'])

# Remove spines
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Save and show
fig_path = _out_dir / "chi_square_type_reci_filtered_princeton.svg"
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()
#%%
import os
import matplotlib.pyplot as plt
import numpy as np

# Save path (already created above)


# Compute proportions
reci = connections[connections['reciprocal'] == 1]
same_reci = reci[reci['same_type'] == 1]
same_reci_value_graph = len(same_reci) / len(reci)

non_reci = connections[connections['reciprocal'] == 0]
same_non_reci = non_reci[non_reci['same_type'] == 1]
same_non_reci_value_graph = len(same_non_reci) / len(non_reci)

nstype = connections[connections['same_type'] == 0]
nstyper = nstype[nstype['reciprocal'] == 1]
nsame_value_graph = len(nstyper) / len(nstype)

stype = connections[connections['same_type'] == 1]
styper = stype[stype['reciprocal'] == 1]
same_value_graph = len(styper) / len(stype)


total_not_same=len(nstype)
total_reci=len(reci)
total_same=len(stype)
total_not_reci=len(non_reci)
# Convert to millions
total_not_same_m = total_not_same / 1e6
total_same_m = total_same / 1e6
total_reci_m = total_reci / 1e6
total_not_reci_m = total_not_reci / 1e6

# Pie data
sizes1 = [total_not_same, total_same]
labels1_numeric = [f'{total_not_same_m:.2f}M', f'{total_same_m:.2f}M']
legend_labels1 = ['Not Same', 'Same']

sizes2 = [total_reci, total_not_reci]
labels2_numeric = [f'{total_reci_m:.2f}M', f'{total_not_reci_m:.2f}M']
legend_labels2 = ['Reciprocal', 'Non-Reciprocal']

colors = ['#1f77b4', '#ff7f0e']

# Pie 1: Not Same vs Same
fig1, ax1 = plt.subplots(figsize=(6, 6))
wedges1, _ = ax1.pie(sizes1, labels=[None, None], colors=colors, startangle=90)

# Place numeric values inside slices
for i, wedge in enumerate(wedges1):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = 0.6 * np.cos(np.deg2rad(angle))
    y = 0.6 * np.sin(np.deg2rad(angle))
    ax1.text(x, y, labels1_numeric[i], ha='center', va='center', fontsize=10)

ax1.set_title('Proportion of Not Same vs. Same')
ax1.legend(wedges1, legend_labels1, title="Type", loc="center left", bbox_to_anchor=(1, 0.5))

fig1_path = _out_dir / "proportion_not_same_vs_same_SIfiltered_princeton.svg"
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')

# Pie 2: Reciprocal vs Non-Reciprocal
fig2, ax2 = plt.subplots(figsize=(6, 6))
wedges2, _ = ax2.pie(sizes2, labels=[None, None], colors=colors, startangle=90)

# Place numeric values inside slices
for i, wedge in enumerate(wedges2):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = 0.6 * np.cos(np.deg2rad(angle))
    y = 0.6 * np.sin(np.deg2rad(angle))
    ax2.text(x, y, labels2_numeric[i], ha='center', va='center', fontsize=10)

ax2.set_title('Proportion of Reciprocal vs. Non-Reciprocal')
ax2.legend(wedges2, legend_labels2, title="Type", loc="center left", bbox_to_anchor=(1, 0.5))

fig2_path = _out_dir / "proportion_reciprocal_vs_nonreciprocal_SI_filtered_princeton.svg"
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')

plt.show()

print(fig1_path)
print(fig2_path)
#%%
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
#%% Imports
# Brighten color function


# Brighten color function
def brighten_color(color, amount=0.35):
    """
    Brighten a matplotlib color.
    amount=0.6 means mix 60% of white into the original color.
    """
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return c + (white - c) * amount

# Base and brightened colors
base_blue = '#1f77b4'
brighter_blue = brighten_color(base_blue, amount=0.35)

#%% Barplots

# _out_dir already created above

# Compute proportions
reci = connections[connections['reciprocal'] == 1]
same_reci = reci[reci['same_type'] == 1]
same_reci_value_graph = len(same_reci) / len(reci)

non_reci = connections[connections['reciprocal'] == 0]
same_non_reci = non_reci[non_reci['same_type'] == 1]
same_non_reci_value_graph = len(same_non_reci) / len(non_reci)

nstype = connections[connections['same_type'] == 0]
nstyper = nstype[nstype['reciprocal'] == 1]
nsame_value_graph = len(nstyper) / len(nstype)

stype = connections[connections['same_type'] == 1]
styper = stype[stype['reciprocal'] == 1]
same_value_graph = len(styper) / len(stype)
#%%


fig, axes = plt.subplots(1, 2, figsize=(3.8, 2))

# Define bar colors (blue and brighter blue)
bar_colors = [base_blue, brighter_blue]

# Right plot: P(Same-Type | Reci / Non-Reci)
sns.barplot(
    ax=axes[1], 
    x=['Reciprocal', 'Non-Reciprocal'], 
    y=[same_reci_value_graph, same_non_reci_value_graph],
    palette=bar_colors
)
axes[1].set_ylabel('Proportion of Same-Type')
axes[1].set_title('Proportions of Same-Type')
axes[1].set_xticklabels(['Pr(Same|Reciprocal)', 'Pr(Same|Non-Reciprocal)'])

# Left plot: P(Reciprocal | Same-Type / Not Same-Type)
sns.barplot(
    ax=axes[0], 
    x=['Not Same', 'Same'], 
    y=[nsame_value_graph, same_value_graph],
    palette=bar_colors
)
# Left plot text settings
axes[0].set_xlabel("Same-Type Status", fontsize=8)
axes[0].set_ylabel("Reciprocity Value", fontsize=8)
axes[0].set_title("Graph Reciprocity", fontsize=8)
axes[0].set_xticklabels(['Pr(Reciprocal|Not Same)', 'Pr(Reciprocal|Same)'], fontsize=4, rotation=0)
axes[0].tick_params(axis='y', labelsize=8)

# Right plot text settings
axes[1].set_xlabel("Reciprocity Status", fontsize=8)
axes[1].set_ylabel("Proportion of Same-Type", fontsize=8)
axes[1].set_title("Proportions of Same-Type", fontsize=8)
axes[1].set_xticklabels(['Pr(Same|Reciprocal)', 'Pr(Same|Non-Reciprocal)'], fontsize=4, rotation=0)
axes[1].tick_params(axis='y', labelsize=8)

# Remove top and right spines
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Save and show
fig_path = _out_dir / "chi_square_type_reci_filtered_princeton.svg"
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

#%% Pie charts

# Save path again (already created above, safe)
# Compute proportions again (already computed above)

total_not_same = len(nstype)
total_reci = len(reci)
total_same = len(stype)
total_not_reci = len(non_reci)

# Convert to millions
total_not_same_m = total_not_same / 1e6
total_same_m = total_same / 1e6
total_reci_m = total_reci / 1e6
total_not_reci_m = total_not_reci / 1e6

# Pie data
sizes1 = [total_not_same, total_same]
labels1_numeric = [f'{total_not_same_m:.2f}M', f'{total_same_m:.2f}M']
legend_labels1 = ['Not Same', 'Same']

sizes2 = [total_reci, total_not_reci]
labels2_numeric = [f'{total_reci_m:.2f}M', f'{total_not_reci_m:.2f}M']
legend_labels2 = ['Reciprocal', 'Non-Reciprocal']

# Define colors for pie
colors = [base_blue, brighter_blue]

# Pie 1: Not Same vs Same
fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
wedges1, _ = ax1.pie(sizes1, labels=[None, None], colors=colors, startangle=90,textprops={'fontsize': 8} )

# Place numeric values inside slices
for i, wedge in enumerate(wedges1):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = 0.6 * np.cos(np.deg2rad(angle))
    y = 0.6 * np.sin(np.deg2rad(angle))
    ax1.text(x, y, labels1_numeric[i], ha='center', va='center', fontsize=8)

#ax1.set_title('Proportion of Not Same vs. Same')
ax1.legend(wedges1, legend_labels1, title="Type", loc="center left", bbox_to_anchor=(1, 0.5))

fig1_path = _out_dir / "proportion_not_same_vs_same_SIfiltered_princeton.svg"
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
#%%
# Pie 2: Reciprocal vs Non-Reciprocal
fig2, ax2 = plt.subplots(figsize=(1.5, 1.5))
wedges2, _ = ax2.pie(sizes2, labels=[None, None], colors=colors, startangle=90,textprops={'fontsize': 8} )

# Place numeric values inside slices
for i, wedge in enumerate(wedges2):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = 0.6 * np.cos(np.deg2rad(angle))
    y = 0.6 * np.sin(np.deg2rad(angle))
    ax2.text(x, y, labels2_numeric[i], ha='center', va='center', fontsize=8)

#ax2.set_title('Proportion of Reciprocal vs. Non-Reciprocal')
ax2.legend(wedges2, legend_labels2, title="Type", loc="center left", bbox_to_anchor=(1, 0.5))

fig2_path = _out_dir / "proportion_reciprocal_vs_nonreciprocal_SI_filtered_princeton.svg"
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')

plt.show()

print(fig1_path)
print(fig2_path)
#%%



#reciprocity 5+





sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

from scipy.stats import gaussian_kde
#%%


nodesG=pd.read_feather(NEURON_TABLE_NONP_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)

#%%
connections=pd.read_feather(CONNECTIONS_TABLE_NONP_FTR)
#%%
connections5=connections[connections['sum_syn']>=5]
#%%
rc5=connections5[connections5['reciprocal']==1]

#%%
rc5=len(connections5[connections5['reciprocal']==1])/len(connections5)
#%%
rc=len(connections[connections['reciprocal']==1])/len(connections)
#%%
# Create a set of all (pre, post) pairs in the data
pairs_set = set(zip(connections5['pre'], connections5['post']))

# Check if the reverse (post, pre) exists for each row
connections5['reciprocal'] = connections5.apply(
    lambda row: int((row['post'], row['pre']) in pairs_set),
    axis=1
)

#%%

len(connections5[connections5['reciprocal']==1])/len(connections5)
#%%
# Sample size (total observations)
n = contingency_table.sum()
r, c = contingency_table.shape

# Cramer's V (effect size)
cramers_v = np.sqrt(chi2 / (n * (min(r-1, c-1))))

print("Chi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2:.3f}")
print(f"p-value: {p:.3e}")
print(f"Degrees of Freedom: {dof}")
print(f"Cramer's V (effect size): {cramers_v:.3f}")

# Optional: check smallest expected frequency
print(f"Min expected frequency: {expected.min():.2f}")
#%%