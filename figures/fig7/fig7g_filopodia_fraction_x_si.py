# Fig 7G: fraction of boutons with a postsynaptic terminal (left) and postsynaptic sites per bouton, among boutons with at least one (right), against SI.
# Ported from the authors' original script fig7g.py.
# Changes: paths via config.py; figure saves switched on and placed before plt.show().
# The second plot had two commented-out saves under different names; the first is switched on, the second (after plt.show()) stays off.
# The dashed lines at 0.24 and 0.75 are typed in; the 'All' lines use every super-class with no SI filter.
# The axon and dendrite lines keep intrinsic boutons with SI >= 0.1 (left) or SI > 0.1 (right), binned in equal-count
# SI bins over axon and dendrite boutons together; r and p are computed over boutons, not neurons.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BOUTON_FILOPODIA_TABLE_FTR, BOUTON_TABLE_FTR, OUTPUT_DIR

_out_dir = OUTPUT_DIR / "fig7" / "fig7g_filopodia_fraction_x_si"
_out_dir.mkdir(parents=True, exist_ok=True)

#%%

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
#%%
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
# Fill null primary_type with last 9 digits of neuron
bouton_table['primary_type'] = bouton_table['primary_type'].fillna(
    bouton_table['neuron'].astype(str).str[-9:]
)
#%%
# bouton_table = b_table.copy()

bouton_table['primary_type'] = bouton_table['primary_type'].fillna(
    bouton_table['neuron'].astype(str).str[-9:]
)
#%%
intrinsic_superclasses = [
    'central',
    'optic',
    'visual_projection',
    'visual_centrifugal'
]

bouton_table_all = bouton_table
# Filter to intrinsic superclasses
bouton_table = bouton_table[bouton_table['super_class'].isin(intrinsic_superclasses)]

#%%

filopodia= pd.read_feather(BOUTON_FILOPODIA_TABLE_FTR)

#%%
# filopodia = bwf_table.copy()
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
bouton_table_all = bouton_table_all.merge(
    filopodia[['neuron', 'bwf', 'fil_count']],
    left_on=['neuron', 'cluster'],
    right_on=['neuron', 'bwf'],
    how='left'
)
# drop helper column
bouton_table = bouton_table.drop(columns='bwf')
bouton_table_all = bouton_table_all.drop(columns='bwf')

# ensure fil_count is only present when filopodia == True
bouton_table.loc[~bouton_table['filopodia'], 'fil_count'] = None
bouton_table_all.loc[~bouton_table_all['filopodia'], 'fil_count'] = None

#%%%
# filopodia True but missing fil_count (should be rare or zero)
missing = bouton_table.loc[
    bouton_table['filopodia'] & bouton_table['fil_count'].isna()
]

print("Missing fil_count rows:", len(missing))
#%%
del filopodia

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ----------------------------------------------------
# 1. Keep only rows with valid comp (A or D) and SI >= 0.1
# ----------------------------------------------------
df = bouton_table[
    (bouton_table['comp'].isin(['A', 'D'])) &
    (bouton_table['SI'] >= 0.1)
].copy()

df_all = bouton_table_all[
    bouton_table_all['comp'].isin(['A', 'D'])
].copy()
#%%
# ----------------------------------------------------
# 2. Create equal-count SI bins (quantiles)
# ----------------------------------------------------
n_bins = 10

si_edges = np.quantile(
    df['SI'],
    q=np.linspace(0, 1, n_bins + 1)
)
si_edges = np.unique(si_edges)

si_edges_all = np.quantile(
    df_all['SI'],
    q=np.linspace(0, 1, n_bins + 1)
)
si_edges_all = np.unique(si_edges_all)

# ----------------------------------------------------
# 3. Assign SI bins
# ----------------------------------------------------
df['SI_bin'] = pd.cut(
    df['SI'],
    bins=si_edges,
    include_lowest=True
)

df_all['SI_bin'] = pd.cut(
    df_all['SI'],
    bins=si_edges_all,
    include_lowest=True
)

# ----------------------------------------------------
# 4. Compute filopodia fraction per (SI_bin, comp)
# ----------------------------------------------------
binned = (
    df
    .groupby(['SI_bin', 'comp'], observed=True)
    .agg(
        filopodia_fraction=('filopodia', 'mean'),
        si_left=('SI', 'min'),
        si_right=('SI', 'max')
    )
    .reset_index()
)

# ----------------------------------------------------
# 4b. Compute filopodia fraction for ALL data
# ----------------------------------------------------
binned_all = (
    df_all
    .groupby('SI_bin', observed=True)
    .agg(
        filopodia_fraction=('filopodia', 'mean'),
        si_left=('SI', 'min'),
        si_right=('SI', 'max')
    )
    .reset_index()
)

binned_all['si_center'] = (binned_all['si_left'] + binned_all['si_right']) / 2
binned_all = binned_all.sort_values('si_center')

# ----------------------------------------------------
# 5. Compute bin centers
# ----------------------------------------------------
binned['si_center'] = (binned['si_left'] + binned['si_right']) / 2

# ----------------------------------------------------
# 6. Split A and D
# ----------------------------------------------------
A = binned[binned['comp'] == 'A'].sort_values('si_center')
D = binned[binned['comp'] == 'D'].sort_values('si_center')

# ----------------------------------------------------
# 7. Pearson correlations (RAW data)
# ----------------------------------------------------
raw_A = df[df['comp'] == 'A']
raw_D = df[df['comp'] == 'D']

r_A, p_A = stats.pearsonr(raw_A['SI'], raw_A['filopodia'])
r_D, p_D = stats.pearsonr(raw_D['SI'], raw_D['filopodia'])

r_all, p_all = stats.pearsonr(
    df_all['SI'],
    df_all['filopodia']
)

# ----------------------------------------------------
# 8. Helper for p-value formatting
# ----------------------------------------------------
def format_p(p, threshold=1e-3, min_exp=-30):
    if p <= 0 or not np.isfinite(p):
        return rf"<10^{{{min_exp}}}"
    if p < threshold:
        exp = int(np.floor(np.log10(p)))
        if exp < min_exp:
            return rf"<10^{{{min_exp}}}"
        return rf"<10^{{{exp}}}"
    return f"{p:.3f}"

# ----------------------------------------------------
# 9. Plot (UPDATED)
# ----------------------------------------------------
plt.figure(figsize=(6, 6))

# Axon
plt.plot(
    A['si_center'],
    A['filopodia_fraction'],
    marker='o',
    markersize=8,
    markeredgecolor='black',
    alpha=.8,
    label='Axon',
    color='#8B2BE2'
)

# Dendrite
plt.plot(
    D['si_center'],
    D['filopodia_fraction'],
    marker='o',
    markersize=8,
    markeredgecolor='black',
    alpha=.8,
    label='Dendrite',
    color='#F4B95A'
)

# All boutons
plt.plot(
    binned_all['si_center'],
    binned_all['filopodia_fraction'],
    marker='o',
    markersize=8,
    markeredgecolor='black',
    alpha=.8,
    label='All boutons',
    color='#1f77b4'
)

# ----------------------------------------------------
# 10. Annotations
# ----------------------------------------------------
plt.text(0.05, 0.82, rf"Axon: $r={r_A:.2f},\ p{format_p(p_A)}$", color='#8B2BE2')
plt.text(0.05, 0.74, rf"Dendrite: $r={r_D:.2f},\ p{format_p(p_D)}$", color='#F4B95A')
plt.text(0.05, 0.66, rf"All: $r={r_all:.2f},\ p{format_p(p_all)}$", color='#1f77b4')
print(rf"Axon: $r={r_A:.2f},\ p{format_p(p_A)}$")
print(rf"Dendrite: $r={r_D:.2f},\ p{format_p(p_D)}$")
print(rf"All: $r={r_all:.2f},\ p{format_p(p_all)}$")

# ----------------------------------------------------
# 11. Axis styling (UPDATED)
# ----------------------------------------------------
ax = plt.gca()

ax.set_xlim(0, 0.9)
ax.set_ylim(0, 0.9)

ticks = [0, 0.2, 0.4,0.6,0.8]
ax.set_xticks(ticks)
ax.set_yticks(ticks)


# ✅ Add horizontal dashed lines
ax.axhline(0.24, linestyle='--', linewidth=1.5, color='gray', alpha=0.8)
ax.axhline(0.75, linestyle='--', linewidth=1.5, color='gray', alpha=0.8)


ax.set_xlabel('SI')
ax.set_ylabel('Boutons with Filopodia fraction')
ax.set_title('Boutons with Filopodia fraction vs SI (SI ≥ 0.1, equal-count bins)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Equal axis length
ax.set_aspect('equal', adjustable='box')

plt.legend()
plt.tight_layout()

# ----------------------------------------------------
# 12. Save & show
# ----------------------------------------------------
plt.savefig(
    _out_dir / 'SI_vs_fraction_all_new.svg',
    format='svg'
)
plt.show()

#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ----------------------------------------------------
# 0. Prepare data
# ----------------------------------------------------
bt_filo = bouton_table_all[bouton_table_all['fil_count'] >= 1].copy()
bt_filo_SI = bouton_table[bouton_table['fil_count'] >= 1].copy()

# Filter SI > 0.1 for axon / dendrite
bt_filo_SI = bt_filo_SI[bt_filo_SI['SI'] > 0.1].copy()

# ----------------------------------------------------
# 1. Equal-count SI bins for ALL boutons
# ----------------------------------------------------
n_bins_all = 20

si_edges_all = np.quantile(
    bt_filo['SI'].dropna(),
    q=np.linspace(0, 1, n_bins_all + 1)
)
si_edges_all = np.unique(si_edges_all)

bt_filo['SI_bin'] = pd.cut(
    bt_filo['SI'],
    bins=si_edges_all,
    include_lowest=True
)

# ----------------------------------------------------
# 2. Aggregate ALL boutons
# ----------------------------------------------------
binned_all = (
    bt_filo
    .groupby('SI_bin', observed=True)
    .agg(
        fil_count_mean=('fil_count', 'mean'),
        si_left=('SI', 'min'),
        si_right=('SI', 'max')
    )
    .reset_index()
)

binned_all['si_center'] = (binned_all['si_left'] + binned_all['si_right']) / 2
binned_all = binned_all.sort_values('si_center')

# ----------------------------------------------------
# 3. Equal-count SI bins for AXON / DENDRITE
# ----------------------------------------------------
n_bins_SI = 20

si_edges_SI = np.quantile(
    bt_filo_SI['SI'].dropna(),
    q=np.linspace(0, 1, n_bins_SI + 1)
)
si_edges_SI = np.unique(si_edges_SI)

bt_filo_SI['SI_bin'] = pd.cut(
    bt_filo_SI['SI'],
    bins=si_edges_SI,
    include_lowest=True
)

# ----------------------------------------------------
# 4. Aggregate AXON / DENDRITE
# ----------------------------------------------------
binned_SI = (
    bt_filo_SI
    .groupby(['SI_bin', 'comp'], observed=True)
    .agg(
        fil_count_mean=('fil_count', 'mean'),
        si_left=('SI', 'min'),
        si_right=('SI', 'max')
    )
    .reset_index()
)

binned_SI['si_center'] = (binned_SI['si_left'] + binned_SI['si_right']) / 2

A = binned_SI[binned_SI['comp'] == 'A'].sort_values('si_center')
D = binned_SI[binned_SI['comp'] == 'D'].sort_values('si_center')

# ----------------------------------------------------
# 5. Pearson correlations (RAW data)
# ----------------------------------------------------
r_all, p_all = stats.pearsonr(bt_filo['SI'], bt_filo['fil_count'])

r_A, p_A = stats.pearsonr(
    bt_filo_SI[bt_filo_SI['comp'] == 'A']['SI'],
    bt_filo_SI[bt_filo_SI['comp'] == 'A']['fil_count']
)

r_D, p_D = stats.pearsonr(
    bt_filo_SI[bt_filo_SI['comp'] == 'D']['SI'],
    bt_filo_SI[bt_filo_SI['comp'] == 'D']['fil_count']
)

# ----------------------------------------------------
# 6. Helper for p-value formatting
# ----------------------------------------------------
def format_p(p, threshold=1e-3, min_exp=-30):
    if p <= 0 or not np.isfinite(p):
        return rf"<10^{{{min_exp}}}"
    if p < threshold:
        exp = int(np.floor(np.log10(p)))
        if exp < min_exp:
            return rf"<10^{{{min_exp}}}"
        return rf"<10^{{{exp}}}"
    return f"{p:.3f}"

# ----------------------------------------------------
# 7. Plot
# ----------------------------------------------------
plt.figure(figsize=(6, 6))  # square figure
ax = plt.gca()

# ALL boutons
plt.plot(
    binned_all['si_center'],
    binned_all['fil_count_mean'],
    marker='o',
    markersize=8,
    markeredgecolor='black',
    linestyle='-',
    linewidth=2,
    alpha=0.8,
    color='#1f77b4',
    label='All boutons'
)

# AXON
plt.plot(
    A['si_center'],
    A['fil_count_mean'],
    marker='o',
    markersize=8,
    markeredgecolor='black',
    alpha=.8,
    linestyle='-',
    linewidth=2,
    color='#8B2BE2',
    label='Axon (SI > 0.1)'
)

# DENDRITE
plt.plot(
    D['si_center'],
    D['fil_count_mean'],
    marker='o',
    markersize=8,
    markeredgecolor='black',
    alpha=.8,
    linestyle='-',
    linewidth=2,
    color='#F4B95A',
    label='Dendrite (SI > 0.1)'
)

# ----------------------------------------------------
# 8. Annotations
# ----------------------------------------------------
plt.text(
    0.45, 7.3,
    rf"All: $r={r_all:.2f},\ p{format_p(p_all)}$",
    color='#1f77b4',
    fontsize=12
)

plt.text(
    0.45, 6.7,
    rf"Axon: $r={r_A:.2f},\ p{format_p(p_A)}$",
    color='#8B2BE2',
    fontsize=12
)

plt.text(
    0.45, 6.1,
    rf"Dendrite: $r={r_D:.2f},\ p{format_p(p_D)}$",
    color='#F4B95A',
    fontsize=12
)

# ----------------------------------------------------
# 9. Styling (REQUESTED CHANGES)
# ----------------------------------------------------
plt.xlabel('SI')
plt.ylabel('Average Post on Bouton')
plt.title('Average Post on Bouton vs SI\n(only boutons with ≥1 filopodium)')

plt.xlim(0, 0.8)
plt.ylim(0, 8)

plt.xticks([0, 0.2, 0.4, 0.6, 0.8])
plt.yticks([0, 2, 4, 6, 8])

# ax.set_aspect('equal', adjustable='box')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.tight_layout()

# ----------------------------------------------------
# 10. Save & show
# ----------------------------------------------------
plt.savefig(
    _out_dir / 'SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg',
    format='svg'
)

plt.show()

# plt.savefig(
#     _out_dir / 'SI_vs_post_on_bouton_AD_2_new.svg',
#     format='svg'
# )

#%%%



# ---------------------------------------
# df: counts by compartment
# ---------------------------------------
counts_df = df['comp'].value_counts().rename_axis('comp').reset_index(name='count')

print(counts_df)
print("Total rows in df_all:", len(df_all))