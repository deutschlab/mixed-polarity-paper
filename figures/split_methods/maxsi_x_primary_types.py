# Split methods: the Fig 2 within- vs between-type SD plot (Supp 2-S2 D layout), redone with the MaxSI cut's SI.
# Ported from the authors' original script Fig2/fig_2/SI x Primary_types/MaxSI x Primary_types.py.
# Changes: paths via config.py and the repository preamble (which also sets editable Arial text in the SVGs);
# nothing else.
# The Welch ANOVA here covers the types with at least 2 neurons (F(7402, ...)); the one in the Supp 2-S2 D
# legend has F(7510, ...), so it was run on all 7,511 types, singletons included.
# Reads MaxSI_SI from SI_COMPARISONS_FTR (the authors' table of the four cuts' scores; intrinsic neurons only).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SI_COMPARISONS_FTR, SI_UPDATED_FTR
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

_out_dir = OUTPUT_DIR / "split_methods/maxsi_x_primary_types"
_out_dir.mkdir(parents=True, exist_ok=True)
#%%
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from methods_all import *
import os
import pingouin as pg   # pip install pingouin
import seaborn as sns
import pickle
import statsmodels.api as sm

from scipy.stats import f, f_oneway, gaussian_kde, levene, ttest_1samp
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# Load data

nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG=nodesG.drop(columns=['SI'])
all_SI_df=pd.read_feather(SI_UPDATED_FTR)
all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%

df = pd.read_feather(
    SI_COMPARISONS_FTR
)


#%%720575940637687605,720575940615226086
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
nodesG.drop(columns=['SI'],inplace=True)
#%%
nodesG=nodesG.merge(df,how='left',left_on='neuron',right_on='neuron_id')
#%%
aa=nodesG.head()
#%%
nodesG.rename(columns={'MaxSI_SI':'SI'},inplace=True)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])

#%%
# Group once and get all needed stats
agg_df = nodesG.groupby('primary_type')['SI'].agg(['count', 'mean', 'std']).dropna()

agg_df = agg_df[agg_df['count'] >= 2]

count_of_total_neurons=agg_df['count'].sum()
# Extract stds and means
var = agg_df['std']  # within-group stds
group_means = agg_df['mean']  # group means

# Compute between-group std
between_std = group_means.std()
plt.rc('axes', titlesize=8)         # Title size
plt.rc('axes', labelsize=8)         # X and Y label size
plt.rc('xtick', labelsize=4)        # X tick label size
plt.rc('ytick', labelsize=8)        # Y tick label size
plt.rc('legend', fontsize=8)        # Legend text size
plt.rc('legend', title_fontsize=8) 

# Plot
plt.figure(figsize=(4, 2))
sns.histplot(var, bins=80, color='skyblue')
plt.xticks([0,0.1,0.2,0.3],size=4)

plt.axvline(x=between_std, color='black', linestyle='--', label='Between-group std',linewidth=0.4)

plt.xlim(0, 0.3)
plt.text(0.49, plt.ylim()[1] * 0.9, f'n = {len(var)} primary types',
         fontsize=8, color='black', ha='right')

plt.xlabel('std(SI) within group')
#plt.title('Distribution of within-group std(SI) across primary types')
plt.legend()
sns.despine(right=True, top=True)
plt.xlabel('SD of SI', size=8)
plt.ylabel('Neurons', size=8)
plt.xticks(size=8)
plt.yticks(size=8)
plt.tight_layout()
plt.savefig(
    _out_dir / "std_of_si_within_groups_vs_between_princeton_maxSI.svg"
)
plt.show()


#%%


#%%
# ------------------------------------------------------------
# Primary-type analysis: keep only primary_types with >=2 neurons
# ------------------------------------------------------------

# Starting dataframe
analysis_df = nodesG[['neuron', 'SI', 'primary_type', 'super_class']].dropna()

# Count neurons per primary_type
ptype_counts_all = (
    analysis_df.groupby('primary_type')['neuron']
    .nunique()
    .sort_values(ascending=False)
)

valid_ptypes = ptype_counts_all[ptype_counts_all >= 2].index

# Filter to primary types with >=2 neurons
anova_df = analysis_df[analysis_df['primary_type'].isin(valid_ptypes)].copy()

# Counts
n_total_before = analysis_df['neuron'].nunique()
n_total_after = anova_df['neuron'].nunique()

k_total_before = analysis_df['primary_type'].nunique()
k_total_after = anova_df['primary_type'].nunique()

n_excluded_neurons = n_total_before - n_total_after
k_excluded_ptypes = k_total_before - k_total_after

print("=== PRIMARY TYPE ANALYSIS COUNTS ===")
print(f"Intrinsic neurons before primary_type >=2 filter: {n_total_before:,}")
print(f"Primary types before filter: {k_total_before:,}")
print()
print(f"Intrinsic neurons included in analysis: {n_total_after:,}")
print(f"Primary types included, n >= 2: {k_total_after:,}")
print()
print(f"Excluded neurons from singleton primary types: {n_excluded_neurons:,}")
print(f"Excluded singleton primary types: {k_excluded_ptypes:,}")

# Per super_class counts after filtering
print("\nIncluded neurons per super_class:")
sclass_counts = (
    anova_df.groupby('super_class')['neuron']
    .nunique()
    .loc[['central', 'optic', 'visual_projection', 'visual_centrifugal']]
)

for sclass, n in sclass_counts.items():
    print(f"{sclass:<22} n = {n:,}")

# Expected degrees of freedom for classical ANOVA
N = anova_df['neuron'].nunique()
k = anova_df['primary_type'].nunique()

print("\nExpected classical ANOVA df:")
print(f"df1 = k - 1 = {k - 1:,}")
print(f"df2 = N - k = {N - k:,}")
#%%
#%%
# ------------------------------------------------------------
# ANOVA / Welch ANOVA using only valid primary_types
# ------------------------------------------------------------

from scipy.stats import levene, f_oneway
import pingouin as pg

groups = [g['SI'].values for _, g in anova_df.groupby('primary_type')]

levene_stat, levene_p = levene(*groups)
print(f"Levene’s test: W = {levene_stat:.3f}, p = {levene_p:.3e}")

if levene_p < 0.05:
    print("\nVariances differ → Using Welch ANOVA\n")
    welch = pg.welch_anova(
        data=anova_df,
        dv='SI',
        between='primary_type'
    )
    print(welch)

    F_val = welch['F'].iloc[0]
    p_val = welch['p-unc'].iloc[0]
    df1 = welch['ddof1'].iloc[0]
    df2 = welch['ddof2'].iloc[0]
    np2 = welch['np2'].iloc[0]

    print(
        f"\nReport: Welch ANOVA, "
        f"F({df1:.0f}, {df2:.2f}) = {F_val:.3f}, "
        f"p < 0.001, η²p = {np2:.3f}"
    )

else:
    print("\nVariances are equal → Using classical one-way ANOVA\n")
    f_stat, p_value = f_oneway(*groups)

    N = anova_df['neuron'].nunique()
    k = anova_df['primary_type'].nunique()
    df1 = k - 1
    df2 = N - k

    print(f"ANOVA: F({df1}, {df2}) = {f_stat:.3f}, p = {p_value:.3e}")