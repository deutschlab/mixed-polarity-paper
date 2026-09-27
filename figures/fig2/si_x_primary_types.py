
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
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
nodesG = pd.read_feather(NEURON_TABLE_FTR)

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
_out_dir = OUTPUT_DIR / "fig2" / "SI_x_primary_types"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    _out_dir / "std_of_si_within_groups_vs_between_princeton.svg"
)
plt.show()






# ANOVA and post-hoc analysis


anova_df=nodesG[['SI','primary_type']].dropna()

# Group the SI values by primary_type
groups = [group['SI'].values for _, group in anova_df.groupby('primary_type')]

# Perform the ANOVA test
f_stat, p_value = f_oneway(*groups)

print("F-statistic:", f_stat)
print("P-value:", p_value)

# Calculate degrees of freedom
k = anova_df['primary_type'].nunique()  # Number of unique groups
N = len(anova_df)  # Total number of observations

df1 = k - 1  # Between groups degrees of freedom
df2 = N - k  # Within groups degrees of freedom

print("Degrees of Freedom Between Groups (df1):", df1)
print("Degrees of Freedom Within Groups (df2):", df2)

# Significance level
alpha = 0.05

# Calculate critical F-value
f_crit = f.ppf(1 - alpha, df1, df2)
print("Critical F-value:", f_crit)

# var = within-group SDs
t_stat, p_val = ttest_1samp(var, between_std)
print(f"t = {t_stat:.2f}, p = {p_val:.3e}")

# Alternative ANOVA workflow

# Prepare data (already filtered in your script)
anova_df = nodesG[['SI', 'primary_type']].dropna()

# Group SI values by type
groups = [group['SI'].values for _, group in anova_df.groupby('primary_type')]

# 1) Levene's Test – equal variance assumption
levene_stat, levene_p = levene(*groups)
print(f"Levene’s Test: W={levene_stat:.3f}, p={levene_p:.3e}")

# Choose ANOVA type based on Levene's result
if levene_p >= 0.05:
    print("\nVariances are equal → Using classical one-way ANOVA\n")
    f_stat, p_value = f_oneway(*groups)
else:
    print("\nVariances differ → Using Welch ANOVA\n")
    welch = pg.welch_anova(data=anova_df, dv='SI', between='primary_type')
    print(welch)

    # For effect sizes and post-hoc we continue with OLS model
    f_stat = welch['F'].iloc[0]
    p_value = welch['p-unc'].iloc[0]

print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3e}")

# Degrees of freedom
k = anova_df['primary_type'].nunique()
N = len(anova_df)
df1 = k - 1
df2 = N - k
print(f"DF: {df1}, {df2}")

# 2) Effect size: η² and ω²
# ANOVA model for sums of squares
model = ols('SI ~ C(primary_type)', data=anova_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

ss_between = anova_table['sum_sq']['C(primary_type)']
ss_within = anova_table['sum_sq']['Residual']
ss_total = ss_between + ss_within

eta_sq = ss_between / ss_total
omega_sq = (ss_between - (df1 * ss_within / df2)) / (ss_total + ss_within)

print(f"\nEffect sizes:")
print(f"η² (eta squared) = {eta_sq:.3f}")
print(f"ω² (omega squared) = {omega_sq:.3f}")

# 3) Post-hoc analysis
if levene_p >= 0.05:
    print("\nTukey HSD Post-Hoc\n")
    tukey = pairwise_tukeyhsd(endog=anova_df['SI'],
                              groups=anova_df['primary_type'],
                              alpha=0.05)
    print(tukey)
else:
    print("\nGames-Howell Post-Hoc (Welch condition)\n")
    gh = pg.pairwise_gameshowell(dv='SI', between='primary_type', data=anova_df)
    print(gh[['A', 'B', 'pval', 'hedges']].head())

#%% Primary types with >=2 neurons: Welch ANOVA and partial eta squared
# Types with a single neuron are left out: they have no within-type spread.
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

    p_txt = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3g}"
    print(
        f"\nReport: Welch ANOVA, "
        f"F({df1:.0f}, {df2:.2f}) = {F_val:.3f}, "
        f"{p_txt}, η²p = {np2:.3f}"
    )

else:
    print("\nVariances are equal → Using classical one-way ANOVA\n")
    f_stat, p_value = f_oneway(*groups)

    N = anova_df['neuron'].nunique()
    k = anova_df['primary_type'].nunique()
    df1 = k - 1
    df2 = N - k

    print(f"ANOVA: F({df1}, {df2}) = {f_stat:.3f}, p = {p_value:.3e}")
