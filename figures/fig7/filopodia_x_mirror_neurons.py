# Supp 7-S4 A (within- vs between-type SD of filopodia_fraction, the fraction of a neuron's MSBs with a postsynaptic terminal) and B (left vs right mirror types).
# Ported from the authors' original script Fig 7/filapodia_extraction/filopodia X mirror neurons.py.
# Changes: paths via config.py, repeated import cells merged into the preamble, and the
# bouton-fraction table renamed df2 -> bwf_df, because the ANOVA cells below reuse df1/df2
# as degrees of freedom and the later merge would otherwise receive an int. The unused load of
# SYN_BOUTON_FTR is commented out (see the TODO below; the import is kept so it can be restored).
# The original copies read "syn_bouton_filopodia(new).ftr" and "neurons_nt_bwf_frac(new).ftr";
# the config.py constants name the files without "(new)".
# The ANOVA cells (Welch ANOVA, Games-Howell over every primary type) take a long time and
# only print; the two SVGs do not depend on them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    METHODS_DIR, NEURON_TABLE_FTR, SYN_BOUTON_FTR, NEURONS_NT_BWF_FTR, SI_UPDATED_FTR,
    NEURON_ANNOTATIONS_CSV, OUTPUT_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "fig7" / "filopodia_x_mirror_neurons"
_out_dir.mkdir(parents=True, exist_ok=True)
#%%

# TODO (to confirm with the authors): this table is loaded but never used here (df1 is next assigned as a
# degrees-of-freedom count in the ANOVA cells), and the file is not deposited, so this line alone
# would stop Supp 7-S4 from running. Commented out; to be confirmed with the authors.
# df1=pd.read_feather(SYN_BOUTON_FTR)
#%%

bwf_df=pd.read_feather(NEURONS_NT_BWF_FTR)
#%%
# Load data
nodesG = pd.read_feather(NEURON_TABLE_FTR)

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
nodesG=nodesG.merge(bwf_df[['neuron','filopodia_fraction']],how='left',on='neuron')

#%%
# Group once and get all needed stats
agg_df = nodesG.groupby('primary_type')['filopodia_fraction'].agg(['count', 'mean', 'std']).dropna()
#%%
agg_df = agg_df[agg_df['count'] >= 2]
#%%
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
    _out_dir / "std_of_filopodia_fraction_within_groups_vs_between_princeton.svg"
)
plt.show()






#%%ANOVA


anova_df=nodesG[['SI','primary_type']].dropna()

#%%
import pandas as pd
from scipy.stats import f_oneway


# Group the SI values by primary_type
groups = [group['SI'].values for _, group in anova_df.groupby('primary_type')]

# Perform the ANOVA test
f_stat, p_value = f_oneway(*groups)

print("F-statistic:", f_stat)
print("P-value:", p_value)
#%%# Calculate degrees of freedom
k = anova_df['primary_type'].nunique()  # Number of unique groups
N = len(anova_df)  # Total number of observations

df1 = k - 1  # Between groups degrees of freedom
df2 = N - k  # Within groups degrees of freedom

print("Degrees of Freedom Between Groups (df1):", df1)
print("Degrees of Freedom Within Groups (df2):", df2)
#%%
from scipy.stats import f

# Significance level
alpha = 0.05

# Calculate critical F-value
f_crit = f.ppf(1 - alpha, df1, df2)
print("Critical F-value:", f_crit)
#%%
from scipy.stats import ttest_1samp

# var = within-group SDs
t_stat, p_val = ttest_1samp(var, between_std)
print(f"t = {t_stat:.2f}, p = {p_val:.3e}")
#%%new?!

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg   # pip install pingouin

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
    
    
    #%%next fig. mirror neurons.
    
    
    #%%

nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG=nodesG.drop(columns=['SI'])
all_SI_df=pd.read_feather(SI_UPDATED_FTR)
all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
nodesG[nodesG['neuron'].isin([720575940628378378,720575940612792830])]['SI']

#%%720575940637687605,720575940615226086
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
nodesGog=pd.read_csv(NEURON_ANNOTATIONS_CSV)

nodesGog=nodesGog.rename(columns={'root_id':'neuron'})

nodesG=nodesG.merge(nodesGog[['neuron','side']],on='neuron',how='left')

       #%%
nodesG=nodesG[nodesG['side']!='center']
#%%
nodesG=nodesG.dropna(subset='primary_type')
#%%
nodesG=nodesG.merge(bwf_df,on='neuron',how='left')
#%%
nodesG['p_type_count']=nodesG.groupby(by='primary_type')['primary_type'].transform('count')
#%%

ptypes=nodesG.groupby(by=['primary_type'])
#%%
two_p_types=nodesG[nodesG['p_type_count']==2]

two_p_types=two_p_types[[ 'filopodia_fraction', 'primary_type',
       'side']]


#%%

pivot_df=pd.pivot_table(two_p_types,values='filopodia_fraction',index='primary_type',columns='side')
clean_df = pivot_df.dropna(subset=['left', 'right'])

#%%

custom_palette = {
    'ascending': '#6EB6F6',
    'visual_centrifugal': '#44733B',
    'descending': '#803D3D',
    'endocrine': '#8973B2',
    'motor': '#B48667',
    'optic': '#F4D826',
    'sensory': '#848484',
    'central': '#F9574E',
    'visual_projection': '#D5A848'
}
clean_df = pivot_df.dropna(subset=['left', 'right']).reset_index()

# Merge 'super_class' from nodesG based on 'primary_type'
super_class_map = nodesG[['primary_type', 'super_class']].drop_duplicates()
clean_df = clean_df.merge(super_class_map, on='primary_type', how='left')



#%%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
clean_df=clean_df[clean_df['super_class'].isin(['visual_centrifugal','optic','central','visual_projection'])]
#%%
# Calculate Pearson R
r_value, p_value = pearsonr(clean_df['left'], clean_df['right'])

fig, ax = plt.subplots(figsize=(2.5, 2.5))

sns.scatterplot(
    data=clean_df,
    x='left',
    y='right',
    hue='super_class',
    palette=custom_palette,
    s=2,
    ax=ax
)

# Add identity line x = y
min_val = min(clean_df['left'].min(), clean_df['right'].min())
max_val = max(clean_df['left'].max(), clean_df['right'].max())
ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='black',
    linestyle='--',
    linewidth=1,
    label='x = y'
)

# Text annotations
ax.text(
    0.98, 0.97,
    f'n = {len(clean_df)}',
    fontsize=8,
    color='black',
    ha='right',
    va='top',
    transform=ax.transAxes
)

ax.text(
    0.05, 0.95,
    f'R = {r_value:.2f}',
    fontsize=8,
    transform=ax.transAxes
)

# Labels and ticks
ax.set_xlabel('SI Left', size=8)
ax.set_ylabel('SI Right', size=8)
ax.set_title('SI Left vs Right', size=8)

ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.tick_params(axis='both', labelsize=8)

# Legend
ax.legend(
    title='Super Class',
    bbox_to_anchor=(0.5, 1.35),
    loc='center',
    ncol=3,
    fontsize=6,
    title_fontsize=7
)

sns.despine(ax=ax, right=True, top=True)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

plt.savefig(
    _out_dir / "filopodia_frac_corr_sides_princeton_v2.svg",
    bbox_inches='tight',
    pad_inches=0.05
)

plt.show()
#%%
pivot_df['dif']=abs(pivot_df['left']-pivot_df['right'])
#%%
from scipy.stats import ttest_rel
import numpy as np

# Drop NaNs to ensure pairs are valid
paired_df = clean_df.dropna(subset=['left', 'right'])

# Run paired t-test
t_stat, p_val = ttest_rel(paired_df['left'], paired_df['right'])
mean_diff = np.mean(paired_df['left'] - paired_df['right'])

print(f"Paired t-test:")
print(f"  t = {t_stat:.3f}")
print(f"  p = {p_val:.3e}")
print(f"  Mean difference (Left - Right) = {mean_diff:.4f}")

from scipy.stats import wilcoxon

w_stat, p_wilcoxon = wilcoxon(paired_df['left'], paired_df['right'])
print(f"Wilcoxon signed-rank test: W = {w_stat}, p = {p_wilcoxon:.3e}")
import numpy as np

# Compute Cohen’s d for paired samples
mean_diff = clean_df['left'].mean() - clean_df['right'].mean()
sd_diff = np.std(clean_df['left'] - clean_df['right'], ddof=1)
cohen_d = mean_diff / sd_diff
print(f"Cohen's d (paired) = {cohen_d:.3f}")

#%%
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Initialize threshold range (SI from 0.05 to 0.6 with small increments)
si_thresholds = np.arange(0.05, 0.65, 0.05)

# Prepare a list to store results
results = []

# Iterate through thresholds
for threshold in si_thresholds:
    # Subset for SI below the threshold
    subset_low = clean_df[clean_df['left'] < threshold]
    # Subset for SI equal to or above the threshold
    subset_high = clean_df[clean_df['left'] >= threshold]
    
    # Calculate correlations for both subsets
    if len(subset_low) > 1:
        r_low, _ = pearsonr(subset_low['left'], subset_low['right'])
    else:
        r_low = np.nan  # Not enough data for correlation
    
    if len(subset_high) > 1:
        r_high, _ = pearsonr(subset_high['left'], subset_high['right'])
    else:
        r_high = np.nan  # Not enough data for correlation
    
    # Store the results
    results.append({
        'SI_threshold': threshold,
        'r_low (SI < threshold)': r_low,
        'r_high (SI >= threshold)': r_high
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)
#%%



