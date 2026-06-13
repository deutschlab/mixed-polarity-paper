
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    NEURON_TABLE_FTR, CONNECTIONS_TABLE_FTR, RECI_PROP_FTR, OUTPUT_DIR, METHODS_DIR,
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
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)

#%%
connections=pd.read_feather(CONNECTIONS_TABLE_FTR)
#%%
connections=connections.merge(nodesG[['root_id','super_class']],left_on='pre',right_on='root_id',how='left')
connections=connections.merge(nodesG[['root_id','super_class']],left_on='post',right_on='root_id',how='left').dropna()
#%%
aa=connections[connections['pre']==connections['post']]
#%%
#%%
connections=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 
       'sum_syn', 'reciprocal', 
       'same_type']].dropna()

#%%

# Step 1: Identify all unique neurons from 'pre' and 'post' columns
unique_neurons = nodesG[['root_id']]
#%%
# Step 2: Count total unique partners for each neuron
partner_counts = (
    pd.concat([
        connections[['pre', 'post', 'sum_syn']],
        connections[['post', 'pre', 'sum_syn']].rename(columns={'post': 'pre', 'pre': 'post'})
    ])
    .groupby('pre')
    .agg(
        total_partners=('post', 'nunique'),
        total_synapses=('sum_syn', 'sum')
    )
)
#%%
# Step 3: Count reciprocal connections for each neuron
reciprocal_df = connections[connections['reciprocal'] == 1]

reciprocal_counts = (
    pd.concat([
        reciprocal_df[['pre', 'post', 'sum_syn']],
        reciprocal_df[['post', 'pre', 'sum_syn']].rename(columns={'post': 'pre', 'pre': 'post'})
    ])
    .groupby('pre')
    .agg(
        reciprocal_partners=('post', 'nunique'),
        reciprocal_synapses=('sum_syn', 'sum')
    )
)

#%%
# Combine the two series into a single DataFrame with each neuron as a row
final_neuron_stats_df = (
    pd.DataFrame(index=partner_counts.index)
    .join(partner_counts, how='left')
    .join(reciprocal_counts, how='left')
    .fillna(0)
    .astype(int)
    .reset_index()
    .rename(columns={'index': 'neuron'})
)
#%%
final_neuron_stats_df_d=final_neuron_stats_df.dropna()
#%%
final_neuron_stats_df['pre']=final_neuron_stats_df['pre'].astype(str)
nodesG['root_id']=nodesG['root_id'].astype(str)

#%%
final_neuron_stats_df['non_reciprocal_partners']=final_neuron_stats_df['total_partners']-final_neuron_stats_df['reciprocal_partners']
final_neuron_stats_df['non_reciprocal_synapses']=final_neuron_stats_df['total_synapses']-final_neuron_stats_df['reciprocal_synapses']
final_neuron_stats_df['non_reciprocal_ratio']=final_neuron_stats_df['non_reciprocal_synapses']/final_neuron_stats_df['non_reciprocal_partners']
final_neuron_stats_df['reciprocal_ratio']=final_neuron_stats_df['reciprocal_synapses']/final_neuron_stats_df['reciprocal_partners']
final_neuron_stats_df['ratio']=final_neuron_stats_df['reciprocal_partners']/final_neuron_stats_df['total_partners']
final_neuron_stats_df['ratio_syn']=final_neuron_stats_df['reciprocal_ratio']/final_neuron_stats_df['non_reciprocal_ratio']
final_neuron_stats_df=final_neuron_stats_df.merge(nodesG[['root_id','super_class']],left_on='pre',right_on='root_id',how='right')
#%%
final_neuron_stats_df['reciprocal_ratio']=final_neuron_stats_df['reciprocal_ratio']/2
#%%


#%%
_out_dir = OUTPUT_DIR / "fig5" / "reciprocal_fraction"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(1.8,1.8))
sns.histplot(final_neuron_stats_df, x='ratio', bins=50)
#plt.title('Proportion of Reciprocal Partners')

# Clean up axis
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('Neurons',size=8)
plt.xlabel('Reciprocal Fraction',size=8)
plt.yticks([0,3000,6000,9000,12000,15000],size=8)

plt.xticks([0.0, 0.15, 0.3, 0.45, 0.6], labels=["0", "0.15", "0.3", "0.45", "0.6"], size=8)
plt.yticks(size=8)

# Add annotation for n
n_samples = len(final_neuron_stats_df)
n_k =n_samples #round(n_samples / 1000, 1)
plt.text(0.95, 0.95, f"n = {n_k}", transform=ax.transAxes,
         ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', edgecolor='white'))
plt.xlim(0,0.6)
# Save and show
fig_path = _out_dir / "reciprocal_partners_hist_princeton.svg"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Figure saved at: {fig_path}")
#%%
final_neuron_stats_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows that now contain NaN (formerly inf)
final_neuron_stats_df.dropna(inplace=True)


#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
plt.figure(figsize=(3, 3))

# Calculate ratio safely
ratios = final_neuron_stats_df['reciprocal_ratio'] / final_neuron_stats_df['non_reciprocal_ratio']
ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
mean_ratio = ratios.median()

# Create a jointgrid
g = sns.JointGrid(
    data=final_neuron_stats_df,
    x='non_reciprocal_ratio',
    y='reciprocal_ratio',
    xlim=(0, 10),
    ylim=(0, 10),
    height=2,    ratio=3  # <-- allocate more space to marginals

)

# Main 2D KDE plot
g.plot_joint(
    sns.kdeplot,
    fill=True,
    cmap="mako_r",
    thresh=0.19,
    levels=15,
    linewidths=1
)

# Add reference line
x_vals = np.linspace(0, 15, 150)
g.ax_joint.plot(x_vals, x_vals, linestyle='--', color='blue', linewidth=0.75, label='x = y')

# Marginal distributions
sns.kdeplot(
    data=final_neuron_stats_df,
    x='non_reciprocal_ratio',
    ax=g.ax_marg_x,
    color='black',
    fill=True
)

sns.kdeplot(
    data=final_neuron_stats_df,
    y='reciprocal_ratio',
    ax=g.ax_marg_y,
    color='black',
    fill=True
)

# Add legend manually to joint plot
g.ax_joint.legend(loc='upper left')

# Add sample count
g.ax_joint.text(0.95, 0.95, f"n = {len(final_neuron_stats_df):,}",
                transform=g.ax_joint.transAxes, ha='right', va='top',
                fontsize=8, bbox=dict(facecolor='white', edgecolor='white'))

# Aesthetics
g.set_axis_labels("Synanpses/Partners - Non Reciprocal", "Synanpses/Partners - Reciprocal", fontsize=8)
g.ax_joint.tick_params(labelsize=8)
g.ax_marg_x.tick_params(labelsize=8)
g.ax_marg_y.tick_params(labelsize=8)

# Set custom ticks
g.ax_joint.set_xticks([0, 2, 4, 6, 8,10])
g.ax_joint.set_yticks([0, 2, 4, 6, 8,10])
# Clean spines
sns.despine(ax=g.ax_joint, top=True, right=True)
sns.despine(ax=g.ax_marg_x, top=True, right=True)
sns.despine(ax=g.ax_marg_y, top=True, right=True)

# Save
kde_path = _out_dir / "reciprocal_vs_nonreciprocal_kde_with_marginals_princeton.svg"
plt.tight_layout()
plt.savefig(kde_path, dpi=300, bbox_inches='tight')
plt.show()



#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Melt the two ratio columns into a long-form DataFrame
melted = final_neuron_stats_df.melt(
    value_vars=['non_reciprocal_ratio', 'reciprocal_ratio'],
    var_name='type'
)
#%%
final_neuron_stats_df['reciprocal_ratio'].std()
#%%
#%%
import matplotlib.pyplot as plt
import seaborn as sns
ratio_percent =  (
    final_neuron_stats_df['reciprocal_ratio'] - final_neuron_stats_df['non_reciprocal_ratio']
) / (
    final_neuron_stats_df['reciprocal_ratio'] + final_neuron_stats_df['non_reciprocal_ratio']
)

plt.figure(figsize=(1.5, 1.5))
ax= sns.histplot(ratio_percent.dropna(), bins=50, kde=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axvline(0, linestyle='--', color='black',linewidth=0.75)
#plt.title('Relative Strength (%) of Reciprocal vs Non-Reciprocal Synapses')
plt.xlabel('Symmetric Ratio')
plt.ylabel('Neurons')
plt.xlim(-1,1)
plt.xticks([-1,-0.5,0,0.5,1],size=8,labels=['-1','-0.5',"0","0.5","1"])
plt.yticks([0,3000,6000,9000],size=8)


plt.rc('axes', titlesize=8)         # Title size
plt.rc('axes', labelsize=8)         # X and Y label size
plt.rc('xtick', labelsize=8)        # X tick label size
plt.rc('ytick', labelsize=8)        # Y tick label size
plt.rc('legend', fontsize=8)        # Legend text size
plt.rc('legend', title_fontsize=8) 
plt.tight_layout()
fig_path = _out_dir / "hist_of_reci_non_reci_ratios_princeton.svg"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()


#%%
import scipy.stats as stats

diff = t1 - t2

# Histogram and KDE
plt.figure(figsize=(6, 4))
sns.histplot(diff.dropna(), kde=True, color='skyblue', bins=50)
plt.axvline(0, color='black', linestyle='--')
plt.title('Distribution of Differences (Recip − NonRecip)')
plt.xlabel('Difference')
plt.ylabel('Neuron Count')
plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
shapiro_test = stats.shapiro(diff.dropna())
print("Shapiro-Wilk p-value:", shapiro_test.pvalue)

t_stat, p_val = stats.ttest_rel(t1, t2, nan_policy='omit')
from scipy.stats import wilcoxon

w_stat, w_pval = wilcoxon(t1, t2)
print(f"Paired t-test p-value: {p_val:.10e}")
print(f"Wilcoxon signed-rank p-value: {w_pval:.10e}")


# Differences
diff = t1 - t2

# Combine into one DataFrame
desc_df = pd.DataFrame({
    'Reciprocal': t1,
    'Non-Reciprocal': t2,
    'Difference': diff
})

# Summary stats
summary = desc_df.describe().T[['mean', 'std', '50%']]
summary = summary.rename(columns={'50%': 'median'})
print(summary)
#%%
final_neuron_stats_df.reset_index(drop=True).to_feather(RECI_PROP_FTR)
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)
 #%%
final_neuron_stats_df['root_id']=final_neuron_stats_df['root_id'].astype(np.int64)
 
nodesG=nodesG.merge(final_neuron_stats_df,on='root_id',how='left')
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bin 'SI' into 100 bins
nodesG['SI_bin'] = pd.cut(nodesG['SI'], bins=100)

# Compute mean and std of 'ratio' per 'SI' bin
binned_stats = nodesG.groupby('SI_bin').agg(
    ratio_mean=('ratio', 'mean'),
    ratio_std=('ratio', 'std')
).reset_index()

# Get bin centers for x-axis
binned_stats['SI_bin_center'] = binned_stats['SI_bin'].apply(lambda x: x.mid)

# Plot
plt.figure(figsize=(2,1.8))
sns.lineplot(data=binned_stats, x='SI_bin_center', y='ratio_mean', label='Mean Ratio',linewidth=0.75)
plt.fill_between(
    binned_stats['SI_bin_center'],
    binned_stats['ratio_mean'] - binned_stats['ratio_std'],
    binned_stats['ratio_mean'] + binned_stats['ratio_std'],
    alpha=0.3,
    label='±1 SD'
)
plt.xticks([0,0.2,0.4,0.6,0.8,1],size=8)
plt.yticks([0,0.1,0.2,0.3,0.4],size=8)

plt.xlabel('SI (binned)')
plt.ylabel('Reciprocal Ratio')
plt.ylim(0,0.4)
sns.despine(top=True,right=True)
#plt.title('Ratio Mean and Standard Deviation over SI Bins')
plt.legend()
plt.tight_layout()
plt.savefig(_out_dir / "reci_vs_SI_princeton.svg")
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
custom_palette = { "ascending": "#6EB6F6", "visual_centrifugal": "#44733B", "descending": "#803D3D", "endocrine": "#8973B2"," ":'white', "motor": "#B48667", "optic": "#F4D826", "sensory": "#848484", "central": "#F9574E", "visual_projection": "#D5A848", }


# Freedman–Diaconis rule
def freedman_diaconis_bins(x):
    x = np.asarray(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    n = len(x)
    h = 2 * iqr / np.cbrt(n)
    if h == 0:
        return 10
    bins = int(np.ceil((x.max() - x.min()) / h))
    return max(1, bins)

# Decide global bins for SI
n_bins = freedman_diaconis_bins(nodesG['SI'].dropna())
nodesG['SI_bin'] = pd.cut(nodesG['SI'], bins=n_bins)

# Compute stats
binned_stats = (
    nodesG.groupby(['super_class_x','SI_bin'])
    .agg(
        ratio_mean=('ratio','mean'),
        ratio_std=('ratio','std')
    )
    .reset_index()
)
binned_stats['SI_bin_center'] = binned_stats['SI_bin'].apply(lambda x: x.mid)

# Select subclasses
selected_classes = ["optic", "central", "visual_projection", "visual_centrifugal"]

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(6,4), sharex=True, sharey=True)

for ax, sc in zip(axes.flat, selected_classes):
    df_sc = binned_stats[binned_stats['super_class_x'] == sc]
    color = custom_palette.get(sc, "grey")
    
    # Plot mean ± std (ratio vs SI)
    ax.plot(df_sc['SI_bin_center'], df_sc['ratio_mean'], color=color, linewidth=0.8)
    ax.fill_between(
        df_sc['SI_bin_center'],
        df_sc['ratio_mean'] - df_sc['ratio_std'],
        df_sc['ratio_mean'] + df_sc['ratio_std'],
        color=color, alpha=0.3
    )
    
    # Overlay smoothed SI distribution (secondary axis)
    ax2 = ax.twinx()
    sns.kdeplot(
        data=nodesG[nodesG['super_class_x']==sc],
        x="SI", ax=ax2, color="black", lw=0.6, alpha=0.6, fill=False
    )
    ax2.set_yticks([])
    ax2.set_ylabel("")
    ax2.set_ylim(0, None)  # auto-scale
    
    # Styling
    ax.set_title(sc, fontsize=9)
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_yticks([0,0.1,0.2,0.3,0.4])
    ax.tick_params(axis='both', labelsize=7)
    ax.set_ylim(0,0.4)
    sns.despine(ax=ax, top=True, right=True)
    sns.despine(ax=ax2, top=True, right=True, left=True)

# Shared labels
fig.text(0.5, 0.01, 'SI', ha='center', fontsize=9)
fig.text(0.01, 0.5, 'Reciprocal Ratio', va='center', rotation='vertical', fontsize=9)

plt.tight_layout(rect=[0.03, 0.03, 1, 1])
plt.savefig(_out_dir / "reci_vs_SI_FD_withDist.svg")
plt.show()

#%%%
import seaborn as sns
import matplotlib.pyplot as plt

# Sort by ratio for consistent rolling
nodesG_sorted = nodesG.sort_values('ratio')

# Compute rolling mean (e.g., window of 50)
window_size = 50
nodesG_sorted['SI_rolling'] = nodesG_sorted['SI'].rolling(window=window_size, center=True).mean()

# Plot
plt.figure(figsize=(6, 4))
sns.lineplot(data=nodesG_sorted, x='ratio', y='SI_rolling')
plt.xlabel('Ratio')
plt.ylabel('Rolling Mean SI')
plt.title('Rolling Mean of SI over Ratio')
plt.tight_layout()
plt.show()


#%%

final_neuron_stats_df = pd.read_feather(RECI_PROP_FTR)
final_neuron_stats_df['root_id']=final_neuron_stats_df['root_id'].astype(np.int64)
#%%
final_neuron_stats_df.rename(columns={'pre':'neuron'},inplace=True)
#%%
final_neuron_stats_df['neuron']=final_neuron_stats_df['neuron'].astype(np.int64)
#%%
aa=final_neuron_stats_df.merge(nodesG[['neuron','SI']],on='neuron',how='left')
#%%
a=final_neuron_stats_df.head()
#%%
corr=aa[['SI','ratio']].corr()
group_corr = aa.groupby("super_class")[["SI", "ratio"]].corr()

# Compute correlation of SI vs ratio inside each super_class_x
corrs = (
    aa.groupby("super_class")
      .apply(lambda g: g["SI"].corr(g["ratio"]))
      .reset_index(name="corr_SI_ratio")
)

# Add global correlation across all data
overall_corr = aa["SI"].corr(aa["ratio"])
corrs = pd.concat([corrs, pd.DataFrame([["All", overall_corr]], columns=["super_class_x", "corr_SI_ratio"])])

# Visualization
plt.figure(figsize=(5, 4))
sns.heatmap(corrs.set_index("super_class").T, 
            annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)

plt.title("Correlation (SI vs ratio) per super_class")
plt.ylabel("")
plt.show()
