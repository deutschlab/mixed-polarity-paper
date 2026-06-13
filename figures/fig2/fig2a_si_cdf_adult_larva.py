import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SI_UPDATED_FTR, LARVA_SYNAPSES_FTR, LARVA_SI_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
#allsynapses=allsynapses.head(1000000)
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.drop(columns=['SI'])
all_SI_df=pd.read_feather(SI_UPDATED_FTR)
all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])


nodesG['correct_pos']=  ((nodesG['pre_A']  +  nodesG['post_D']    )/(nodesG['pre_A']  +  nodesG['post_A']+nodesG['pre_D']  +  nodesG['post_D']  ) )
df=nodesG[['neuron','super_class','SI','correct_pos']]
df_comb=df[['correct_pos','SI']].sort_values(by='SI').dropna().reset_index(drop=True)

df_comb=df[['correct_pos','SI']].sort_values(by='SI').dropna()

df_comb['SI']*=10000
df_comb_g=df_comb.groupby(by='SI')['correct_pos'].mean().reset_index()
df_comb_g['SI']=df_comb_g['SI']/10000

df_comb_g['rolling_correct'] = df_comb_g['correct_pos'].rolling(window=400).mean()
df_comb_g['rolling_correct']=df_comb_g['rolling_correct']*100


connectors=pd.read_feather(LARVA_SYNAPSES_FTR)
connectors['comp']=connectors['pre_compartment']+connectors['post_compartment']

SI_larva=pd.read_feather(LARVA_SI_FTR)
SI_larva.columns=['neuron','SI','linker']
SI_larva['neuron']=SI_larva['neuron'].astype(np.int64)

connectors=connectors.merge(SI_larva,left_on='pre',right_on='neuron',how='left')
connectors=connectors.merge(SI_larva,left_on='post',right_on='neuron',how='left')
pre_g=connectors.groupby(by=['pre','pre_compartment']).size().reset_index()
pre_g = pd.pivot_table(
    pre_g,                   # ← not pre_g
    values='pre_compartment',
    index='pre',
    columns='pre_compartment',
    aggfunc='sum',
)

post_g=connectors.groupby(by=['post','post_compartment']).size().reset_index()
post_g = pd.pivot_table(
    post_g,                   # ← not pre_g
    values='post_compartment',
    index='post',
    columns='post_compartment',
    aggfunc='sum',
)
pre_g=pre_g.reset_index()
pre_g.columns=['neuron','pre_A','pre_D','pre_L']
post_g=post_g.reset_index()
post_g.columns=['neuron','post_A','post_D','post_L']
merged=pre_g.merge(post_g)

merged['axon_percent']=merged['pre_A']/(merged['post_A']+merged['pre_A'])*100
merged['dend_percent']=merged['post_D']/(merged['post_A']+merged['post_D'])*100
merged=merged.dropna(subset=['axon_percent','dend_percent'])

merged['correct_pos']=(merged['pre_A']+merged['post_D'])/(merged['post_A']+merged['pre_A']+merged['post_A']+merged['post_D'])*100

SI_larva=SI_larva.merge(merged)
df_comb_larva=SI_larva[['correct_pos','SI']].sort_values(by='SI').dropna()


df_comb_larva['SI']*=10000
df_comb_larva_g=df_comb_larva.groupby(by='SI')['correct_pos'].mean().reset_index()
df_comb_larva_g['SI']=df_comb_larva_g['SI']/10000

df_comb_larva_g['rolling_correct'] = df_comb_larva_g['correct_pos'].rolling(window=50).mean()
df_comb_larva_g['rolling_correct']=df_comb_larva_g['rolling_correct']


def freedman_diaconis_bins(data):
    data = np.asarray(data)
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    n = len(data)
    bin_width = 2 * iqr / np.cbrt(n)
    if bin_width <= 0:
        return int(np.sqrt(n))  # fallback
    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return n_bins

# Compute bins for larva data and nodesG
bins_larva = freedman_diaconis_bins(SI_larva['SI'])
bins_nodes = freedman_diaconis_bins(nodesG['SI'])

#%% FIG 2: SI distributions (Adult + Larva) + CDFs (dual y-axis, clear labeling)

fig, ax = plt.subplots(figsize=(1.9, 1.5))

# ---------- Background histograms (visual only, no Y-axis) ----------
ax_hist = ax.twinx()

# Adult distribution
sns.histplot(
    data=nodesG['SI'],
    bins=bins_nodes,
    color='blue',
    ax=ax_hist,
    stat='density',     # normalized just for shape
    element='step',
    fill=True,
    alpha=0.15,
    linewidth=0.3,
    label='_nolegend_'
)

# Larva distribution
sns.histplot(
    data=SI_larva['SI'],
    bins=bins_larva * 3,
    color='#F4B95A',    # light orange
    ax=ax_hist,
    stat='density',
    element='step',
    fill=True,
    alpha=0.25,         # slightly more opaque for visibility
    linewidth=0.3,
    label='_nolegend_'
)

# Hide Y-axis for histograms
ax_hist.get_yaxis().set_visible(True)
ax_hist.set_ylim(0, 7)

# ---------- Main CDFs ----------
cdf_values_adult = np.sort(nodesG['SI'].dropna())
cdf_y_adult = np.arange(1, len(cdf_values_adult) + 1) / len(cdf_values_adult) * 100
ax.plot(
    cdf_values_adult, cdf_y_adult,
    color='blue', lw=0.6, label=f'Adult (n={len(nodesG)})'
)

cdf_values_larva = np.sort(SI_larva['SI'].dropna())
cdf_y_larva = np.arange(1, len(cdf_values_larva) + 1) / len(cdf_values_larva) * 100
ax.plot(
    cdf_values_larva, cdf_y_larva,
    color='#F4B95A', lw=0.6, label=f'Larva (n={len(SI_larva)})'
)

# ---------- Labels and layout ----------
ax.set_xlabel('Segregation Index (SI)')
ax.set_ylabel('Cumulative % of neurons')
ax.set_xlim(0, 1)
ax.set_xticks([0,0.25,0.5,0.75,1])
ax.set_ylim(0, 100)

ax.legend(fontsize=4, loc='upper left', frameon=False)
sns.despine(ax=ax, right=False, left=False)
plt.tight_layout()

_out_dir = OUTPUT_DIR / "fig2" / "SI_CDF_adult_larva"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    _out_dir / "SI_dist_and_CDF_single_yaxis_v2.svg",
    format='svg'
)
plt.show()

# ======================
#  NORMALITY CHECK
# ======================
adult_SI = nodesG['SI'].dropna()
larva_SI = SI_larva['SI'].dropna()

p_adult = shapiro(adult_SI.sample(5000, random_state=0) if len(adult_SI) > 5000 else adult_SI)[1]
p_larva = shapiro(larva_SI)[1]
print(f"Adult SI normality p = {p_adult:.3e}")
print(f"Larva SI normality p = {p_larva:.3e}")

# ======================
#  MANN–WHITNEY U TEST (non-parametric)
# ======================
U, p_mw = mannwhitneyu(adult_SI, larva_SI, alternative='two-sided')
print(f"\nMann–Whitney U test:\nStatistic = {U:.3f}, p-value = {p_mw:.3e}")

# Rank-biserial correlation (effect size)
rbc = 1 - (2 * U) / (len(adult_SI) * len(larva_SI))
print(f"Rank-biserial correlation (effect size): r = {rbc:.3f}")

# ======================
#  PARAMETRIC CHECK (Welch’s t-test after transformation)
# ======================
asi_adult = np.arcsin(np.sqrt(adult_SI))
asi_larva = np.arcsin(np.sqrt(larva_SI))
t_stat, p_t = ttest_ind(asi_adult, asi_larva, equal_var=False)
print(f"\nWelch t-test (arcsine sqrt transformed): t = {t_stat:.3f}, p = {p_t:.3e}")

# Cohen's d (parametric effect size)
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp2 = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    return (np.mean(x) - np.mean(y)) / np.sqrt(sp2)

d = cohens_d(asi_adult, asi_larva)
print(f"Cohen's d (arcsine sqrt): d = {d:.3f}")

# ======================
#  INTERPRETATION SUMMARY
# ======================
if p_mw < 0.05:
    print("\n✅ Significant difference between Adult and Larva SI distributions (Mann–Whitney).")
else:
    print("\n❌ No significant difference (Mann–Whitney).")

if p_t < 0.05:
    print("✅ Welch t-test (after transform) also significant.")
else:
    print("❌ Welch t-test not significant.")
