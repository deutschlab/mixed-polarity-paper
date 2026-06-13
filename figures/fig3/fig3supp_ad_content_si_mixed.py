import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, SI_UPDATED_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

from scipy.stats import gaussian_kde
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
#allsynapses=allsynapses.head(1000000)
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG=nodesG.drop(columns=['SI'])
all_SI_df=pd.read_feather(SI_UPDATED_FTR)
all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')
a=nodesG[['neuron','SI','super_class','nodes','primary_type','axon_correct']]

#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])

        #%%
#%%
df=nodesG[['neuron','super_class','SI','axon_correct','dend_correct']]
#%%
df_comb=df[['axon_correct','dend_correct','SI']].sort_values(by='SI').dropna().reset_index(drop=True)

#%%

df_comb_axon=df[['axon_correct','SI']].sort_values(by='SI').dropna()

df_comb_axon['SI']*=10000
df_comb_g_axon=df_comb_axon.groupby(by='SI')['axon_correct'].mean().reset_index()
df_comb_g_axon['SI']=df_comb_g_axon['SI']/10000

df_comb_g_axon['rolling_correct'] = df_comb_g_axon['axon_correct'].rolling(window=400).mean()
df_comb_g_axon['rolling_correct']=df_comb_g_axon['rolling_correct']*100
#%%

df_comb_dend=df[['dend_correct','SI']].sort_values(by='SI').dropna()

df_comb_dend['SI']*=10000
df_comb_g_dend=df_comb_dend.groupby(by='SI')['dend_correct'].mean().reset_index()
df_comb_g_dend['SI']=df_comb_g_dend['SI']/10000

df_comb_g_dend['rolling_correct'] = df_comb_g_dend['dend_correct'].rolling(window=400).mean()
df_comb_g_dend['rolling_correct']=df_comb_g_dend['rolling_correct']*100
#%%

# --- Define SI grid ---
si_points = np.arange(0, 1.01, 0.05)

results = []

for si_point in si_points:

    # Find nearest rolling values
    axon_idx = (df_comb_g_axon['SI'] - si_point).abs().idxmin()
    dend_idx = (df_comb_g_dend['SI'] - si_point).abs().idxmin()

    axon_val = df_comb_g_axon.loc[axon_idx, 'rolling_correct']
    dend_val = df_comb_g_dend.loc[dend_idx, 'rolling_correct']

    results.append({
        'SI_point': si_point,
        'axon_correct': axon_val,
        'dend_correct': dend_val
    })


#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
nodesG=nodesG[['neuron','SI','axon_correct','dend_correct']]
#%%
thr_df = pd.DataFrame(results)  # columns: SI_point, axon_correct, dend_correct (these are thresholds in %)

rows = []

for _, r in thr_df.iterrows():

    si_point  = float(r['SI_point'])
    axon_thr  = float(r['axon_correct'])  # %
    dend_thr  = float(r['dend_correct'])  # %

    # ---- label neurons for this SI_point ----
    tmp_nodes = nodesG[['neuron','axon_correct','dend_correct']].copy()
    tmp_nodes['axon_label'] = np.where(tmp_nodes['axon_correct'] * 100 > axon_thr, 'A', 'M')
    tmp_nodes['dend_label'] = np.where(tmp_nodes['dend_correct'] * 100 > dend_thr, 'D', 'M')

    # ---- attach labels to synapses (pre=x, post=y) ----
    tmp = allsynapses[['synapse_id','pre','post','npil','comp']].copy()
    tmp = tmp[tmp['comp'].isin(['AD','DD','AA','DA'])].copy()

    tmp = tmp.merge(
        tmp_nodes[['neuron','axon_label','dend_label']],
        left_on='pre', right_on='neuron', how='left'
    ).rename(columns={'axon_label':'axon_label_x','dend_label':'dend_label_x'}).drop(columns=['neuron'])

    tmp = tmp.merge(
        tmp_nodes[['neuron','axon_label','dend_label']],
        left_on='post', right_on='neuron', how='left'
    ).rename(columns={'axon_label':'axon_label_y','dend_label':'dend_label_y'}).drop(columns=['neuron'])

    tmp = tmp.dropna(subset=['axon_label_x','dend_label_x','axon_label_y','dend_label_y'])

    # ---- comp2 reconstruction ----
    tmp['c1'] = tmp['comp'].str[0]
    tmp['c2'] = tmp['comp'].str[1]

    tmp['comp_first_corrected'] = np.where(
        tmp['c1'].eq('A'), tmp['axon_label_x'],
        np.where(tmp['c1'].eq('D'), tmp['dend_label_x'], 'O')
    )
    tmp['comp_second_corrected'] = np.where(
        tmp['c2'].eq('A'), tmp['axon_label_y'],
        np.where(tmp['c2'].eq('D'), tmp['dend_label_y'], 'O')
    )

    tmp['comp2'] = tmp['comp_first_corrected'] + tmp['comp_second_corrected']

    # ---- summarize (fractions; switch to counts if you want) ----
    counts = tmp['comp2'].value_counts()
    fracs  = counts / counts.sum()

    row = {
        'SI_point': si_point,
        'axon_thr_percent': axon_thr,
        'dend_thr_percent': dend_thr,
        'n_syn': int(counts.sum())
    }
    for k, v in fracs.items():
        row[f'frac_{k}'] = float(v)

    rows.append(row)
#%%
df_comp2_by_SI = pd.DataFrame(rows).fillna(0)
#%%

base = allsynapses.loc[allsynapses['comp'].isin(['AD','AA','DD','DA']), 'comp'].value_counts()
base_fracs = (base / base.sum()).to_dict()
base_n = int(base.sum())

# ensure needed columns exist
for k in base_fracs.keys():
    col = f'frac_{k}'
    if col not in df_comp2_by_SI.columns:
        df_comp2_by_SI[col] = 0.0

# overwrite the SI=0 row
idx0 = df_comp2_by_SI.index[np.isclose(df_comp2_by_SI['SI_point'].astype(float), 0.0)][0]
for c in [c for c in df_comp2_by_SI.columns if c.startswith('frac_')]:
    df_comp2_by_SI.loc[idx0, c] = 0.0

for k, v in base_fracs.items():
    df_comp2_by_SI.loc[idx0, f'frac_{k}'] = float(v)

df_comp2_by_SI.loc[idx0, 'n_syn'] = base_n
df_comp2_by_SI.loc[idx0, 'axon_thr_percent'] = 0.0
df_comp2_by_SI.loc[idx0, 'dend_thr_percent'] = 0.0

#%%


df_plot = df_comp2_by_SI[['SI_point']].copy()
df_plot['AD_ratio'] = df_comp2_by_SI.get('frac_AD', 0.0)
df_plot = df_plot.sort_values('SI_point').reset_index(drop=True)

x = df_plot['SI_point']
y = df_plot['AD_ratio']

plt.figure(figsize=(8,5))

# dashed part: SI <= 0.10
mask = x <= 0.10
plt.plot(x[mask], y[mask], linestyle='--', marker='o', color='grey')

# solid part: from last dashed point onward
if mask.any():
    idx = x[mask].index[-1]
    plt.plot(x.loc[idx:], y.loc[idx:], marker='o', color='blue')
else:
    plt.plot(x, y, marker='o', color='blue')

plt.xlim(0, 0.7)
plt.ylim(0, 1)

plt.xlabel("SI th")
plt.ylabel("AD proportion")
plt.title("AD proportion vs SI threshold")
plt.grid(False)
sns.despine(top=True, right=True)
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig3" / "supp_ad_content_si_mixed"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_th_mixed.svg")
plt.show()


#%% ============================================================
# 3) Pie per SI_point (comp2 distribution) — no saving
# ============================================================

custom_palette = {
    'AA': '#8b2be2',
    'AM': '#e5d0ff',
    'AD': '#9F4800',
    'DA': '#B3B3B3',
    'DD': '#F4B95A',
    'MD': '#fff2cc',
    'MM': 'darkgrey'
}

main_keys = ['AM', 'AA', 'AD', 'DA', 'DD', 'MD', 'MM']
frac_cols = [c for c in df_comp2_by_SI.columns if c.startswith('frac_')]

for _, r in df_comp2_by_SI.sort_values('SI_point').iterrows():

    si_point = float(r['SI_point'])

    # comp2 fractions
    comp_fracs = {c.replace('frac_', ''): float(r[c]) for c in frac_cols}
    comp_fracs = {k: v for k, v in comp_fracs.items() if v > 0}

    if len(comp_fracs) == 0:
        continue

    # order: main then others
    other_keys = sorted([k for k in comp_fracs.keys() if k not in main_keys])
    ordered_labels = [k for k in main_keys if k in comp_fracs] + other_keys
    values = np.array([comp_fracs[k] for k in ordered_labels], dtype=float)

    colors = [custom_palette.get(lbl, '#9E9E9E') for lbl in ordered_labels]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(
        values,
        labels=ordered_labels,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p >= 1 else '',
        startangle=90,
        counterclock=False,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )
    ax.axis('equal')

    ax.set_title(
        f"SI={si_point:.2f} | ax_thr={float(r['axon_thr_percent']):.1f}% | "
        f"dend_thr={float(r['dend_thr_percent']):.1f}% | n={int(r['n_syn'])}"
    )

    plt.tight_layout()
    plt.show()