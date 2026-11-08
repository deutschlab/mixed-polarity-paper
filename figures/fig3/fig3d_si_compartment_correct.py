import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, SI_UPDATED_FTR, OUTPUT_DIR, METHODS_DIR,
                    NEURON_TABLE_NONP_FTR, SYNAPSE_TABLE_NONP_FTR)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

from scipy.stats import gaussian_kde
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved
# DATA_SOURCE "princeton" (Fig 3D-3F): the Princeton synapses, each cell's SI from SI_UPDATED_FTR.
# DATA_SOURCE "buhmann" (Supp 3-S2 A-B): the Buhmann synapses and neuron table (NEURON_TABLE_NONP_FTR,
# SYNAPSE_TABLE_NONP_FTR), whose SI is the cell's SI on the Buhmann synapses; written to a folder of its own.
# FIG3D_DATA_SOURCE in the environment overrides the setting.
DATA_SOURCE = os.environ.get("FIG3D_DATA_SOURCE", "princeton")
if DATA_SOURCE not in ("princeton", "buhmann"):
    raise SystemExit(f'DATA_SOURCE must be "princeton" or "buhmann", not {DATA_SOURCE!r}')
#allsynapses=allsynapses.head(1000000)
if DATA_SOURCE == "princeton":
    nodesG=pd.read_feather(NEURON_TABLE_FTR)
    #%%
    nodesG=nodesG.drop(columns=['SI'])
    all_SI_df=pd.read_feather(SI_UPDATED_FTR)
    all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
    nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')
else:
    nodesG=pd.read_feather(NEURON_TABLE_NONP_FTR).dropna(subset='SI')
    assert nodesG['neuron'].dtype == np.int64 and nodesG['neuron'].is_unique
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
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns
custom_palette = {
    'axon_correct': '#92278f',
    'dend_correct': '#fbb042'
}
fig, ax1 = plt.subplots(figsize=(8, 4))

# --- Plot SI distributions ---
sns.histplot(nodesG['SI'], bins=60, color='blue', alpha=0.3, ax=ax1, label='SI dist')

ax1.set_xlabel('Segregation Index (SI)')
ax1.set_ylabel('Cells')
ax1.legend(loc='upper left')

# --- Plot rolling correctness ---
# The drawn curves are smoothed for display only: at each cell, the mean over the cells whose SI lies within
# DISPLAY_HALF_WIDTH of it, so the smoothing has the same width in SI everywhere (a fixed number of cells would span a
# much wider SI range where cells are few, near SI 0.5 and above). The cut-offs at SI 0.1 (axon_val, dend_val, below)
# still come from the trailing 400-cell mean above, so the old-rule labels and every number they feed are unchanged;
# the drawn curves differ from them at SI 0.1 by well under a point.
DISPLAY_HALF_WIDTH = 0.0125
ax2 = ax1.twinx()
for col, color, label in [('axon_correct', '#92278f', 'Axon rolling correct (%)'),
                          ('dend_correct', '#fbb042', 'Dendrite rolling correct (%)')]:
    d = df[[col, 'SI']].dropna().sort_values('SI').reset_index(drop=True)
    si, val = d['SI'].to_numpy(), d[col].to_numpy()
    lo = np.searchsorted(si, si - DISPLAY_HALF_WIDTH, side='left')
    hi = np.searchsorted(si, si + DISPLAY_HALF_WIDTH, side='right')
    csum = np.concatenate([[0.0], np.cumsum(val)])
    smooth = (csum[hi] - csum[lo]) / (hi - lo) * 100
    ax2.plot(si, smooth, color=color, lw=2, label=label)

ax2.set_ylabel('Rolling mean of correctness (%)')
ax2.legend(loc='upper right')

# --- Define SI point ---
si_point = 0.1

# --- Find nearest rolling values ---
axon_idx = (df_comb_g_axon['SI'] - si_point).abs().idxmin()
dend_idx = (df_comb_g_dend['SI'] - si_point).abs().idxmin()

axon_val = df_comb_g_axon.loc[axon_idx, 'rolling_correct']
dend_val = df_comb_g_dend.loc[dend_idx, 'rolling_correct']

# --- Draw vertical lines and points ---
ax2.axvline(si_point, color='gray', linestyle='--', lw=1.2, alpha=0.7)
ax2.vlines(si_point, ymin=min(axon_val, dend_val), ymax=max(axon_val, dend_val),
           colors='gray', linestyle='--', lw=1, alpha=0.6)

ax2.scatter(si_point, axon_val, color='black', s=50, zorder=5)
ax2.scatter(si_point, dend_val, color='black', s=50, zorder=5)
ax2.axhline(axon_val, color='#92278f', linestyle='--', lw=1, alpha=0.7)
ax2.axhline(dend_val, color='#fbb042', linestyle='--', lw=1, alpha=0.7)

# --- Annotate each value ---
ax2.text(0.45, axon_val, f'Axon: {axon_val:.2f}%', color='#92278f', va='bottom',size=15)   # inside the axes (x limit 0.6)
ax2.text(0.45, dend_val-0.5, f'Dend: {dend_val:.2f}%', color='#fbb042', va='bottom',size=15)

# --- Titles and layout ---
plt.title('SI Distribution and Rolling Correctness (Axon vs Dendrite)')
plt.tight_layout()
plt.xlim(0,0.6)
_out_dir = OUTPUT_DIR / "fig3" / "SI_x_correct_compartment"
if DATA_SOURCE == "buhmann":
    _out_dir = OUTPUT_DIR / "fig3" / "SI_x_correct_compartment_buhmann_si_rule"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "axon_dend_correct_SI_v2.svg")
plt.show()

print(f"Rolling correctness at SI = {si_point}:")
print(f"  Axon → {axon_val:.2f}%")
print(f"  Dendrite → {dend_val:.2f}%")

#%%
if DATA_SOURCE == "princeton":
    allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
else:
    allsynapses=pd.read_feather(SYNAPSE_TABLE_NONP_FTR).rename(columns={'id': 'synapse_id'})
#%%
cell_type = nodesG.set_index('neuron')['primary_type']   # for the 3F pies below
nodesG=nodesG[['neuron','SI','axon_correct','dend_correct']]
#%%
# MIXED_RULE "si" (the main figure): a cell with SI below 0.1 counts as mixed, in both its axon and its dendrite.
# MIXED_RULE "cutoffs" (the supplement): a compartment counts as mixed when its share of correct synapses is not
# above the rolling curve's value at SI 0.1 (axon_val for axons, dend_val for dendrites).
MIXED_RULE = "si"
if MIXED_RULE == "si":
    nodesG['axon_label'] = np.where(nodesG['SI'] >= 0.1, 'A', 'M')
else:
    nodesG['axon_label'] = nodesG['axon_correct'].apply(lambda x: 'A' if x*100 > axon_val else 'M')

# Verify result
print(nodesG['axon_label'].value_counts())


if MIXED_RULE == "si":
    nodesG['dend_label'] = np.where(nodesG['SI'] >= 0.1, 'D', 'M')
else:
    nodesG['dend_label'] = nodesG['dend_correct'].apply(lambda x: 'D' if x*100 > dend_val else 'M')

# Verify result
print(nodesG['dend_label'].value_counts())
#%%
allsynapses=allsynapses.merge(nodesG[['neuron','dend_label','axon_label']],left_on='pre',right_on='neuron',how='left')
#%%
allsynapses=allsynapses.merge(nodesG[['neuron','dend_label','axon_label']],left_on='post',right_on='neuron',how='left')
#%%

allsynapses=allsynapses.dropna(subset=['neuron_x','neuron_y'])
#%%
allsynapses = allsynapses[allsynapses['comp'].isin(['AD', 'DD', 'AA', 'DA'])]
#%%
aa=allsynapses.query('SI_pre>=0.1 and SI_post>0.1')

#%%
allsynapses['comp_first']=allsynapses['comp'].str[0]
allsynapses['comp_second']=allsynapses['comp'].str[1]
#%%
aaa=allsynapses.head(1000)
#%%
#%%
# each side's compartment label: the cell's axon label on an A side, its dendrite label on a D side
# ('O' for any other side, L); vectorised, so the whole synapse table fits in memory
allsynapses['comp_first_corrected'] = np.select(
    [allsynapses['comp_first'] == 'A',
     allsynapses['comp_first'] == 'D'],
    [allsynapses['axon_label_x'],
     allsynapses['dend_label_x']],
    default='O'
)

#%%
# the same for the second side
allsynapses['comp_second_corrected'] = np.select(
    [allsynapses['comp_second'] == 'A',
     allsynapses['comp_second'] == 'D'],
    [allsynapses['axon_label_y'],
     allsynapses['dend_label_y']],
    default='O'
)

#%%
allsynapses2=allsynapses.head(10000)
#%%
allsynapses['comp2']=allsynapses['comp_first_corrected']+allsynapses['comp_second_corrected']
# GROUP_MIXED = True: every synapse with a mixed end (AM, MA, MD, DM, MM) is one category, "Mixed"
GROUP_MIXED = True
if GROUP_MIXED:
    allsynapses.loc[allsynapses['comp2'].str.contains('M'), 'comp2'] = 'Mixed'
#%%
allsynapses=allsynapses[['synapse_id','pre', 'post', 'post_x', 'post_y', 'post_z', 'pre_x', 'pre_y', 'pre_z',
       'npil', 'comp2','SI_pre','SI_post' ]]
#%%

#%%

#%%
import matplotlib.pyplot as plt

# Count occurrences of each combination
comp_counts = allsynapses['comp2'].value_counts()

# Plot pie chart
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(
    comp_counts,
    labels=comp_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False
)
ax.set_title('Distribution of comp2 combinations')
plt.tight_layout()
plt.savefig(_out_dir / "synaptic_type_pie_bign_v2.svg")

plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# --- Custom color palette ---
# --- Custom color palette (lighter AM/MD, darker greys for others) ---
custom_palette = {
    'AA': '#8b2be2',   # deep purple
    'AM': '#e5d0ff',   # very light lavender
    'AD': '#9F4800',   # brown
    'DA': '#B3B3B3',   # medium gray
    'DD': '#F4B95A',   # gold
    'MD': '#fff2cc'    # very light pastel gold,
    ,'MM': 'darkgrey'
    , 'Mixed': 'black'
}

# Darker gray for "other" categories
colors = [custom_palette.get(label, '#4F4F4F') for label in comp_counts.index]


# Use darker grey for "other" categories

# --- Count values ---
comp_counts = allsynapses['comp2'].value_counts()

# --- Order categories to keep AM next to AA and MD next to DD ---

main_keys = ['AM', 'AA', 'AD', 'DA', 'DD', 'MD']
other_keys = [k for k in comp_counts.index if k not in main_keys]
ordered_labels = main_keys + other_keys
comp_counts = comp_counts.reindex(ordered_labels).dropna()

# --- Assign colors: custom for main categories, darker grey for others ---
colors = [custom_palette.get(label, '#9E9E9E') for label in comp_counts.index]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    comp_counts,
    labels=comp_counts.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'}  # thicker white lines between slices
)
for label, txt, pct in zip(comp_counts.index, texts, autotexts):
    if label in ('Mixed', 'AA', 'AD'):
        pct.set_color('white')   # readable on the dark slices
    elif label == 'DA':
        txt.set_text(f'DA {pct.get_text()}')   # thin slice: the percentage goes beside its label
        pct.set_visible(False)
plt.savefig(_out_dir / "synaptic_type_pie2_v2.svg", bbox_inches='tight')

plt.show()

#%% 3F's two pies with mixed ends (Dudi's plan, row 159): every synapse between the cells above, each end labelled
# by MIXED_RULE, split into pairs of the same primary_type and pairs of different types. No SI filter: the mixed
# label takes its place.
same_type = cell_type.reindex(allsynapses['pre']).to_numpy() == cell_type.reindex(allsynapses['post']).to_numpy()
for name, keep in (("syn_type_same_ptype_pie_mixed.svg", same_type),
                   ("syn_type_diff_ptype_pie_mixed.svg", ~same_type)):
    counts = allsynapses.loc[keep, 'comp2'].value_counts()
    order = [k for k in ['AA', 'AD', 'DA', 'DD'] if k in counts.index]
    counts = counts.reindex(order + [k for k in counts.index if k not in order])
    fig, ax = plt.subplots(figsize=(4, 4))
    _, _, pcts = ax.pie(counts, labels=counts.index, colors=[custom_palette.get(k, '#4F4F4F') for k in counts.index],
                        autopct='%1.1f%%', startangle=90, counterclock=False,
                        wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
    for label, pct in zip(counts.index, pcts):
        if label == 'Mixed':
            pct.set_color('white')   # readable on the black slice
    ax.set_title(f"{'Same' if name.startswith('syn_type_same') else 'Different'} type connections, "
                 f"Total Syn: {int(counts.sum()):,}")
    plt.tight_layout()
    plt.savefig(_out_dir / name)
    plt.close(fig)
    print(name, int(counts.sum()), (counts / counts.sum() * 100).round(1).to_dict())


    #%%
allsynapses['npil'] = allsynapses['npil'].str.replace(r'_(L|R)$', '', regex=True)
# synapses outside every neuropil (npil 'None', in the Buhmann table) get no bar
_no_npil = allsynapses['npil'].isna() | (allsynapses['npil'] == 'None')
print(f"synapses outside every neuropil, left out of the neuropil bars: {int(_no_npil.sum()):,}")
allsynapses = allsynapses[~_no_npil]

#%%
#allsynapses.to_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\article\synapses_783_article_princeton_mixed.ftr')

#%%
aaa=allsynapses.head(1000)
#%%
import matplotlib.pyplot as plt
import pandas as pd

# --- Custom color palette (same as before, with fallback for others) ---
custom_palette = {
    'AA': '#8b2be2',   # deep purple
    'AM': '#e5d0ff',   # very light lavender
    'AD': '#9F4800',   # brown
    'DA': '#B3B3B3',   # medium gray
    'DD': '#F4B95A',   # gold
    'MD': '#fff2cc' ,
    'MM':'black'
    
    , 'Mixed': 'black'
}
default_color = '#4F4F4F'  # darker grey for other categories

# --- Desired order ---
order = ['AD', 'AA', 'DD', 'DA', 'AM', 'MD', 'DM', 'MA','MM']

# --- Count comp2 per npil ---
# --- Count comp2 per npil ---
comp_counts = allsynapses.groupby(['npil', 'comp2']).size().unstack(fill_value=0)

# --- Normalize to percentages ---
comp_counts_pct = comp_counts.div(comp_counts.sum(axis=1), axis=0) * 100

# --- Reorder columns according to desired order ---
existing = [c for c in order if c in comp_counts_pct.columns]
remaining = [c for c in comp_counts_pct.columns if c not in order]
comp_counts_pct = comp_counts_pct[existing + remaining]

# --- Sort rows (npils) by descending AD percentage ---
if 'AD' in comp_counts_pct.columns:
    comp_counts_pct = comp_counts_pct.sort_values(by='AD', ascending=False)

# --- Build color list ---
colors = [custom_palette.get(c, default_color) for c in comp_counts_pct.columns]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
comp_counts_pct.plot(
    kind='bar',
    stacked=True,
    color=colors,
    ax=ax,
    width=0.8,
    edgecolor='white',
    linewidth=0.7
)

ax.set_ylabel('Percentage of comp2 combinations')
ax.set_xlabel('Neuropil (npil)')
ax.set_title('Stacked Distribution of comp2 per npil (sorted by AD%)')
ax.legend(title='comp2', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(_out_dir / "synaptic_type_npils_bign_v2.svg")

plt.show()

#%%




import matplotlib.pyplot as plt
import pandas as pd

# --- Custom comp2 color palette (same as before) ---
custom_palette = {
    'AA': '#8b2be2',   # deep purple
    'AM': '#e5d0ff',   # very light lavender
    'AD': '#9F4800',   # brown
    'DA': '#B3B3B3',   # medium gray
    'DD': '#F4B95A',   # gold
    'MD': '#fff2cc',   # very light pastel gold
    'MM': 'black'
    , 'Mixed': 'black'
}
default_color = '#4F4F4F'  # darker grey for other categories

# --- Desired order ---
order = ['AD', 'AA', 'DD', 'DA', 'AM', 'MD', 'DM', 'MA', 'MM']

# --- Count comp2 per npil ---
comp_counts = allsynapses.groupby(['npil', 'comp2']).size().unstack(fill_value=0)

# --- Normalize to percentages ---
comp_counts_pct = comp_counts.div(comp_counts.sum(axis=1), axis=0) * 100

# --- Reorder columns according to desired order ---
existing = [c for c in order if c in comp_counts_pct.columns]
remaining = [c for c in comp_counts_pct.columns if c not in order]
comp_counts_pct = comp_counts_pct[existing + remaining]

# --- Sort rows (npils) by descending AD percentage ---
if 'AD' in comp_counts_pct.columns:
    comp_counts_pct = comp_counts_pct.sort_values(by='AD', ascending=False)

# --- Build comp2 colors ---
colors = [custom_palette.get(c, default_color) for c in comp_counts_pct.columns]

# --- Neuropil color families ---
npil_colors = {
    # Blue family
    'FB': '#00bfbf', 'EB': '#00cccc', 'PB': '#0099cc', 'NO': '#0073b2',
    'AMMC': '#4040ff', 'FLA': '#3366ff', 'CAN': '#3366ff', 'PRW': '#3333ff',
    'SAD': '#3333ff', 'GNG': '#3333ff', 'AL': '#3399ff', 'LH': '#3366ff', 'BU': '#3333ff',

    # Orange family
    'MB-CA': '#ff9966', 'MB-PED': '#ff9966', 'MB-VL': '#ff9966',
    'MB-VLCP': '#ff9966', 'MB-ML': '#ff9966',

    # Yellow family
    'LAL': '#ffcc33', 'SLP': '#ffcc33', 'SIP': '#ffcc33', 'SMP': '#ffcc33',
    'CRE': '#ffcc33', 'IB': '#ffcc33', 'ATL': '#ffcc33',

    # Green family
    'VES': '#33cc66', 'EPA': '#00cc66', 'GOR': '#00cc66', 'SPS': '#00cc66',
    'IPS': '#00cc66', 'AOTU': '#00cc66',

    # Light blue family
    'AVLP': '#3399ff', 'PVLP': '#3399ff', 'PLP': '#3399ff', 'WED': '#3399ff',

    # Purple family
    'ME': '#cc3399', 'AME': '#cc3399', 'LO': '#9933cc',
    'LOP': '#9933cc', 'LA': '#9933cc', 'OCG': '#9933cc'
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
comp_counts_pct.plot(
    kind='bar',
    stacked=True,
    color=colors,
    ax=ax,
    width=0.8,
    edgecolor='white',
    linewidth=0.7
)

# --- Style ---
ax.set_ylabel('Percentage of comp2 combinations')
ax.set_xlabel('Neuropil (npil)')
ax.set_title('Stacked Distribution of comp2 per npil (sorted by AD%)')
ax.legend(title='comp2', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Color x-axis labels by neuropil family ---
for label in ax.get_xticklabels():
    npil_name = label.get_text()
    label.set_color(npil_colors.get(npil_name, 'black'))
    label.set_rotation(45)
    label.set_ha('right')

plt.tight_layout()
plt.savefig(_out_dir / "synaptic_type_npils2_v2.svg")

plt.show()




#%%
import matplotlib.pyplot as plt
import pandas as pd

# --- Custom color palette (same shades as before) ---
custom_palette = {
    'AA': '#8b2be2',   # deep purple
    'AD': '#9F4800',   # brown
    'DD': '#F4B95A',   # gold
    'DA': '#B3B3B3'    # medium gray
}

# --- Keep only these four ---
keep_keys = ['AD', 'AA', 'DD', 'DA']

# --- Count comp2 per npil ---
comp_counts = allsynapses.groupby(['npil', 'comp2']).size().unstack(fill_value=0)

# --- Filter to only relevant columns ---
comp_counts = comp_counts[[c for c in keep_keys if c in comp_counts.columns]]

# --- Normalize to percentages ---
comp_counts_pct = comp_counts.div(comp_counts.sum(axis=1), axis=0) * 100

# --- Sort by descending AD percentage ---
if 'AD' in comp_counts_pct.columns:
    comp_counts_pct = comp_counts_pct.sort_values(by='AD', ascending=False)

# --- Build color list ---
colors = [custom_palette.get(c, '#4F4F4F') for c in comp_counts_pct.columns]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
comp_counts_pct.plot(
    kind='bar',
    stacked=True,
    color=colors,
    ax=ax,
    width=0.8,
    edgecolor='white',
    linewidth=0.7
)

ax.set_ylabel('Percentage of comp2 combinations')
ax.set_xlabel('Neuropil (npil)')
ax.set_title('Stacked Distribution of comp2 per npil (AD, AA, DD, DA)')
ax.legend(title='comp2', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(_out_dir / "synaptic_type_npils3_v2.svg")

plt.show()
#%%
