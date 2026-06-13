import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
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
a=nodesG[['neuron','SI','super_class','nodes','primary_type','axon_correct']]
#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
df=nodesG[['neuron','super_class','SI','axon_correct','dend_correct']]
#%%
df_comb=df[['axon_correct','dend_correct','SI']].sort_values(by='SI').dropna().reset_index(drop=True)
#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
nodesG=nodesG[['neuron','SI','axon_correct','dend_correct']]

#%%
allsynapses=allsynapses.merge(nodesG[['neuron']],left_on='pre',right_on='neuron',how='left')
#%%
allsynapses=allsynapses.merge(nodesG[['neuron']],left_on='post',right_on='neuron',how='left')
#%%
aa=allsynapses.head(1000)

#%%
allsynapses=allsynapses.dropna(subset=['neuron'])
#%%
allsynapses=allsynapses.drop(columns=['neuron'])
#%%
allsynapses = allsynapses[allsynapses['comp'].isin(['AD', 'DD', 'AA', 'DA'])]
#%%
allsynapses=allsynapses[['synapse_id', 'pre', 'post', 'npil', 'comp', 'SI_pre', 'SI_post']]
#%%
allsynapses['SI_pre_binary']=allsynapses['SI_pre'].apply(lambda x: 0 if x<0.1 else 1)
#%%
allsynapses['SI_post_binary']=allsynapses['SI_post'].apply(lambda x: 0 if x<0.1 else 1)

#%%
allsynapses['comp_first']=allsynapses['comp'].str[0]
allsynapses['comp_second']=allsynapses['comp'].str[1]
#%%
#%%
#%%
allsynapses['comp2'] = (
    np.where(
        allsynapses['SI_pre_binary'] == 0,
        'M',
        allsynapses['comp'].str[0]
    )
    +
    np.where(
        allsynapses['SI_post_binary'] == 0,
        'M',
        allsynapses['comp'].str[1]
    )
)



#%%
aaa=allsynapses.head(1000)
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
_out_dir = OUTPUT_DIR / "fig3" / "syntype_composition"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "synaptic_type_pie_bign.svg")

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
plt.xlim(0,0.6)
ax.set_title('Distribution of comp2 combinations')
plt.tight_layout()
plt.savefig(_out_dir / "synaptic_type_pie2.svg")

plt.show()


    #%%
allsynapses['npil'] = allsynapses['npil'].str.replace(r'_(L|R)$', '', regex=True)

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
plt.savefig(_out_dir / "synaptic_type_npils_bign.svg")

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
    # 🟦 Blue family
    'FB': '#00bfbf', 'EB': '#00cccc', 'PB': '#0099cc', 'NO': '#0073b2',
    'AMMC': '#4040ff', 'FLA': '#3366ff', 'CAN': '#3366ff', 'PRW': '#3333ff',
    'SAD': '#3333ff', 'GNG': '#3333ff', 'AL': '#3399ff', 'LH': '#3366ff', 'BU': '#3333ff',

    # 🟧 Orange family
    'MB-CA': '#ff9966', 'MB-PED': '#ff9966', 'MB-VL': '#ff9966',
    'MB-VLCP': '#ff9966', 'MB-ML': '#ff9966',

    # 🟨 Yellow family
    'LAL': '#ffcc33', 'SLP': '#ffcc33', 'SIP': '#ffcc33', 'SMP': '#ffcc33',
    'CRE': '#ffcc33', 'IB': '#ffcc33', 'ATL': '#ffcc33',

    # 🟩 Green family
    'VES': '#33cc66', 'EPA': '#00cc66', 'GOR': '#00cc66', 'SPS': '#00cc66',
    'IPS': '#00cc66', 'AOTU': '#00cc66',

    # 🟦 Light blue family
    'AVLP': '#3399ff', 'PVLP': '#3399ff', 'PLP': '#3399ff', 'WED': '#3399ff',

    # 💜 Purple family
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
plt.savefig(_out_dir / "synaptic_type_npils2.svg")

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
plt.savefig(_out_dir / "synaptic_type_npils3.svg")

plt.show()
#%%
