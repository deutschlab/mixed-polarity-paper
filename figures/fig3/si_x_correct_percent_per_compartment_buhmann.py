import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_NONP_FTR, SYNAPSE_TABLE_NONP_FTR, OUTPUT_DIR, METHODS_DIR
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
nodesG=pd.read_feather(NEURON_TABLE_NONP_FTR)

#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class'])

        #%%
#%%
df=nodesG[['neuron','super_class','SI','axon_correct','dend_correct']]
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
ax1.set_ylabel('Count')
ax1.legend(loc='upper left')

# --- Plot rolling correctness ---
ax2 = ax1.twinx()
ax2.plot(df_comb_g_axon['SI'], df_comb_g_axon['rolling_correct'], color='#92278f', lw=2, label='Axon rolling correct (%)')
ax2.plot(df_comb_g_dend['SI'], df_comb_g_dend['rolling_correct'], color='#fbb042', lw=2, label='Dendrite rolling correct (%)')

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
ax2.text(si_point + 0.003+0.72, axon_val, f'Axon: {axon_val:.2f}%', color='#92278f', va='bottom',size=15)
ax2.text(si_point + 0.003+0.72, dend_val-0.5, f'Dend: {dend_val:.2f}%', color='#fbb042', va='bottom',size=15)

# --- Titles and layout ---
plt.title('SI Distribution and Rolling Correctness (Axon vs Dendrite)')
plt.tight_layout()
plt.xlim(0,0.6)
_out_dir = OUTPUT_DIR / "fig3" / "si_x_correct_compartment_buhmann"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "axon_dend_correct_SI_buhman.svg")
plt.show()

print(f"Rolling correctness at SI = {si_point}:")
print(f"  Axon → {axon_val:.2f}%")
print(f"  Dendrite → {dend_val:.2f}%")

#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_NONP_FTR)
#%%
nodesG=nodesG[['neuron','SI','axon_correct','dend_correct']]
#%%
# Use the previously computed axon_val threshold from SI=0.1
threshold = axon_val  # from earlier step

# Create a new column based on comparison
nodesG['axon_label'] = nodesG['axon_correct'].apply(lambda x: 'A' if x*100 > threshold else 'M')

# Verify result
print(nodesG['axon_label'].value_counts())


# Use the previously computed axon_val threshold from SI=0.1
threshold = dend_val  # from earlier step

# Create a new column based on comparison
nodesG['dend_label'] = nodesG['dend_correct'].apply(lambda x: 'D' if x*100 > threshold else 'M')

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

allsynapses['comp_first']=allsynapses['comp'].str[0]
allsynapses['comp_second']=allsynapses['comp'].str[1]
#%%
aaa=allsynapses.head(1000)
#%%
allsynapses=allsynapses[['comp','dend_label_x',
'axon_label_x', 'neuron_y', 'dend_label_y', 'axon_label_y',
'comp_first', 'comp_second','npil']]
#%%
allsynapses['comp_first_corrected'] = allsynapses.apply(
    lambda row: (
        row['axon_label_x'] if row['comp_first'] == 'A'
        else row['dend_label_x'] if row['comp_first'] == 'D'
        else 'O'
    ),
    axis=1
)

#%%
allsynapses['comp_second_corrected'] = allsynapses.apply(
    lambda row: (
        row['axon_label_y'] if row['comp_second'] == 'A'
        else row['dend_label_y'] if row['comp_second'] == 'D'
        else 'O'
    ),
    axis=1
)

#%%
allsynapses2=allsynapses.head(10000)
#%%
allsynapses['comp2']=allsynapses['comp_first_corrected']+allsynapses['comp_second_corrected']
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
plt.savefig(_out_dir / "synaptic_type_pie_buhman.svg")

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
    'MD': '#fff2cc'    # very light pastel gold
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
plt.savefig(_out_dir / "synaptic_type_pie2_buhman.svg")

plt.show()


    #%%
allsynapses['npil'] = allsynapses['npil'].str.replace(r'_(L|R)$', '', regex=True)

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
plt.savefig(_out_dir / "synaptic_type_npils_buhman.svg")

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
plt.savefig(_out_dir / "synaptic_type_npils2_buhman.svg")

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
