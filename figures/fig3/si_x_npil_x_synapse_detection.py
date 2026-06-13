
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_NONP_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
from math import ceil

import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
#%%

syn_dfp=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
#%%

syn_dfp['npil'] = syn_dfp['npil'].str.replace(r'(_R|_L)$', '', regex=True)
#%%
# Keep only the four comp types
syn_dfp_filtered = syn_dfp[syn_dfp['comp'].isin(['AD','AA','DA','DD'])]

# --- Counts ---
count_table = syn_dfp_filtered.groupby(['npil','comp']).size().unstack(fill_value=0)

# --- Totals ---
count_table['total'] = count_table.sum(axis=1)

# --- Percentages ---
percent_table = count_table.div(count_table['total'], axis=0) * 100

# --- Combine (counts + percentages) ---
result = count_table.add_suffix('_count').join(
    percent_table.add_suffix('_pct')
)
#%%
del syn_dfp
#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_NONP_FTR)
#%%
# Keep only the four comp types
syn_df_filtered = syn_df[syn_df['comp'].isin(['AD','AA','DA','DD'])]

# --- Counts ---
count_table = syn_df_filtered.groupby(['npil','comp']).size().unstack(fill_value=0)

# --- Totals ---
count_table['total'] = count_table.sum(axis=1)

# --- Percentages ---
percent_table = count_table.div(count_table['total'], axis=0) * 100

# --- Combine (counts + percentages) ---
result_old = count_table.add_suffix('_count').join(
    percent_table.add_suffix('_pct'))
#%%
import re
import matplotlib.patheffects as pe

npil_colors = {
    'FB':'#12bfb2','EB':'#23c6c6','PB':'#0aa0c9','NO':'#0a76b2',
    'AMMC':'#4455ff','FLA':'#2f62ff','CAN':'#2f62ff','PRW':'#3333ff',
    'SAD':'#3333ff','GNG':'#3333ff','AL':'#3399ff','LH':'#3366ff','BU':'#3333ff',

    # MB family
    'MB_PED':'#ff9966','MB_VL':'#ff9966','MB_ML':'#ff9966','MB_CA':'#ff9966',

    # central yellow family
    'LAL':'#ffcc33','SLP':'#ffcc33','SIP':'#ffcc33','SMP':'#ffcc33','CRE':'#ffcc33',
    'IB':'#ffcc33','ATL':'#ffcc33',

    # green family
    'VES':'#33cc66','EPA':'#00cc66','GOR':'#00cc66','SPS':'#00cc66','IPS':'#00cc66','AOTU':'#00cc66',

    # light-blue family
    'AVLP':'#3399ff','PVLP':'#3399ff','PLP':'#3399ff','WED':'#3399ff',

    # purple family
    'ME':'#cc3399','AME':'#cc3399','LO':'#9933cc','LOP':'#9933cc','LA':'#9933cc','OCG':'#9933cc'
}

def npil_color(name: str) -> str:
    return npil_colors.get(str(name), '#444444')  # fallback grey if not found
#%%

# --- build comparison dataframe ---
r = result.copy()
ro = result_old.copy()
if 'npil' in r.columns:
    r = r.set_index('npil')
if 'npil' in ro.columns:
    ro = ro.set_index('npil')

pri = r.filter(regex=r'_pct$').add_suffix('_pri')
old = ro.filter(regex=r'_pct$').add_suffix('_old')
cmp_df = pri.join(old, how='inner').dropna(how='any')

# (optional) show which npils are missing from the color map
missing = sorted({idx for idx in cmp_df.index if str(idx) not in npil_colors})
if missing:
    print("[Unmapped npils] Add to npil_colors:", missing)

# --- plotting ---
limits = {'AD': (0, 90), 'AA': (0, 65), 'DA': (0, 20), 'DD': (0, 80)}
limits = {'AD': (0,100), 'AA': (0,100), 'DA': (0,100), 'DD': (0,100)}

types = ['AD','AA','DA','DD']

fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=False, sharey=False)
axes = axes.ravel()

for ax, t in zip(axes, types):
    x = cmp_df[f'{t}_pct_pri']
    y = cmp_df[f'{t}_pct_old']

    lo, hi = limits[t]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], '--', lw=0.6, color='gray')

    for npil, xv, yv in zip(cmp_df.index, x, y):
        ax.text(
            xv, yv, str(npil), fontsize=3, ha='center', va='center',
            color=npil_color(npil),
            path_effects=[pe.withStroke(linewidth=0.4, foreground='white')]
        )

    r_val = x.corr(y)
    ax.set_title(f'{t} (r={r_val:.2f})' if pd.notna(r_val) else t, fontsize=7)
    ax.grid(True, ls=':', lw=0.4)
    ax.set_xlabel('Princeton (%)', size=6)
    ax.set_ylabel('Buhamn (%)', size=6)

fig.suptitle('Percent by comp type per npil: Princeton (x) vs Old (y)', y=0.98, fontsize=10)
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig3" / "si_x_npil_x_synapse_detection"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    _out_dir / "SI_corrr_npils.svg",
    bbox_inches='tight'
)
plt.show()

#%%
# --- make numeric IDs for each npil (order = cmp_df.index) ---
# --- assign numbers to npils once ---
npil_to_id = {npil: i for i, npil in enumerate(cmp_df.index)}

# --- plotting ---
fig, axes = plt.subplots(2, 2, figsize=(5,5), sharex=False, sharey=False)
axes = axes.ravel()

for ax, t in zip(axes, types):
    x = cmp_df[f'{t}_pct_pri']
    y = cmp_df[f'{t}_pct_old']

    lo, hi = limits[t]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], '--', lw=0.6, color='gray')

    for npil, xv, yv in zip(cmp_df.index, x, y):
        ax.text(
            xv, yv, str(npil_to_id[npil]),   # numbers instead of npil names
            fontsize=2.5, ha='center', va='center',
            color=npil_color(npil),
            path_effects=[pe.withStroke(linewidth=0.4, foreground='white')]
        )

    r_val = x.corr(y)
    ax.set_title(f'{t} (r={r_val:.2f})' if pd.notna(r_val) else t, fontsize=7)
    ax.grid(True, ls=':', lw=0.4)
    ax.set_xlabel('Princeton (%)', size=8)
    ax.set_ylabel('Buhamn (%)', size=8)

# --- build legend entries: numbers + npil names, colored accordingly ---
handles = []
labels = []
for npil, idx in npil_to_id.items():
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=npil_color(npil),
                              markersize=5))
    labels.append(f"{idx}: {npil}")

fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
           fontsize=5, title="NPIL mapping")

fig.suptitle('Percent by comp type per npil: Princeton (x) vs Old (y)', y=0.98, fontsize=8)
plt.tight_layout(rect=[0,0,0.85,1])  # leave space for legend
plt.savefig(
    _out_dir / "SI_corrr_npils_numbers.svg",
    bbox_inches='tight'
)
plt.show()
