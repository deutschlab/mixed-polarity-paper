
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_NONP_FTR, NEURON_TABLE_FTR, NEURON_TABLE_NONP_FTR,
                    SI_UPDATED_FTR, OUTPUT_DIR, METHODS_DIR)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
from math import ceil

import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved
# C_RULE "si" (Supp 3-S2 C): the synapses and labels of Fig 3E (Princeton) and Supp 3-S2 B (Buhmann), as
# figures/fig3/fig3d_si_compartment_correct.py makes them with MIXED_RULE "si": synapses between cells of the
# four intrinsic super-classes, each cell with its own SI (Princeton: SI_UPDATED_FTR; Buhmann: its SI in
# NEURON_TABLE_NONP_FTR), a synapse with an end on a cell with SI below 0.1 counted as Mixed, _L/_R taken off
# the neuropil names and synapses outside every neuropil left out. Each neuropil's shares of AA, AD, DA, DD and
# Mixed are its bar heights in 3E and 3-S2 B. It also gives Pearson's r on the synapses whose two cells have
# SI >= 0.1 on both detectors (printed and written to pearson_r.csv).
# C_RULE "all_cells": every AA/AD/DA/DD synapse of the two synapse tables, with no cell or SI filter and no
# Mixed; written to a folder of its own.
C_RULE = "si"
if C_RULE not in ("si", "all_cells"):
    raise SystemExit(f'C_RULE must be "si" or "all_cells", not {C_RULE!r}')
SI_CUTOFF = 0.1
FOUR = ['AA', 'AD', 'DA', 'DD']
types = FOUR + ['Mixed'] if C_RULE == "si" else FOUR   # the main figures' order
_out_dir = OUTPUT_DIR / "fig3" / "si_x_npil_x_synapse_detection"
if C_RULE == "all_cells":
    _out_dir = OUTPUT_DIR / "fig3" / "si_x_npil_x_synapse_detection_all_cells"
_out_dir.mkdir(parents=True, exist_ok=True)
#%% the cells and their SI on each detector, as fig3d_si_compartment_correct.py reads them
cell_si = {}
if C_RULE == "si":
    nodesG = pd.read_feather(NEURON_TABLE_FTR).drop(columns=['SI'])
    all_SI_df = pd.read_feather(SI_UPDATED_FTR).rename(columns={'root_id': 'neuron'})
    nodesG = nodesG.merge(all_SI_df, on='neuron', how='left').dropna(subset='SI')
    nodes_buh = pd.read_feather(NEURON_TABLE_NONP_FTR).dropna(subset='SI')
    for name, nodes in (('princeton', nodesG), ('buhmann', nodes_buh)):
        nodes = nodes[nodes['super_class'].isin(['central', 'optic', 'visual_projection', 'visual_centrifugal'])]
        nodes = nodes.dropna(subset=['dend_correct', 'axon_correct', 'super_class', 'primary_type'])
        assert nodes['neuron'].dtype == np.int64 and nodes['neuron'].is_unique
        cell_si[name] = nodes.set_index('neuron')['SI']
        print(f"{name}: {len(nodes):,} intrinsic cells, {int((nodes['SI'] >= SI_CUTOFF).sum()):,} with SI >= {SI_CUTOFF}")
    # the cells with SI >= 0.1 on both detectors, for the check below
    polarized_both = np.intersect1d(cell_si['princeton'].index[cell_si['princeton'] >= SI_CUTOFF],
                                    cell_si['buhmann'].index[cell_si['buhmann'] >= SI_CUTOFF])
    print(f"cells with SI >= {SI_CUTOFF} on both detectors: {len(polarized_both):,}")
    del nodesG, nodes_buh, nodes
#%% each detector's synapses, labelled, counted per neuropil
counts = {}        # neuropil x type, all labelled synapses (the 3E / 3-S2 B bars)
counts_both = {}   # neuropil x type, synapses whose two cells are polarized on both detectors
for name, path in (('princeton', SYNAPSE_TABLE_FTR), ('buhmann', SYNAPSE_TABLE_NONP_FTR)):
    syn = pd.read_feather(path, columns=['pre', 'post', 'comp', 'npil'])
    assert syn['pre'].dtype == np.int64 and syn['post'].dtype == np.int64
    syn = syn[syn['comp'].isin(['AA', 'AD', 'DA', 'DD'])]
    if C_RULE == "si":
        si = cell_si[name]
        syn = syn[syn['pre'].isin(si.index) & syn['post'].isin(si.index)]
        mixed = (si.reindex(syn['pre']).to_numpy() < SI_CUTOFF) | (si.reindex(syn['post']).to_numpy() < SI_CUTOFF)
        syn['type'] = np.where(mixed, 'Mixed', syn['comp'])
        syn['npil'] = syn['npil'].str.replace(r'_(L|R)$', '', regex=True)
        no_npil = syn['npil'].isna() | (syn['npil'] == 'None')
        print(f"{name}: synapses outside every neuropil, left out: {int(no_npil.sum()):,}")
        syn = syn[~no_npil]
        both = syn['pre'].isin(polarized_both) & syn['post'].isin(polarized_both)
        counts_both[name] = syn[both].groupby(['npil', 'type']).size().unstack(fill_value=0)
    else:
        syn['type'] = syn['comp']
        if name == 'princeton':
            syn['npil'] = syn['npil'].str.replace(r'(_R|_L)$', '', regex=True)
    counts[name] = syn.groupby(['npil', 'type']).size().unstack(fill_value=0)
    print(f"{name}: {int(counts[name].to_numpy().sum()):,} synapses in {len(counts[name])} neuropils")
    print(counts[name].sum().to_dict())
    counts[name].to_csv(_out_dir / f"npil_counts_{name}.csv")
    del syn
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
# --- each neuropil's share of each type, in percent of its labelled synapses; Princeton (_pri) on x, Buhmann
# (_old) on y; neuropils with synapses on both detectors ---
shares = {}
for key, table in (('all', counts), ('polarized_on_both', counts_both)):
    if not table:
        continue
    parts = []
    for name, suffix in (('princeton', '_pri'), ('buhmann', '_old')):
        count_table = table[name].copy()
        count_table['total'] = count_table.sum(axis=1)
        percent_table = count_table.div(count_table['total'], axis=0) * 100
        parts.append(percent_table.add_suffix('_pct').add_suffix(suffix))
    shares[key] = parts[0].join(parts[1], how='inner').dropna(how='any')
cmp_df = shares['all']
cmp_df.to_csv(_out_dir / "npil_shares.csv")

# (optional) show which npils are missing from the color map
missing = sorted({idx for idx in cmp_df.index if str(idx) not in npil_colors})
if missing:
    print("[Unmapped npils] Add to npil_colors:", missing)

# --- Pearson r per type; under C_RULE "si" also on the synapses between cells polarized on both detectors ---
r_rows = []
for t in types:
    row = {'type': t, 'r': cmp_df[f'{t}_pct_pri'].corr(cmp_df[f'{t}_pct_old']), 'neuropils': len(cmp_df)}
    if C_RULE == "si" and t in FOUR:
        cmp_both = shares['polarized_on_both']
        row['r_cells_polarized_on_both'] = cmp_both[f'{t}_pct_pri'].corr(cmp_both[f'{t}_pct_old'])
        row['neuropils_polarized_on_both'] = len(cmp_both)
        row['synapses_polarized_on_both_princeton'] = int(counts_both['princeton'].to_numpy().sum())
        row['synapses_polarized_on_both_buhmann'] = int(counts_both['buhmann'].to_numpy().sum())
    r_rows.append(row)
r_table = pd.DataFrame(r_rows)
print(r_table.to_string(index=False))
r_table.to_csv(_out_dir / "pearson_r.csv", index=False)

# --- plotting ---
limits = {t: (0, 100) for t in types}
nrows, ncols = (2, 2) if len(types) == 4 else (2, 3)

fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 6), sharex=False, sharey=False)
axes = axes.ravel()
for ax in axes[len(types):]:
    ax.axis('off')

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
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Princeton (% of the neuropil's synapses)", size=6)
    ax.set_ylabel("Buhmann (% of the neuropil's synapses)", size=6)

plt.tight_layout()
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
fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 5), sharex=False, sharey=False)
axes = axes.ravel()
for ax in axes[len(types):]:
    ax.axis('off')

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
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Princeton (% of the neuropil's synapses)", size=8)
    ax.set_ylabel("Buhmann (% of the neuropil's synapses)", size=8)

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

plt.tight_layout(rect=[0,0,0.85,1])  # leave space for legend
plt.savefig(
    _out_dir / "SI_corrr_npils_numbers.svg",
    bbox_inches='tight'
)
plt.show()
