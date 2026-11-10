# Fig 1C: the three Fig 1B cells split into axon and dendrite by MaxSI, with each compartment's share of canonical
# synapses (postsynaptic terminals on the dendrite, presynaptic terminals on the axon).
# The cells are the ones listed for Fig 1B-C in Table 2 of the paper: LC31b 720575940628463468, LC15
# 720575940614955304, and 720575940633095137 (DNp09 in the legend, its hemibrain type; FlyWire types it DNp71).
# Each skeleton is healed and its synapses attached as step 09 does it, then labelled from the cut node and axon side
# that step 09 chose (a cut is a node with one parent and one child; side 1 is its child and everything below it,
# side 2 the rest; synapses on the cut node belong to neither side). MaxSI has no linker, so the panel has two colours.
# The printed shares count the cell's synapses in the MaxSI synapse table, as the MaxSI neuron table's pre_A, post_A,
# pre_D and post_D do; a few synapses attached to the skeleton are not in that table. The script checks itself: the
# drawn labels must give back the cell's MaxSI SI (step 09's summary; synapses on the cut node count on neither side),
# and every synapse of the MaxSI synapse table must have the drawn label of its node. The soma (navis's, which is
# the skeleton's root in these three cells, as the script checks) is drawn as a black dot: everything on its side of
# the cut takes one colour, which is why a trunk can cross the other compartment's arbor.
# Reads the step 09 MaxSI batch summaries (MAXSI_LABEL_BATCH_DIR), MAXSI_SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR and
# the skeletons under SWC_DIR. Run after step 09 (MaxSI), with PIPELINE_RUN = "rebuild".
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (METHODS_DIR, OUTPUT_DIR, MAXSI_LABEL_BATCH_DIR, MAXSI_SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR,
                    SWC_DIR, PIPELINE_RUN)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
from methods_all import upload_swc, attach_synapses_princeton
from node_cut_si import get_descendants_from_node, safe_SI_calc

import pandas as pd
import numpy as np
import navis
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved

if PIPELINE_RUN != "rebuild":
    raise SystemExit('config.py must be on PIPELINE_RUN = "rebuild" (the MaxSI tables of step 09)')
_out_dir = OUTPUT_DIR / "fig1" / "fig1c_example_splits"
_out_dir.mkdir(parents=True, exist_ok=True)

#%% Settings
CELLS = [(720575940628463468, 'LC31b'), (720575940614955304, 'LC15'), (720575940633095137, 'DNp09')]
GREY_TITLE = {'DNp09'}   # not intrinsic to the brain; grey in Fig 1B, as the legend says
COMP_COLORS = {'axon': '#92278f', 'dendrite': '#fbb040', 'none': '#bbbbbb'}
SCALE_BAR_NM = 20_000
ids = [c for c, _ in CELLS]

#%% Step 09's MaxSI cut and SI, and the cells' synapses
summ = pd.concat([pd.read_feather(f, columns=['neuron', 'best_node_id', 'MaxSI_SI', 'axon_side'])
                  for f in sorted(Path(MAXSI_LABEL_BATCH_DIR).glob('summary_batch_*.ftr'))], ignore_index=True)
assert summ['neuron'].dtype == np.int64 and summ['neuron'].is_unique
summ = summ[summ['neuron'].isin(ids)].set_index('neuron')
assert len(summ) == len(ids), "a Fig 1C cell has no MaxSI cut in step 09's summaries"
syn = ds.dataset(str(SYNAPSE_TABLE_RAW_FTR), format='feather') \
        .to_table(filter=ds.field('pre').isin(ids) | ds.field('post').isin(ids)).to_pandas()
assert not (syn['pre'] == syn['post']).any(), "the raw synapse table has self-synapses"
msyn = ds.dataset(str(MAXSI_SYNAPSE_TABLE_FTR), format='feather') \
         .to_table(columns=['synapse_id', 'pre', 'post', 'comp'],
                   filter=ds.field('pre').isin(ids) | ds.field('post').isin(ids)).to_pandas()
assert msyn['pre'].dtype == np.int64 and msyn['post'].dtype == np.int64


def cut_labels(neuron, node_id, axon_side):
    nodes_ = neuron.nodes
    children = nodes_.loc[nodes_['parent_id'] == node_id, 'node_id'].tolist()
    assert len(children) == 1, f"cut node {node_id} has {len(children)} children"
    side_1 = get_descendants_from_node(neuron, children[0])
    lab = pd.Series('dendrite', index=nodes_['node_id'])
    lab[lab.index.isin(side_1)] = 'axon' if axon_side == 1 else 'dendrite'
    lab[~lab.index.isin(side_1)] = 'dendrite' if axon_side == 1 else 'axon'
    lab[node_id] = 'none'
    return lab


#%% Each skeleton, labelled by the MaxSI cut, checked against the tables
skel, labels, shares = {}, {}, {}
for nid, name in CELLS:
    copies = list(Path(SWC_DIR).glob(f"*/{nid}.swc"))
    assert len(copies) == 1, f"{nid}: {len(copies)} SWC files"
    healed = navis.heal_skeleton(upload_swc(nid))
    att = attach_synapses_princeton(healed, syn[(syn['pre'] == nid) | (syn['post'] == nid)])
    att = att[0] if isinstance(att, navis.NeuronList) else att
    r = summ.loc[nid]
    lab = cut_labels(att, int(r['best_node_id']), int(r['axon_side']))
    c = att.connectors.assign(comp=lambda d: d['node_id'].map(lab))
    n = {(comp, kind): int(((c['comp'] == comp) & (c['type'] == kind)).sum())
         for comp in ('axon', 'dendrite') for kind in ('pre', 'post')}
    si = safe_SI_calc(0, n[('axon', 'pre')], n[('axon', 'post')], n[('dendrite', 'pre')], n[('dendrite', 'post')])[0]
    print(f"{nid} {name}: SI from the labels {si:.6f}, step 09 {r['MaxSI_SI']:.6f}")
    assert abs(si - r['MaxSI_SI']) < 1e-9, f"{nid}: the drawn labels don't give step 09's MaxSI SI"
    # the MaxSI synapse table, synapse by synapse: this cell's side of each synapse against the drawn label
    s = msyn[(msyn['pre'] == nid) | (msyn['post'] == nid)].copy()
    s['kind'] = np.where(s['pre'] == nid, 'pre', 'post')
    s['table'] = np.where(s['pre'] == nid, s['comp'].str[0], s['comp'].str[1])
    drawn = c.assign(drawn=c['comp'].map({'axon': 'A', 'dendrite': 'D', 'none': 'N'}))[['synapse_id', 'drawn']]
    s = s.merge(drawn, on='synapse_id', how='left', validate='1:1')
    extra = len(c) - len(s)
    print(f"   {len(s):,} synapses in the MaxSI synapse table, {extra} more attached to the skeleton; drawn label "
          f"differs from the table for {int((s['drawn'] != s['table']).sum())}")
    assert s['drawn'].notna().all() and (s['drawn'] == s['table']).all(), f"{nid}: a drawn label differs from the table"
    tab = {(comp, kind): int(((s['table'] == comp[0].upper()) & (s['kind'] == kind)).sum())
           for comp in ('axon', 'dendrite') for kind in ('pre', 'post')}
    shares[nid] = {'dendrite': (tab[('dendrite', 'post')], tab[('dendrite', 'pre')] + tab[('dendrite', 'post')]),
                   'axon': (tab[('axon', 'pre')], tab[('axon', 'pre')] + tab[('axon', 'post')])}
    assert att.soma is not None and list(att.root) == [att.soma], f"{nid}: the soma is not the skeleton's root"
    skel[nid], labels[nid] = att, lab

#%% Numbers for the legend: each compartment's canonical share (MaxSI synapse table)
rows = []
for nid, name in CELLS:
    for comp in ('dendrite', 'axon'):
        k, tot = shares[nid][comp]
        rows.append({'cell': name, 'neuron': nid, 'compartment': comp, 'canonical': k, 'total': tot,
                     'percent': round(100 * k / tot, 1)})
        print(f"{name} {comp}: {k:,}/{tot:,} = {100 * k / tot:.1f}%")
pd.DataFrame(rows).to_csv(_out_dir / "fig1c_canonical_shares.csv", index=False)

#%% Figure: the three cells side by side, front view (x, -y), each at its own scale with its own 20 um bar.
# Each percentage sits next to its compartment, in its colour, by a fixed rule (no position per cell): on the side of
# the compartment that faces away from the other one (from the median position of each compartment's synapses), just
# outside the compartment's whole extent, level with its synapses' median. The 20 um bar is under each cell, at its left.
fig, axes = plt.subplots(1, len(CELLS), figsize=(5.6, 2.8))
fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.16, wspace=0.45)
for ax, (nid, name) in zip(axes, CELLS):
    nd = skel[nid].nodes.set_index('node_id')
    xy = np.c_[nd['x'].to_numpy(), -nd['y'].to_numpy()]
    pos = dict(zip(nd.index, map(tuple, xy)))
    has_parent = nd['parent_id'].to_numpy() >= 0
    child_ids = nd.index.to_numpy()[has_parent]
    segs = [(pos[ch], pos[p]) for ch, p in zip(child_ids, nd['parent_id'].to_numpy()[has_parent])]
    ax.add_collection(LineCollection(segs, colors=[COMP_COLORS[labels[nid][ch]] for ch in child_ids], linewidths=0.6))
    pad = 0.15 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect('equal', adjustable='datalim'); ax.axis('off')
    ax.text(0.5, 1.06, name, transform=ax.transAxes, ha='center', va='bottom',
            color='grey' if name in GREY_TITLE else 'black')
    cn = skel[nid].connectors.assign(comp=lambda d: d['node_id'].map(labels[nid]))
    syn_xy = {comp: np.c_[cn.loc[cn['comp'] == comp, 'x'], -cn.loc[cn['comp'] == comp, 'y']]
              for comp in ('axon', 'dendrite')}
    mid = {comp: np.median(p, axis=0) for comp, p in syn_xy.items()}
    comp_of = labels[nid].reindex(nd.index).to_numpy()
    for comp, other in (('axon', 'dendrite'), ('dendrite', 'axon')):
        lo, hi = xy[comp_of == comp].min(axis=0), xy[comp_of == comp].max(axis=0)
        v = mid[comp] - mid[other]
        k, tot = shares[nid][comp]
        if abs(v[0]) >= abs(v[1]):   # the compartments lie side by side: label beside the box
            at, off, ha, va = ((hi[0] if v[0] > 0 else lo[0]), mid[comp][1]), (4 if v[0] > 0 else -4, 0), \
                ('left' if v[0] > 0 else 'right'), 'center'
        else:                        # one above the other: label above or below the box
            at, off, ha, va = (mid[comp][0], (hi[1] if v[1] > 0 else lo[1])), (0, 4 if v[1] > 0 else -4), \
                'center', ('bottom' if v[1] > 0 else 'top')
        ax.annotate(f"{100 * k / tot:.1f}%", xy=at, xytext=off, textcoords='offset points', color=COMP_COLORS[comp],
                    ha=ha, va=va, annotation_clip=False)
    ax.apply_aspect()
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    ax.annotate('', xy=(x0, y0), xytext=(x0 + SCALE_BAR_NM, y0), annotation_clip=False,
                arrowprops=dict(arrowstyle='-', lw=1, color='black', shrinkA=0, shrinkB=0),
                textcoords='data')
    ax.annotate(f"{SCALE_BAR_NM // 1000} µm", xy=(x0, y0), xytext=(0, -3), textcoords='offset points', ha='left',
                va='top', annotation_clip=False)
    sx, sy = pos[skel[nid].soma]
    ax.plot(sx, sy, 'o', ms=3, color='black', zorder=3)
fig.text(0.45, 0.02, 'Axon', color=COMP_COLORS['axon'], ha='right', va='bottom')
fig.text(0.47, 0.02, 'Dendrite', color=COMP_COLORS['dendrite'], ha='left', va='bottom')
fig.text(0.58, 0.02, '\u25cf soma', color='black', ha='left', va='bottom')
fig.savefig(_out_dir / "fig1c_example_splits.svg")
