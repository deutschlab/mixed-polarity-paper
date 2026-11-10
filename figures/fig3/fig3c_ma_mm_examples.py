# Fig 3C: two example connections whose synapses count as Mixed. MA: a mixed cell (Dm12) onto the axon of a polar
# cell (L3). MM: the same Dm12 cell onto another mixed cell (Dm10). A cell with SI < 0.1 is mixed, and every synapse
# touching it counts as Mixed (Fig 3D). SI is the MaxSI SI.
# The three cells are picked by a fixed rule (clearly mixed: SI < 0.03 and a mixed type; clearly polar: SI >= 0.29 and
# a polar type; each cell within the middle half of its class's synapse counts; at least 10 synapses in the pair, and
# for MA at least 90% of them on the post cell's axon; the most synapses), applied in the checks repository.
# Each skeleton is healed and its synapses attached as step 09 does it. The polar cell is coloured by step 09's MaxSI
# cut (axon purple, dendrite orange; synapses on the cut node belong to neither side); a mixed cell is drawn black,
# with no axon and dendrite, and its own output (red) and input (cyan) synapses. The connecting synapses (from the
# MaxSI synapse table) are drawn large, at their postsynaptic position, with a zoomed inset. Front view (x, -y),
# both cells at their true positions.
# The script checks itself: each cell's labels give back its step 09 MaxSI SI, and the MA synapses are on the post
# cell's axon in the MaxSI synapse table.
# Reads the step 09 MaxSI batch summaries (MAXSI_LABEL_BATCH_DIR), MAXSI_SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR and the
# skeletons under SWC_DIR. Run after step 09 (MaxSI), with PIPELINE_RUN = "rebuild".
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
_out_dir = OUTPUT_DIR / "fig3" / "fig3c_ma_mm_examples"
_out_dir.mkdir(parents=True, exist_ok=True)

#%% Settings
MIXED_CELL = (720575940620507984, 'Dm12')
EXAMPLES = [('MA', (720575940652492278, 'L3')), ('MM', (720575940637669518, 'Dm10'))]
EXPECTED_SYNAPSES = {'MA': 46, 'MM': 26}   # the pick rule's counts
COLORS = {'axon': '#92278f', 'dendrite': '#fbb040', 'none': '#bbbbbb', 'mixed': 'black', 'mixed partner': '#7f7f7f',
          'pre': '#e41a1c', 'post': '#00b4d8', 'contact': '#ffd700'}
SCALE_BAR_NM = 20_000
ids = [MIXED_CELL[0]] + [c for _, (c, _) in EXAMPLES]

#%% Step 09's MaxSI cut and SI, and the cells' synapses
summ = pd.concat([pd.read_feather(f, columns=['neuron', 'best_node_id', 'MaxSI_SI', 'axon_side'])
                  for f in sorted(Path(MAXSI_LABEL_BATCH_DIR).glob('summary_batch_*.ftr'))], ignore_index=True)
assert summ['neuron'].dtype == np.int64 and summ['neuron'].is_unique
summ = summ[summ['neuron'].isin(ids)].set_index('neuron')
assert len(summ) == len(ids), "a Fig 3C cell has no MaxSI cut in step 09's summaries"
syn = ds.dataset(str(SYNAPSE_TABLE_RAW_FTR), format='feather') \
        .to_table(filter=ds.field('pre').isin(ids) | ds.field('post').isin(ids)).to_pandas()
assert not (syn['pre'] == syn['post']).any(), "the raw synapse table has self-synapses"
msyn = ds.dataset(str(MAXSI_SYNAPSE_TABLE_FTR), format='feather').to_table(
    columns=['synapse_id', 'pre', 'post', 'comp', 'post_x', 'post_y', 'post_z'],
    filter=(ds.field('pre') == MIXED_CELL[0]) & ds.field('post').isin([c for _, (c, _) in EXAMPLES])).to_pandas()
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


#%% Each skeleton, labelled by the MaxSI cut, checked against step 09
skel, labels = {}, {}
for nid in ids:
    copies = list(Path(SWC_DIR).glob(f"*/{nid}.swc"))
    assert len(copies) == 1, f"{nid}: {len(copies)} SWC files"
    healed = navis.heal_skeleton(upload_swc(nid))
    att = attach_synapses_princeton(healed, syn[(syn['pre'] == nid) | (syn['post'] == nid)])
    att = att[0] if isinstance(att, navis.NeuronList) else att
    r = summ.loc[nid]
    lab = cut_labels(att, int(r['best_node_id']), int(r['axon_side']))
    c = att.connectors.assign(comp=lambda d: d['node_id'].map(lab))
    k = lambda comp, kind: int(((c['comp'] == comp) & (c['type'] == kind)).sum())
    si = safe_SI_calc(0, k('axon', 'pre'), k('axon', 'post'), k('dendrite', 'pre'), k('dendrite', 'post'))[0]
    print(f"{nid}: SI from the labels {si:.6f}, step 09 {r['MaxSI_SI']:.6f}")
    assert abs(si - r['MaxSI_SI']) < 1e-9, f"{nid}: the labels don't give step 09's MaxSI SI"
    skel[nid], labels[nid] = att, lab
SI = summ['MaxSI_SI']
assert SI[MIXED_CELL[0]] < 0.1 and SI[EXAMPLES[1][1][0]] < 0.1 and SI[EXAMPLES[0][1][0]] >= 0.1

#%% The connecting synapses
for name, (post, ptype) in EXAMPLES:
    s = msyn[msyn['post'] == post]
    print(f"{name} {MIXED_CELL[1]} -> {ptype}: {len(s)} synapses"
          + (f", {int((s['comp'].str[1] == 'A').sum())} on the {ptype} axon" if name == 'MA' else ""))
    assert len(s) == EXPECTED_SYNAPSES[name], (name, len(s))
    if name == 'MA':
        assert (s['comp'].str[1] == 'A').all(), "an MA synapse is not on the post cell's axon"
pd.DataFrame([{'example': name, 'pre': MIXED_CELL[0], 'pre_type': MIXED_CELL[1], 'pre_MaxSI_SI': round(SI[MIXED_CELL[0]], 4),
               'post': post, 'post_type': ptype, 'post_MaxSI_SI': round(SI[post], 4),
               'synapses': int((msyn['post'] == post).sum())} for name, (post, ptype) in EXAMPLES]) \
    .to_csv(_out_dir / "fig3c_examples.csv", index=False)


#%% Figure: per example, an overview (both cells at their true positions, the same scale in both examples, cropped to
# the mixed cell's extent) and beside it a zoom on the contact. The two mixed cells are drawn black (Dm12, on top) and
# light grey (Dm10); the polar cell by compartment. The mixed cell's own output and input synapses are shown in the
# zooms only.
from matplotlib.lines import Line2D
CROP = 0.15          # the overviews show the mixed cell's extent plus this share on each side
GREY = '#a6a6a6'


def segments(nid):
    nd = skel[nid].nodes.set_index('node_id')
    xy = np.c_[nd['x'].to_numpy(), -nd['y'].to_numpy()]
    pos = dict(zip(nd.index, map(tuple, xy)))
    has_parent = nd['parent_id'].to_numpy() >= 0
    child_ids = nd.index.to_numpy()[has_parent]
    return [(pos[ch], pos[p]) for ch, p in zip(child_ids, nd['parent_id'].to_numpy()[has_parent])], child_ids, xy


def draw(ax, post, zoom):
    segs, child_ids, _ = segments(post)
    if SI[post] < 0.1:
        ax.add_collection(LineCollection(segs, colors=GREY, linewidths=1.2 if zoom else 0.9, rasterized=True))
    else:
        ax.add_collection(LineCollection(segs, colors=[COLORS[labels[post][ch]] for ch in child_ids],
                                         linewidths=1.2 if zoom else 0.6, rasterized=True))
    segs, _, _ = segments(MIXED_CELL[0])
    ax.add_collection(LineCollection(segs, colors='black', linewidths=0.6 if zoom else 0.4, rasterized=True))
    if zoom:   # the mixed cell's own synapses
        cn = skel[MIXED_CELL[0]].connectors
        for kind in ('pre', 'post'):
            p = cn[cn['type'] == kind]
            ax.scatter(p['x'], -p['y'], s=2, color=COLORS[kind], alpha=0.8, linewidths=0, rasterized=True)
    s = msyn[msyn['post'] == post]
    ax.scatter(s['post_x'], -s['post_y'], s=14 if zoom else 8, color=COLORS['contact'], edgecolors='black',
               linewidths=0.4, zorder=5)


_, _, dm12_xy = segments(MIXED_CELL[0])
pad = CROP * max(np.ptp(dm12_xy[:, 0]), np.ptp(dm12_xy[:, 1]))
x0, x1 = dm12_xy[:, 0].min() - pad, dm12_xy[:, 0].max() + pad
y0, y1 = dm12_xy[:, 1].min() - pad, dm12_xy[:, 1].max() + pad
fig = plt.figure(figsize=(6.0, 3.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.75, 1, 0.75], wspace=0.08, left=0.02, right=0.98, top=0.78, bottom=0.16)
for k, (name, (post, ptype)) in enumerate(EXAMPLES):
    ax, zx = fig.add_subplot(gs[0, 2 * k]), fig.add_subplot(gs[0, 2 * k + 1])
    draw(ax, post, zoom=False)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.axis('off')
    n_syn = int((msyn['post'] == post).sum())
    # the labels, above the overview and its zoom: the example, then each cell on its own line
    top = fig.add_subplot(gs[0, 2 * k:2 * k + 2]); top.axis('off'); top.set_zorder(-1)
    top.text(0.5, 1.2, f"{name}: {n_syn} synapses", transform=top.transAxes, ha='center', va='bottom')
    top.text(0.0, 1.11, f"{MIXED_CELL[1]} (presynaptic), SI {SI[MIXED_CELL[0]]:.3f}", transform=top.transAxes,
             ha='left', va='bottom', color='black')
    pcol = GREY if SI[post] < 0.1 else COLORS['axon']
    note = ""
    if SI[post] >= 0.1:   # the polar cell's dendrite runs out of the crop
        nd = skel[post].nodes.set_index('node_id')
        dend = labels[post].reindex(nd.index) == 'dendrite'
        if (nd.loc[dend, 'x'] > x1).any() or (nd.loc[dend, 'x'] < x0).any():
            note = "; its dendrite runs out of view"
    top.text(0.0, 1.02, f"{ptype} (postsynaptic), SI {SI[post]:.3f}{note}", transform=top.transAxes, ha='left',
             va='bottom', color=pcol)
    # the zoom
    s = msyn[msyn['post'] == post]
    cx, cy = s['post_x'].to_numpy(), -s['post_y'].to_numpy()
    m = 0.55 * max(np.ptp(cx), np.ptp(cy)) + 1500
    draw(zx, post, zoom=True)
    zx.set_xlim(cx.mean() - m, cx.mean() + m); zx.set_ylim(cy.mean() - m, cy.mean() + m)
    zx.set_aspect('equal'); zx.set_xticks([]); zx.set_yticks([])
    for sp in zx.spines.values():
        sp.set_linewidth(0.4); sp.set_edgecolor('0.5')
    ax.indicate_inset_zoom(zx, edgecolor='0.5', linewidth=0.4)
    zx.plot([cx.mean() - m * 0.9, cx.mean() - m * 0.9 + 2_000], [cy.mean() - m * 0.9] * 2, color='black', lw=1)
    zx.text(cx.mean() - m * 0.9, cy.mean() - m * 0.86, "2 \u00b5m", ha='left', va='bottom')
    if k == 0:   # one scale bar for both overviews (same scale)
        ax.plot([x0, x0 + SCALE_BAR_NM], [y0, y0], color='black', lw=1, clip_on=False)
        ax.annotate(f"{SCALE_BAR_NM // 1000} \u00b5m", xy=(x0, y0), xytext=(0, -3), textcoords='offset points',
                    ha='left', va='top', annotation_clip=False)
handles = [Line2D([], [], color='black', lw=1.5, label='Mixed cell (SI < 0.1)'),
           Line2D([], [], color=COLORS['axon'], lw=1.5, label='Axon'),
           Line2D([], [], color=COLORS['dendrite'], lw=1.5, label='Dendrite'),
           Line2D([], [], ls='', marker='o', ms=3, color=COLORS['pre'], label='Output synapse'),
           Line2D([], [], ls='', marker='o', ms=3, color=COLORS['post'], label='Input synapse'),
           Line2D([], [], ls='', marker='o', ms=5, color=COLORS['contact'], mec='black', mew=0.5,
                  label='Connecting synapse')]
fig.legend(handles=handles, ncol=3, loc='lower center', frameon=False, handlelength=1, columnspacing=0.8)
fig.savefig(_out_dir / "fig3c_ma_mm_examples.svg", dpi=600)
