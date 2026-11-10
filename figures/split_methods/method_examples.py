# Split methods, example cells (new Supp 1-S2): six cells, each split four ways (SFC, MaxSI, MinFisherP, MaxPhi), and
# for each cell the SI of every distinct candidate cut.
# The cells are picked by fixed rules (settings below), not by eye; ties go to the lowest cell ID. The rules and the
# picked cells are written to example_cells.csv.
# The SFC split is redone here as step 01 does it (heal, attach the synapses, synapse flow centrality rerooted to the
# soma, flow threshold 1); the three cut methods use the cut node and axon side that step 09 chose for each cell. A
# cut is a node with one parent and one child; side 1 is its child and everything below it, side 2 the rest, and
# synapses on the cut node belong to neither side.
# The SI distribution reads step 08's table of every candidate node (ALT_SPLIT_BATCH_DIR). Nodes along an unbranched
# stretch without synapses give the same split, so the filled histogram counts each distinct set of side counts once
# (two separate cuts with the same counts are counted once too); the outline counts every node. SFC is not a single
# cut and its SI leaves the linker out, so it is not placed on the cut axis.
# Reads SI_COMPARISONS_FTR, NEURON_TABLE_SFC_FTR, SI_UPDATED_FTR, the step 09 batch summaries of the three cut methods,
# ALT_SPLIT_BATCH_DIR, SYNAPSE_TABLE_RAW_FTR and the skeletons under SWC_DIR. Run after steps 08 and 09 (all three
# methods), with PIPELINE_RUN = "rebuild" and SPLIT_METHOD = "sfc".
# The numbers a legend needs (the pool, its medians, how often the three cut methods agree) are printed.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (METHODS_DIR, OUTPUT_DIR, SI_COMPARISONS_FTR, NEURON_TABLE_SFC_FTR, ALT_SPLIT_BATCH_DIR,
                    MAXSI_LABEL_BATCH_DIR, MINFISHERP_LABEL_BATCH_DIR, MAXPHI_LABEL_BATCH_DIR, SYNAPSE_TABLE_RAW_FTR,
                    SI_UPDATED_FTR, SWC_DIR, PIPELINE_RUN, SPLIT_METHOD)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
from methods_all import upload_swc, attach_synapses_princeton
from flow_centrality import prepare_for_split
from node_cut_si import get_descendants_from_node, safe_SI_calc

import pandas as pd
import numpy as np
import navis
import networkx as nx
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved

if PIPELINE_RUN != "rebuild" or SPLIT_METHOD != "sfc":
    raise SystemExit('config.py must be on PIPELINE_RUN = "rebuild" and SPLIT_METHOD = "sfc"')
_out_dir = OUTPUT_DIR / "split_methods" / "method_examples"
_out_dir.mkdir(parents=True, exist_ok=True)

#%% Settings
INTRINSIC = ['central', 'optic', 'visual_projection', 'visual_centrifugal']
MIN_SYNAPSES = 100        # synapse ends attached to the skeleton (pre + post), from the MaxSI summary
AGREE_MIN_SI = 0.3        # rule "agree": MaxSI SI at least this ...
AGREE_MAX_DIFF = 0.02     # ... and the four methods' SIs within this of each other
MIXED_SI = 0.1            # rule "mixed": MaxSI SI below this
BINS = np.linspace(0, 1, 41)
COMP_COLORS = {'axon': '#92278f', 'dendrite': '#fbb040', 'linker': 'green', 'none': '#bbbbbb'}
METHODS = ['SFC', 'MaxSI', 'MinFisherP', 'MaxPhi']
CUTS = ['MaxSI', 'MinFisherP', 'MaxPhi']
MARKERS = {'MaxSI': '^', 'MinFisherP': 's', 'MaxPhi': 'D'}
SCALE_BAR_NM = 10_000

#%% Every cell's SI under the four methods, and the cut of the three cut methods
def read_summaries(folder, si_col, name):
    s = pd.concat([pd.read_feather(f, columns=['neuron', 'best_node_id', si_col, 'axon_side', 'n_attached_pre',
                                               'n_attached_post'])
                   for f in sorted(Path(folder).glob('summary_batch_*.ftr'))], ignore_index=True)
    assert s['neuron'].dtype == np.int64 and s['neuron'].is_unique
    return s.rename(columns={si_col: f'{name}_SI', 'best_node_id': f'{name}_node', 'axon_side': f'{name}_axon_side',
                             'n_attached_pre': f'{name}_pre', 'n_attached_post': f'{name}_post'})

cells = pd.read_feather(SI_COMPARISONS_FTR, columns=['neuron_id', 'SFC_SI']).rename(columns={'neuron_id': 'neuron'})
cells['neuron'] = cells['neuron'].astype(np.int64)
for folder, col, name in [(MAXSI_LABEL_BATCH_DIR, 'MaxSI_SI', 'MaxSI'),
                          (MINFISHERP_LABEL_BATCH_DIR, 'MinFisherP_SI', 'MinFisherP'),
                          (MAXPHI_LABEL_BATCH_DIR, 'MaxPhi_SI', 'MaxPhi')]:
    cells = cells.merge(read_summaries(folder, col, name), on='neuron', how='inner', validate='1:1')
meta = pd.read_feather(NEURON_TABLE_SFC_FTR, columns=['neuron', 'super_class', 'primary_type'])
meta['neuron'] = meta['neuron'].astype(np.int64)
cells = cells.merge(meta, on='neuron', how='inner', validate='1:1')
cells['n_synapses'] = cells['MaxSI_pre'] + cells['MaxSI_post']
pool = cells[cells['super_class'].isin(INTRINSIC) & (cells['n_synapses'] >= MIN_SYNAPSES)
             & cells[[f'{m}_SI' for m in METHODS]].notna().all(axis=1)].copy()
pool['diff'] = pool['MaxSI_SI'] - pool['SFC_SI']
pool['spread'] = pool[[f'{m}_SI' for m in METHODS]].max(axis=1) - pool[[f'{m}_SI' for m in METHODS]].min(axis=1)
pool = pool.sort_values('neuron').reset_index(drop=True)

#%% A central cell whose cut-SI histogram falls in two groups (rule "central, two groups" below). For each pool cell,
# from step 08's table of every candidate node: its distinct cuts (each distinct set of side counts once, as the
# histogram's grey bars), and on their sorted SIs the widest empty gap between two consecutive cuts (gap), the cuts
# above it (n_high) and the gap as a share of the cell's largest cut SI (rel).
side_cols = ['side_1_pre', 'side_1_post', 'side_2_pre', 'side_2_post']
gaps = []
for f in sorted(Path(ALT_SPLIT_BATCH_DIR).glob('nodes_batch_*.ftr')):
    t_ = ds.dataset(str(f), format='feather').to_table(
        columns=['neuron_id'] + side_cols + ['SI'], filter=ds.field('neuron_id').isin(pool['neuron'].tolist())).to_pandas()
    t_ = t_.dropna(subset=['SI']).drop_duplicates(['neuron_id'] + side_cols).sort_values(['neuron_id', 'SI'])
    t_['d'] = t_.groupby('neuron_id')['SI'].diff()
    t_['rank'] = t_.groupby('neuron_id').cumcount()
    n_cuts = t_.groupby('neuron_id').size()
    widest = t_.dropna(subset=['d']).sort_values(['neuron_id', 'd', 'rank'], ascending=[True, False, True]) \
               .drop_duplicates('neuron_id').set_index('neuron_id')
    gaps.append(pd.DataFrame({'gap': widest['d'], 'n_high': n_cuts.reindex(widest.index) - widest['rank'],
                              'si_max': t_.groupby('neuron_id')['SI'].max().reindex(widest.index)}))
gaps = pd.concat(gaps)
gaps.index = gaps.index.astype(np.int64)
pool = pool.merge(gaps, left_on='neuron', right_index=True, how='left', validate='1:1')
pool['rel'] = pool['gap'] / pool['si_max']
GAP_MIN, REL_MIN, NHIGH_MIN = pool['gap'].quantile(0.9), pool['rel'].quantile(0.9), pool['n_high'].quantile(0.25)
two_groups = (pool['gap'] >= GAP_MIN) & (pool['rel'] >= REL_MIN) & (pool['n_high'] >= NHIGH_MIN)
central = pool[(pool['super_class'] == 'central') & pool['primary_type'].notna()
               & (pool['primary_type'].astype(str) != '')]
SIZE_WIN = tuple(central['n_synapses'].quantile([0.25, 0.75]))
whole_type = central.assign(ok=two_groups).groupby('primary_type')['ok'].agg(['size', 'all'])
whole_type = set(whole_type[(whole_type['size'] >= 2) & whole_type['all']].index)
print(f"two groups: widest gap >= {GAP_MIN:.4f} (90th percentile), gap / largest cut SI >= {REL_MIN:.3f} (90th), "
      f">= {NHIGH_MIN:.0f} distinct cuts above it (25th); central cells of typical size {SIZE_WIN[0]:.0f} to "
      f"{SIZE_WIN[1]:.0f} synapse ends; types where every pool cell has two groups: {len(whole_type)}")

#%% Pick the six cells by rule (ties: lowest cell ID; a cell already picked is skipped)
picked = []
def take(rule, candidates, key):
    candidates = candidates[~candidates['neuron'].isin([p['neuron'] for p in picked])]
    row = candidates.assign(_k=key(candidates)).sort_values(['_k', 'neuron']).iloc[0]
    picked.append({**row.drop('_k').to_dict(), 'rule': rule})

agree = pool[(pool['MaxSI_SI'] >= AGREE_MIN_SI) & (pool['spread'] < AGREE_MAX_DIFF)]
take(f'all four within {AGREE_MAX_DIFF} and MaxSI SI >= {AGREE_MIN_SI}: nearest that set\'s median MaxSI SI',
     agree, lambda c: (c['MaxSI_SI'] - agree['MaxSI_SI'].median()).abs())
take('typical: MaxSI minus SFC SI nearest its median', pool, lambda c: (c['diff'] - pool['diff'].median()).abs())
bimodal = central[two_groups.reindex(central.index) & central['n_synapses'].between(*SIZE_WIN)
                  & central['primary_type'].isin(whole_type)]
take('central, two groups: the widest gap among central cells of typical size whose type has two groups in every '
     'cell', bimodal, lambda c: -c['gap'])
take('largest SFC minus MaxSI SI', pool, lambda c: c['diff'])
mixed = pool[pool['MaxSI_SI'] < MIXED_SI]
take(f'MaxSI SI < {MIXED_SI}: nearest that set\'s median MaxSI SI', mixed,
     lambda c: (c['MaxSI_SI'] - mixed['MaxSI_SI'].median()).abs())
kc = pool[pool['primary_type'].astype(str).str.startswith('KC')]
take('Kenyon cell: MaxSI minus SFC SI nearest the Kenyon cells\' median', kc,
     lambda c: (c['diff'] - kc['diff'].median()).abs())
chosen = pd.DataFrame(picked)
chosen['neuron'] = chosen['neuron'].astype(np.int64)
ids = chosen['neuron'].tolist()

#%% Numbers for the legend (all on the pool)
print(f"pool: {len(pool):,} intrinsic cells with all four SIs and >= {MIN_SYNAPSES} synapse ends; super-classes: "
      + ", ".join(f"{k} {v:.1%}" for k, v in pool['super_class'].value_counts(normalize=True).items()))
for name, s in [('agree', agree), ('mixed', mixed), ('Kenyon cells', kc)]:
    print(f"rule set {name}: {len(s):,} cells")
print("medians: " + ", ".join(f"{m} SI {pool[f'{m}_SI'].median():.4f}" for m in METHODS)
      + f"; MaxSI minus SFC {pool['diff'].median():.4f} (5th to 95th percentile {pool['diff'].quantile(0.05):.4f} to "
      f"{pool['diff'].quantile(0.95):.4f})")
print(f"SI < {MIXED_SI}: MaxSI {(pool['MaxSI_SI'] < MIXED_SI).mean():.1%}, SFC {(pool['SFC_SI'] < MIXED_SI).mean():.1%}; "
      f"SFC SI above MaxSI SI: {(pool['diff'] < 0).mean():.2%}")
same_all = (pool['MaxSI_node'] == pool['MinFisherP_node']) & (pool['MaxSI_node'] == pool['MaxPhi_node'])
print(f"the three cut methods choose the same node: {same_all.mean():.1%}; MaxSI and MinFisherP "
      f"{(pool['MaxSI_node'] == pool['MinFisherP_node']).mean():.1%}; MaxSI and MaxPhi "
      f"{(pool['MaxSI_node'] == pool['MaxPhi_node']).mean():.1%}")
for floor in (300, 500):   # what the two "largest" rules would pick with a higher synapse floor
    big = pool[pool['n_synapses'] >= floor]
    for name, key in [('largest MaxSI minus SFC', -big['diff']), ('largest SFC minus MaxSI', big['diff'])]:
        w = big.assign(_k=key).sort_values(['_k', 'neuron']).iloc[0]
        print(f"with >= {floor} synapses ({len(big):,} cells), {name}: {w['neuron']} {w['primary_type']} "
              f"({w['super_class']}, {int(w['n_synapses'])} synapses), SFC {w['SFC_SI']:.3f}, MaxSI {w['MaxSI_SI']:.3f}")
chosen['synapse_percentile'] = [round((pool['n_synapses'] < n).mean() * 100, 1) for n in chosen['n_synapses']]

#%% Every candidate cut of the six cells (step 08's node table)
cols = ['neuron_id', 'node_id', 'side_1_pre', 'side_1_post', 'side_2_pre', 'side_2_post', 'SI']
nodes = ds.dataset(sorted(str(f) for f in Path(ALT_SPLIT_BATCH_DIR).glob('nodes_batch_*.ftr')), format='feather') \
          .to_table(columns=cols, filter=ds.field('neuron_id').isin(ids)).to_pandas()
nodes['neuron_id'] = nodes['neuron_id'].astype(np.int64)
assert set(nodes['neuron_id']) == set(ids), "a chosen cell has no candidate nodes in step 08's table"
side_cols = ['side_1_pre', 'side_1_post', 'side_2_pre', 'side_2_post']
dist, n_nodes = {}, {}
for nid, g in nodes.groupby('neuron_id'):
    n_nodes[nid] = len(g)
    g = g.dropna(subset=['SI'])
    distinct = g.drop_duplicates(subset=side_cols)
    dist[nid] = (distinct['SI'].to_numpy(), g['SI'].to_numpy())
    best = chosen.loc[chosen['neuron'] == nid, 'MaxSI_SI'].iloc[0]
    assert abs(g['SI'].max() - best) < 1e-9, f"{nid}: step 08's highest SI differs from step 09's MaxSI SI"
chosen['n_candidate_nodes'] = chosen['neuron'].map(n_nodes)
chosen['n_distinct_side_counts'] = chosen['neuron'].map({k: len(v[0]) for k, v in dist.items()})

#%% The six skeletons, labelled four ways
syn = ds.dataset(str(SYNAPSE_TABLE_RAW_FTR), format='feather') \
        .to_table(filter=ds.field('pre').isin(ids) | ds.field('post').isin(ids)).to_pandas()
assert not (syn['pre'] == syn['post']).any(), "the raw synapse table has self-synapses"
si_step01 = pd.read_feather(SI_UPDATED_FTR).rename(columns={'root_id': 'neuron'})
si_step01['neuron'] = si_step01['neuron'].astype(np.int64)
assert si_step01['neuron'].is_unique
si_step01 = si_step01.set_index('neuron')['SI']


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


def si_from_labels(neuron, lab):
    c = neuron.connectors.assign(comp=lambda d: d['node_id'].map(lab))
    n = lambda comp, kind: int(((c['comp'] == comp) & (c['type'] == kind)).sum())
    return safe_SI_calc(0, n('axon', 'pre'), n('axon', 'post'), n('dendrite', 'pre'), n('dendrite', 'post'))[0]


skel, labels = {}, {}
for r in chosen.itertuples(index=False):
    nid = r.neuron
    copies = list(Path(SWC_DIR).glob(f"*/{nid}.swc"))
    assert len(copies) == 1, f"{nid}: {len(copies)} SWC files"
    healed = navis.heal_skeleton(upload_swc(nid))
    att = attach_synapses_princeton(healed, syn[(syn['pre'] == nid) | (syn['post'] == nid)])
    att = att[0] if isinstance(att, navis.NeuronList) else att
    skel[nid] = att
    sfc = navis.split_axon_dendrite(prepare_for_split(att), metric='synapse_flow_centrality', reroot_soma=True,
                                    flow_thresh=1)
    lab = pd.Series('none', index=att.nodes['node_id'])
    for piece in sfc:
        lab[lab.index.isin(piece.nodes['node_id'])] = piece.compartment
        edges = piece.nodes[piece.nodes['parent_id'].isin(piece.nodes['node_id'])]
        g = nx.Graph(); g.add_nodes_from(piece.nodes['node_id']); g.add_edges_from(zip(edges['node_id'], edges['parent_id']))
        print(f"{nid} SFC {piece.compartment}: {piece.n_nodes} nodes in {nx.number_connected_components(g)} piece(s)")
    labels[(nid, 'SFC')] = lab
    for m in CUTS:
        labels[(nid, m)] = cut_labels(att, getattr(r, f'{m}_node'), getattr(r, f'{m}_axon_side'))
    for m in METHODS:   # the drawn labels give back each method's SI (SFC: the linker left out)
        si = si_from_labels(att, labels[(nid, m)])
        print(f"{nid} {m}: SI from the labels {si:.6f}, table {getattr(r, f'{m}_SI'):.6f}"
              + (f", step 01 {si_step01.get(nid, np.nan):.6f}" if m == 'SFC' else ''))
        assert abs(si - getattr(r, f'{m}_SI')) < 1e-9, f"{nid} {m}: the drawn labels don't give the table's SI"
        if m == 'SFC':
            assert abs(si - si_step01[nid]) < 1e-9, f"{nid}: the SFC split here differs from SI_updated.ftr (step 01's SFC SI)"

#%% Figure: one row per cell (highest MaxSI SI first), four skeletons and the SI distribution
chosen = chosen.sort_values('MaxSI_SI', ascending=False).reset_index(drop=True)
fig, axes = plt.subplots(len(chosen), 5, figsize=(7.2, 1.45 * len(chosen)),
                         gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.25], 'wspace': 0.08, 'hspace': 0.45})
fig.subplots_adjust(left=0.1, right=0.92, top=0.93, bottom=0.05)
ymax = max(len(v[1]) for v in dist.values()) * 2   # above any bin of the every-node histogram
for i, r in chosen.iterrows():
    nid = r['neuron']
    nd = skel[nid].nodes.set_index('node_id')
    xy = np.c_[nd['x'].to_numpy(), -nd['y'].to_numpy()]
    pos = dict(zip(nd.index, map(tuple, xy)))
    has_parent = nd['parent_id'].to_numpy() >= 0
    child_ids = nd.index.to_numpy()[has_parent]
    segs = [(pos[c], pos[p]) for c, p in zip(child_ids, nd['parent_id'].to_numpy()[has_parent])]
    pad = 0.05 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    for j, m in enumerate(METHODS):
        ax = axes[i, j]
        lab = labels[(nid, m)]
        ax.add_collection(LineCollection(segs, colors=[COMP_COLORS[lab[c]] for c in child_ids], linewidths=0.5))
        if m in CUTS:
            cx, cy = pos[r[f'{m}_node']]
            ax.plot(cx, cy, MARKERS[m], ms=5, mfc='black', mec='white', mew=0.6, zorder=5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')
        ax.text(0.5, 1.0, f"SI {r[f'{m}_SI']:.2f}", transform=ax.transAxes, va='bottom', ha='center')
        if i == 0:
            ax.set_title(m, pad=12)
        if j == 0:   # a 10 um bar in every row (the rows are at different scales)
            ax.plot([xlim[0], xlim[0] + SCALE_BAR_NM], [ylim[0], ylim[0]], color='black', lw=1, clip_on=False)
            ax.text(xlim[0], ylim[0], f"{SCALE_BAR_NM // 1000} µm", va='top', ha='left')
    ax = axes[i, 4]
    distinct, every = dist[nid]
    ax.hist(distinct, bins=BINS, color='#999999', label='distinct side counts')
    ax.hist(every, bins=BINS, histtype='step', color='black', lw=0.6, label='every node')
    ax.set_yscale('log'); ax.set_xlim(0, 1); ax.set_ylim(0.7, ymax)
    ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(['1', '10', '100', '1000']); ax.minorticks_off()
    strip = blended_transform_factory(ax.transData, ax.transAxes)
    for k, m in enumerate(CUTS):   # one marker row per method, so markers at the same SI don't overlap
        ax.plot(r[f'{m}_SI'], 1.06 + 0.1 * k, MARKERS[m], ms=3.5, transform=strip, clip_on=False, mec='black',
                mfc='black', mew=0.6)
    ax.text(0.98, 0.9, f"{len(distinct):,}", transform=ax.transAxes, ha='right', va='top')
    ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
    ax.tick_params(length=2, pad=1)
    if i < len(chosen) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('SI of the cut')
    if i == len(chosen) // 2:
        ax.set_ylabel('Count (log)')
fig.canvas.draw()
for i, r in chosen.iterrows():   # row labels at a fixed x, centred on the row
    box = axes[i, 4].get_position()
    ctype = r['primary_type'] if isinstance(r['primary_type'], str) else 'no cell type'
    fig.text(0.02, (box.y0 + box.y1) / 2,
             f"{ctype}\n{r['super_class'].replace('_', ' ')}\n{int(r['n_synapses']):,} syn.",
             rotation=90, va='center', ha='center')
key = ([Patch(color=COMP_COLORS[k], label=k) for k in ['axon', 'dendrite']]
       + [Line2D([], [], ls='', marker=MARKERS[m], ms=4, mec='black', mfc='black', label=f'{m} cut') for m in CUTS])
handles, names = axes[0, 4].get_legend_handles_labels()
fig.legend(key + handles, [k.get_label() for k in key] + names, frameon=False, loc='lower center',
           bbox_to_anchor=(0.5, 0.97), ncol=8, handlelength=1, columnspacing=0.9)
fig.savefig(_out_dir / "method_examples.svg", bbox_inches='tight')
plt.close(fig)

out = chosen[['rule', 'neuron', 'primary_type', 'super_class', 'n_synapses', 'synapse_percentile', 'SFC_SI',
              'MaxSI_SI', 'MinFisherP_SI', 'MaxPhi_SI', 'MaxSI_node', 'MinFisherP_node', 'MaxPhi_node',
              'n_candidate_nodes', 'n_distinct_side_counts']]
out.to_csv(_out_dir / "example_cells.csv", index=False)
print(out.to_string(index=False))
