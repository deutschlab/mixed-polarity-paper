# Fig 6 and its supplement: connection types within and between cell types in chosen circuits
# (visual projection LC neurons, antennal-lobe projection neurons, olfactory, Kenyon cells,
# central complex). For each population: the fraction of each type's output synapses that go to
# its own type, split by connection type (AA, DD, AD, DA), with the fraction of reciprocal
# same-type connections; the synapses per same-type connection; and directed graphs of the
# connections between types, one per connection type. A summary figure compares visual
# projection, ALPN and central complex.
# Ported from the corresponding author's MATLAB script FlyWire_AADD_specificCircuits.m (not
# included in this repository).
# Changes:
#   - paths via config.py; the inputs the MATLAB script read but never used afterwards (the
#     sensory ranks, the FlyWire colour table, the dsx/fru and auditory lists, and
#     connections_princeton783.csv) are not read;
#   - its connection table ex_table_npil_comp_pre_post.csv (synapses per pre, post, neuropil and
#     connection type) is not in the repository; it is rebuilt here from the synapse table chosen
#     by DATA_SOURCE (SYNAPSE_TABLE_NONP_FTR or SYNAPSE_TABLE_FTR) by counting synapses per pre,
#     post, npil and comp (all comp values kept, as the MATLAB script did not filter them);
#   - data: see DATA_SOURCE and SI_SOURCE below; the submitted panels come from the Buhmann
#     tables, which this repository does not build (obtain them from the authors and put them in
#     data/derived/); cell types always come from the chosen neuron table's primary_type (the
#     MATLAB script used Codex consolidated_cell_types783.csv, which is not in the repository);
#   - box plots use MATLAB's quartile rule (midpoint interpolation) and 1.5 IQR whiskers, outliers hidden;
#   - the summary plot's horizontal jitter is seeded (the MATLAB rand was not), so a run is
#     reproducible; the jitter only spreads the points;
#   - the between-type graphs use a layered layout built with networkx; MATLAB's 'layered'
#     layout places the nodes differently, the edges and their weights are the same;
#   - where the MATLAB script would stop with an error, the port carries on: a population with no
#     same-type connection is skipped, with no figures (with DATA_SOURCE='princeton' and the
#     older SI, the Kenyon cells, whose SIs are almost all below 0.1); its summary row still gives
#     its cells and different-type connections; such a population may not be one of the summary
#     plot's three (the script stops if it is). An empty between-type panel is left blank, and when
#     all edges of a panel have the same weight they are drawn at the middle width (3). The
#     spread of a single value counts as 0, as in MATLAB;
#   - the population summary is written as CSV (the MATLAB script wrote .xlsx; openpyxl is not
#     part of the environment), and the spoken "Done" at the end is left out.
# svg.fonttype='none' and Arial (repository convention).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (CLASSIFICATION_CSV, NEURON_TABLE_FTR, NEURON_TABLE_NONP_FTR, OUTPUT_DIR, SI_UPDATED_FTR,
                    SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_NONP_FTR)

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "fig6" / "aadd_specific_circuits"
_out_dir.mkdir(parents=True, exist_ok=True)

#%% parameters
SI_threshold = 0.1
# Which synapse detection the tables come from: 'buhmann' (NEURON_TABLE_NONP_FTR and
# SYNAPSE_TABLE_NONP_FTR, the tables the Buhmann comparison scripts read; not built by this
# repository) or 'princeton' (the tables of the rest of the paper). The submitted Fig 6 was made
# from the Buhmann tables: with 'buhmann' every Fig 6D box and whisker is reproduced exactly, and
# 6A-C and the per-type bars of Supp 6-S1/S2 come close but not exactly, because the Buhmann
# table's primary_type is not Codex consolidated_cell_types783 (it merges e.g. FS4A/B/C and has
# 77 ALPN types against the figure's 75); the Supp 6-S3 graphs have the same edges. With
# 'princeton' the numbers differ more (e.g. ALPN AA 0.43 instead of 0.59).
# With 'buhmann', SI and primary_type come from the Buhmann neuron table and SI_SOURCE is
# ignored.
DATA_SOURCE = 'buhmann'
# Which SI stands in for the SI column of the MATLAB script's neuron_data.csv:
# 'neuron_table' (NEURON_TABLE_FTR SI, the SI most figure scripts use) or 'SI_updated'.
SI_SOURCE = 'neuron_table'
assert DATA_SOURCE in ('buhmann', 'princeton'), DATA_SOURCE
assert SI_SOURCE in ('neuron_table', 'SI_updated'), SI_SOURCE
COMP_TYPES = ['AA', 'DD', 'AD', 'DA']
COMP_COLORS = [np.array(c) / 255 for c in ([120, 80, 161], [243, 185, 91], [159, 75, 35], [180, 180, 180])]
rng = np.random.default_rng(0)

#%% read the tables (cell IDs as int64 everywhere)
T_classification = pd.read_csv(CLASSIFICATION_CSV, usecols=['root_id', 'super_class', 'class'])
T_classification = T_classification.rename(columns={'root_id': 'cellID'})
T_classification['cellID'] = T_classification['cellID'].astype(np.int64)

neuron_table = NEURON_TABLE_NONP_FTR if DATA_SOURCE == 'buhmann' else NEURON_TABLE_FTR
synapse_table = SYNAPSE_TABLE_NONP_FTR if DATA_SOURCE == 'buhmann' else SYNAPSE_TABLE_FTR
for table in (neuron_table, synapse_table):
    if not table.exists():
        raise FileNotFoundError(f"{table} is missing. With DATA_SOURCE = 'buhmann' this script needs the Buhmann "
                                "tables, which this repository does not build (docs/data_availability.md); "
                                "set DATA_SOURCE = 'princeton' to run it on the paper's other tables.")
nodes = pd.read_feather(neuron_table, columns=['root_id', 'primary_type', 'SI'])
nodes['root_id'] = nodes['root_id'].astype(np.int64)
assert nodes['root_id'].is_unique
if DATA_SOURCE == 'princeton' and SI_SOURCE == 'SI_updated':
    si_upd = pd.read_feather(SI_UPDATED_FTR, columns=['root_id', 'SI'])
    si_upd['root_id'] = si_upd['root_id'].astype(np.int64)
    nodes = nodes.drop(columns='SI').merge(si_upd, on='root_id', how='left')
T_types = nodes[['root_id', 'primary_type']].dropna().rename(columns={'root_id': 'cellID'})
T_types = T_types[T_types['primary_type'].astype(str).str.strip() != '']   # MATLAB drops empty types too
T_SI = nodes[['root_id', 'SI']].dropna().rename(columns={'root_id': 'cellID'})

# connections per pre, post, neuropil and connection type (stands in for ex_table_npil_comp_pre_post.csv)
syn = feather.read_table(synapse_table, columns=['pre', 'post', 'npil', 'comp'])
T_conn = (syn.group_by(['pre', 'post', 'npil', 'comp']).aggregate([([], 'count_all')]).to_pandas()
          .rename(columns={'count_all': 'count'}).sort_values(['pre', 'post', 'npil', 'comp'], ignore_index=True))
del syn
T_conn['pre'] = T_conn['pre'].astype(np.int64)
T_conn['post'] = T_conn['post'].astype(np.int64)

#%% add cell type and SI to each connection; keep connections between polarized cells of known type
type_of = T_types.set_index('cellID')['primary_type']
si_of = T_SI.set_index('cellID')['SI']
T_conn['pre_Type'] = T_conn['pre'].map(type_of)
T_conn['post_Type'] = T_conn['post'].map(type_of)
T_conn['SI_pre'] = T_conn['pre'].map(si_of)
T_conn['SI_post'] = T_conn['post'].map(si_of)
T_conn = T_conn[(T_conn['SI_pre'] >= SI_threshold) & (T_conn['SI_post'] >= SI_threshold)]
T_conn = T_conn[T_conn['pre_Type'].notna() & T_conn['post_Type'].notna()].reset_index(drop=True)

# the classification table keeps only cells that have a type, with that type added
T_classification = T_classification.merge(T_types, on='cellID', how='inner')

#%% the populations
cls = T_classification
ptype = cls['primary_type'].astype(str)
C = [
    dict(population='Visual projection', SynThreshold=5,
         cells=cls.loc[(cls['super_class'] == 'visual_projection') & ptype.str.startswith('LC'), 'cellID']),
    dict(population='ALPN', SynThreshold=5, cells=cls.loc[cls['class'] == 'ALPN', 'cellID']),
    dict(population='olfactory', SynThreshold=5,
         cells=cls.loc[cls['class'].isin(['ALPN', 'Kenyon_Cell', 'MBON']), 'cellID']),
    dict(population='Kenyon cells', SynThreshold=1,
         cells=cls.loc[(cls['class'] == 'Kenyon_Cell') & (ptype.str.len() >= 2)
                       & ptype.str.lower().str.startswith('kc'), 'cellID']),
    dict(population='Central complex', SynThreshold=20, cells=cls.loc[cls['class'] == 'CX', 'cellID']),
]
PopulationsForSummary = [0, 1, 4]          # visual projection, ALPN, central complex
x_positions = [2, 1, 3.2]                  # their places on the summary plot's x axis


def box_stats(y):
    """Box statistics as MATLAB's boxchart/boxplot draw them: quartiles by midpoint
    interpolation, whiskers at the furthest data point within 1.5 IQR of the box, outliers hidden."""
    y = np.asarray(y, dtype=float)
    q1, med, q3 = np.percentile(y, [25, 50, 75], method='hazen')
    iqr = q3 - q1
    lo = y[y >= q1 - 1.5 * iqr].min()
    hi = y[y <= q3 + 1.5 * iqr].max()
    return dict(med=med, q1=q1, q3=q3, whislo=lo, whishi=hi, fliers=[])


def layered_positions(G):
    """Nodes in layers by longest path after removing cycles (a stand-in for MATLAB's 'layered')."""
    H = nx.DiGraph(G)
    while True:
        try:
            cycle = nx.find_cycle(H)
        except nx.NetworkXNoCycle:
            break
        H.remove_edge(*cycle[-1][:2])
    layer = {}
    for n in nx.topological_sort(H):
        preds = list(H.predecessors(n))
        layer[n] = 0 if not preds else 1 + max(layer[p] for p in preds)
    nx.set_node_attributes(G, layer, 'layer')
    pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal', scale=1)
    return {n: (x, -y) for n, (x, y) in pos.items()}   # first layer on top, as in MATLAB


#%% main loop: for each population, connection types, reciprocity and synapses per connection
fig_summary, ax_summary = plt.subplots(4, 2, figsize=(14, 16))
for a in ax_summary[:, 1]:
    a.axis('off')
x_sorted = sorted(x_positions)
summary_labels = [C[PopulationsForSummary[i]]['population'] for i in np.argsort(x_positions, kind='stable')]
nSummaryPlot = 0
summary_rows = []

for nPopulation, pop in enumerate(C):
    Population, SynThreshold = pop['population'], pop['SynThreshold']
    print('Analyzing population:', Population)

    # connections with both cells in the population
    cells = set(pop['cells'])
    P = T_conn[T_conn['pre'].isin(cells) & T_conn['post'].isin(cells)]

    # within type
    print('Within type')
    T_grouped = P.groupby(['pre_Type', 'post_Type', 'comp'], dropna=False)['count'].sum().reset_index()
    if (T_grouped['pre_Type'] == T_grouped['post_Type']).sum() == 0:
        # e.g. Kenyon cells: almost none has SI >= SI_threshold, so no same-type connection is left
        n_polarized = sum(1 for c in cells if si_of.get(c, -1) >= SI_threshold)
        print(f'  no connection between polarized cells of the same type in this population '
              f'({n_polarized} of {len(cells)} cells have SI >= {SI_threshold}); skipped')
        cells_si = pd.concat([P[['pre', 'SI_pre']].set_axis(['cellID', 'SI'], axis=1),
                              P[['post', 'SI_post']].set_axis(['cellID', 'SI'], axis=1)]).drop_duplicates('cellID')
        diff_sum = P[P['pre_Type'] != P['post_Type']].groupby(['pre', 'post'])['count'].sum()
        summary_rows.append(dict(Population=Population, meanSI=cells_si['SI'].mean(), Cells=len(cells_si),
                                 SynapsesPerConnection_SameType=np.nan,
                                 SynapsesPerConnection_DiffType=diff_sum.mean() if len(diff_sum) else np.nan))
        if nPopulation in PopulationsForSummary:
            raise ValueError(f'{Population} is in the summary plot but has no same-type connections')
        continue
    out_per_type = T_grouped.groupby('pre_Type')['count'].sum()
    same = P[P['pre_Type'] == P['post_Type']]
    rows = []
    for pre_Type_i, comp_i in T_grouped[['pre_Type', 'comp']].drop_duplicates().itertuples(index=False):
        X = T_grouped.loc[(T_grouped['pre_Type'] == pre_Type_i) & (T_grouped['post_Type'] == pre_Type_i)
                          & (T_grouped['comp'] == comp_i), 'count'].sum()
        Y = out_per_type[pre_Type_i]
        frac = np.nan if Y == 0 else X / Y
        # fraction of same-type connections that are reciprocal: 2 * reciprocal pairs / directed connections
        T = same.loc[same['pre_Type'] == pre_Type_i, ['pre', 'post']].drop_duplicates()
        directed = set(zip(T['pre'], T['post']))
        n_pairs = sum(1 for a, b in directed if a < b and (b, a) in directed)   # an autapse never pairs
        recip = 2 * n_pairs / len(T) if len(T) else np.nan
        n_pre = P.loc[P['pre_Type'] == pre_Type_i, 'pre'].nunique()
        n_pre_same = same.loc[same['pre_Type'] == pre_Type_i, 'pre'].nunique()
        rows.append(dict(pre_Type=pre_Type_i, comp=comp_i, FractionSynapses=frac, Pre_Cells_PerType=n_pre,
                         FractionPre_withLateralConnection=n_pre_same / n_pre,
                         FractionConnectionsReciprocal=recip))
    T_grouped = T_grouped.merge(pd.DataFrame(rows), on=['pre_Type', 'comp'], how='left')

    # only connections within type
    T_grouped_ = T_grouped[T_grouped['pre_Type'] == T_grouped['post_Type']]
    preTypes = sorted(T_grouped_['pre_Type'].unique())
    fractionMatrix = np.zeros((len(preTypes), len(COMP_TYPES)))
    Pre_Cells_PerType_, FractionConnectionsReciprocal_ = [], []
    for i, t in enumerate(preTypes):
        rows_t = T_grouped_[T_grouped_['pre_Type'] == t]
        for j, c in enumerate(COMP_TYPES):
            hit = rows_t[rows_t['comp'] == c]
            if len(hit):
                fractionMatrix[i, j] = hit['FractionSynapses'].iloc[0]
        Pre_Cells_PerType_.append(rows_t['Pre_Cells_PerType'].iloc[0])
        FractionConnectionsReciprocal_.append(rows_t['FractionConnectionsReciprocal'].iloc[0])

    # rows sorted by total bar length, longest first (stable for ties, as MATLAB's sort)
    sortIdx = np.argsort(-fractionMatrix.sum(axis=1), kind='stable')
    fractionMatrixSorted = fractionMatrix[sortIdx]
    preTypesSorted = [preTypes[i] for i in sortIdx]
    Pre_Cells_PerTypeSorted = np.array(Pre_Cells_PerType_)[sortIdx]
    FractionConnectionsReciprocalSorted = np.array(FractionConnectionsReciprocal_, dtype=float)[sortIdx]

    # figure: stacked bars per type, with the reciprocal fraction as red markers
    fig, ax1 = plt.subplots(figsize=(10, max(4, 0.3 * len(preTypesSorted) + 2)))
    y = np.arange(1, len(preTypesSorted) + 1)
    left = np.zeros(len(preTypesSorted))
    for j, c in enumerate(COMP_TYPES):
        ax1.barh(y, fractionMatrixSorted[:, j], left=left, height=0.8, color=COMP_COLORS[j], label=c)
        left += fractionMatrixSorted[:, j]
    ax1.invert_yaxis()
    ax1.set_yticks(y)
    ax1.set_yticklabels([str(s).replace('_', '-') for s in preTypesSorted])
    ax1.set_ylim(len(preTypesSorted) + 0.5, 0.5)
    ax1.set_xlabel('Barplot - fraction synapses within type')
    for side in ('top', 'right'):
        ax1.spines[side].set_visible(False)
    for yi, n in zip(y, Pre_Cells_PerTypeSorted):
        ax1.text(-0.14, yi, str(int(n)), ha='left', va='center', fontsize=12)
    for yi, bar_end, x_rec in zip(y, fractionMatrixSorted.sum(axis=1), FractionConnectionsReciprocalSorted):
        if x_rec > bar_end:
            ax1.plot([bar_end, x_rec], [yi, yi], '-', color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax1.plot(FractionConnectionsReciprocalSorted, y, 'o', markerfacecolor='r', markeredgecolor='k', linestyle='none')
    ax1.set_title(Population)
    ax1.legend(loc='best', frameon=False)
    xl = ax1.get_xlim()
    ax1.text((xl[1] - xl[0]) / 4, 1.03 * len(preTypesSorted), 'Fraction reciprocal', color='r', ha='center',
             va='top', fontsize=12)
    plt.savefig(_out_dir / f"{Population}_conn_SameType.svg", bbox_inches='tight')
    plt.show()

    # synapses per connection between cells of the same type (summed over neuropils and connection types)
    T_grouped_sameType = same.groupby(['pre', 'post'])['count'].sum().reset_index(name='sum_count')

    # summary plot (visual projection, ALPN, central complex)
    if nPopulation in PopulationsForSummary:
        x0 = x_positions[nSummaryPlot]
        nSummaryPlot += 1
        mean_frac = fractionMatrixSorted.mean(axis=0)

        ax = ax_summary[0, 0]
        ax.bar(x0, mean_frac.sum(), width=0.4, color='k')
        ax.set_ylim(0, 0.4)
        ax.set_ylabel('Fraction of output synapses\nthat are within-type')

        ax = ax_summary[1, 0]
        bottom = 0.0
        for j, c in enumerate(COMP_TYPES):
            share = mean_frac[j] / mean_frac.sum()
            ax.bar(x0, share, bottom=bottom, width=0.4, color=COMP_COLORS[j], label=c if nSummaryPlot == 1 else None)
            bottom += share
        ax.set_ylabel('Fraction of within-type connections')
        if nSummaryPlot == 1:
            ax.legend(frameon=False)

        ax = ax_summary[2, 0]
        yv = FractionConnectionsReciprocalSorted
        xv = x0 + 0.08 * (rng.random(len(yv)) - 0.5)
        ax.scatter(xv, yv, s=30, c=[(0.5, 0.5, 0.5, 0.3)], edgecolors=[(0, 0, 0, 0.5)], linewidths=0.3)
        m, se = yv.mean(), (yv.std(ddof=1) if len(yv) > 1 else 0.0) / np.sqrt(len(yv))
        ax.errorbar(x0, m, yerr=se, color='k', linewidth=2, capsize=15)
        ax.plot(x0, m, 'ko', markerfacecolor='k', markersize=8)
        ax.set_ylabel('Fraction of reciprocal connections')

        ax = ax_summary[3, 0]
        yb = T_grouped_sameType['sum_count'].to_numpy()
        ax.bxp([box_stats(yb)], positions=[x0], widths=0.4, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=(0.7, 0.7, 0.7), edgecolor='k', linewidth=2),
               whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2), medianprops=dict(color='none'))
        med = box_stats(yb)['med']
        ax.plot([x0 - 0.2, x0 + 0.2], [med, med], 'r', linewidth=3)
        bs = box_stats(yb)
        print(f"  summary {Population}: within-type fraction {mean_frac.sum():.3f}; composition "
              + ", ".join(f"{c} {mean_frac[j] / mean_frac.sum():.3f}" for j, c in enumerate(COMP_TYPES))
              + f"; reciprocal mean {m:.3f} +- {se:.3f} SEM over {len(yv)} types; synapses per same-type "
              f"connection median {bs['med']:g}, box {bs['q1']:g}-{bs['q3']:g}, whiskers {bs['whislo']:g}-{bs['whishi']:g}"
              f" ({len(yb)} connections)")
        sd = yb.std(ddof=1) if len(yb) > 1 else 0.0
        ax.text(x0, yb.mean() + 1.3 * sd, f"{med:g}", ha='center')
        ax.set_ylim(0, max(yb.mean() + 1.5 * sd, 1))
        ax.set_yticks(np.arange(0, 16, 5))
        ax.set_ylabel('Synapses per connection')

        for ax in ax_summary[:, 0]:
            ax.tick_params(labelsize=16)
            ax.yaxis.label.set_size(16)
            ax.set_xticks(x_sorted)
            ax.set_xticklabels(summary_labels)
            ax.set_xlim(0.5, max(x_positions) + 0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        if nSummaryPlot == len(PopulationsForSummary):
            fig_summary.savefig(_out_dir / "SummaryPlot.svg", bbox_inches='tight')

    # synapses per same-type connection (box without outliers, black line at the mean)
    fig, ax = plt.subplots(figsize=(4, 6))
    yb = T_grouped_sameType['sum_count'].to_numpy()
    ax.bxp([box_stats(yb)], positions=[1], widths=0.5, showfliers=False, boxprops=dict(linewidth=2),
           whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2), medianprops=dict(linewidth=2))
    ax.plot([0.75, 1.25], [yb.mean(), yb.mean()], 'k-', linewidth=2)
    ax.set_xlim(-0.5, 2.5)
    ax.get_xaxis().set_visible(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    top = int(np.ceil(ax.get_ylim()[1] * 1.1))
    ax.set_yticks(np.arange(0, top + 1, 1))
    ax.set_ylim(0, top * 1.1)
    ax.set_title(Population)
    ax.set_ylabel('#Synapses')
    plt.savefig(_out_dir / f"{Population}_SynapsesPerSameConnection.svg", bbox_inches='tight')
    plt.show()

    # between types
    print('Between types')
    diff = P[P['pre_Type'] != P['post_Type']]
    T_grouped_diffTypes = diff.groupby(['pre', 'post'])['count'].sum().reset_index(name='sum_count')
    # threshold on synapses (per pre, post, neuropil and connection type row), only to thin the plot
    diff_plot = diff[diff['count'] >= SynThreshold]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, comp in zip(axes.ravel(), ['AD', 'DA', 'AA', 'DD']):
        ax.axis('off')
        Tc = diff_plot[diff_plot['comp'] == comp]
        agg = Tc.groupby(['pre_Type', 'post_Type'])['count'].sum().reset_index()
        ax.set_title(f"{comp} (Syn threshold: {SynThreshold})", fontsize=14)
        if agg.empty:
            continue
        G = nx.DiGraph()
        for r in agg.itertuples(index=False):
            G.add_edge(str(r.pre_Type).replace('_', '-'), str(r.post_Type).replace('_', '-'), weight=r.count)
        w = np.array([G[u][v]['weight'] for u, v in G.edges()], dtype=float)
        widths = 1 + 4 * (w - w.min()) / (w.max() - w.min()) if w.max() > w.min() else np.full(len(w), 3.0)
        pos = layered_positions(G)
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=[(0, 0.447, 0.741)], arrowsize=12)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=20, node_color='k')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_family='Arial')
    plt.savefig(_out_dir / f"{Population}_conn_BetweenTypes.svg", bbox_inches='tight')
    plt.show()

    # mean SI and synapses per connection for the population
    cells_si = pd.concat([P[['pre', 'SI_pre']].set_axis(['cellID', 'SI'], axis=1),
                          P[['post', 'SI_post']].set_axis(['cellID', 'SI'], axis=1)]).drop_duplicates('cellID')
    summary_rows.append(dict(Population=Population, meanSI=cells_si['SI'].mean(), Cells=len(cells_si),
                             SynapsesPerConnection_SameType=T_grouped_sameType['sum_count'].mean(),
                             SynapsesPerConnection_DiffType=T_grouped_diffTypes['sum_count'].mean()))

T_PopulationSummary = pd.DataFrame(summary_rows)
T_PopulationSummary.to_csv(_out_dir / "PopulationSummary.csv", index=False)
print(T_PopulationSummary.to_string(index=False))

#%% connections between two given cell types, reciprocal ones first (interactive cell)
Pre_Type, Post_Type, Min_Syn = 'LC10', 'LC10', 1
T = T_conn[(T_conn['pre_Type'] == Pre_Type) & (T_conn['post_Type'] == Post_Type)]
if T.empty:
    print('No cells for this pre/post combination.')
else:
    T_summary = T.groupby(['pre', 'post'])['count'].sum().reset_index(name='sum_count')
    T_summary = T_summary[T_summary['sum_count'] >= Min_Syn]
    directed = set(zip(T_summary['pre'], T_summary['post']))
    T_summary['reciprocal'] = [int(a != b and (b, a) in directed) for a, b in zip(T_summary['pre'], T_summary['post'])]
    TT = T_summary.sort_values('reciprocal', ascending=False, kind='stable')
    print(TT.head(5).to_string(index=False))
    print('Number of connections (table height):', len(TT))
    print('Number of cells:', len(set(T_summary['pre']) | set(T_summary['post'])))
    print('Fraction reciprocal:', TT['reciprocal'].mean())
    plt.figure()
    plt.hist(TT['sum_count'], bins=np.arange(1, 23))
    plt.show()

#%% connections, with their types, between two given cells (interactive cell)
Cell_1, Cell_2 = 720575940619592768, 720575940611036643
print(T_conn[((T_conn['pre'] == Cell_1) & (T_conn['post'] == Cell_2))
             | ((T_conn['pre'] == Cell_2) & (T_conn['post'] == Cell_1))].to_string(index=False))
