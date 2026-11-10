# Split methods, new supplement figure (Dudi's revision plan, "1 - New S3"): how the four cut rules compare.
#   A: left/right agreement of SI for each cut rule. Pairs are the cell types with exactly two cells, one on each
#      side (the rule of Supp 2-S2 A and figures/split_methods/maxsi_x_primary_types_mirror.py). One scatter per
#      rule (left SI against right SI, coloured by super-class, with Pearson r as in Supp 2-S2 A) and one panel with
#      each rule's agreement, the one-way intraclass correlation ICC(1), and its 95% interval from resampling the
#      pairs (the same resamples for every rule, so the rules' differences have intervals too). The CSV adds Pearson,
#      Spearman, Lin's concordance, the median and mean |left - right|, and ICC(1) over every cell type with two or
#      more cells (a check that does not depend on the pair rule).
#   B: how alike the rules are over all cells. B1: Spearman between the rules' SI (Pearson in the CSV). B2: the share
#      of cells that two rules put on the same side of SI 0.1 (the paper's mixed cut-off); each rule's share of cells
#      below 0.1 is in its axis label. Printed and in mixed_call_agreement.csv: how many cells change from mixed to not
#      mixed between SFC and MaxSI, each way, and, by synapse-count decile, the share of cells whose MaxSI SI is more
#      than 0.1 above their MinFisherP SI.
#   C: the distribution of MaxSI SI minus SFC SI (delta SI). Cells where the two are equal (within 1e-9) are drawn as
#      a separate bar, so they don't hide in the bin next to zero.
# The four SIs are each rule's SI at its own cut, self-synapses removed: SI_UPDATED_SFC_FTR (SFC, steps 01-05),
# MAXSI_SI_UPDATED_FTR, MINFISHERP_SI_UPDATED_FTR and MAXPHI_SI_UPDATED_FTR (step 09 with each METHOD). They do not
# depend on SPLIT_METHOD. Cells: the four intrinsic super-classes with a primary_type and an axon and dendrite
# share (the standard filter, on the SFC neuron table), and an SI under all four rules.
# Run after steps 05 and 09 (all three methods). Panel A checks itself: the MaxSI pairs and R must match
# maxsi_x_primary_types_mirror.py, which reads the same SI from step 08's SI_COMPARISONS_FTR.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (METHODS_DIR, NEURON_ANNOTATIONS_CSV, NEURON_TABLE_SFC_FTR, OUTPUT_DIR, SI_COMPARISONS_FTR,
                    SI_UPDATED_SFC_FTR, MAXSI_SI_UPDATED_FTR, MINFISHERP_SI_UPDATED_FTR, MAXPHI_SI_UPDATED_FTR)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved

METHODS = ['SFC', 'MaxSI', 'MinFisherP', 'MaxPhi']
SI_TABLES = {'SFC': SI_UPDATED_SFC_FTR, 'MaxSI': MAXSI_SI_UPDATED_FTR, 'MinFisherP': MINFISHERP_SI_UPDATED_FTR,
             'MaxPhi': MAXPHI_SI_UPDATED_FTR}
INTRINSIC = ['central', 'optic', 'visual_projection', 'visual_centrifugal']
BOOT, SEED = 2000, 0
MIXED_SI = 0.1   # the paper's cut-off: a cell below it is mixed
sclass_palette = {'central': '#F9574E', 'optic': '#F4D826', 'visual_projection': '#D5A848',
                  'visual_centrifugal': '#44733B'}
_out_dir = OUTPUT_DIR / "split_methods" / "four_methods_si_comparison"
_out_dir.mkdir(parents=True, exist_ok=True)

#%% the cells and their four SIs
nodesG = pd.read_feather(NEURON_TABLE_SFC_FTR, columns=['neuron', 'super_class', 'primary_type', 'dend_correct',
                                                        'axon_correct', 'in_synapses_count', 'out_synapses_count'])
nodesG = nodesG[nodesG['super_class'].isin(INTRINSIC)]
nodesG = nodesG.dropna(subset=['dend_correct', 'axon_correct', 'super_class', 'primary_type'])
assert nodesG.neuron.dtype == np.int64 and nodesG.neuron.is_unique
n_filtered = len(nodesG)
for m in METHODS:
    t = pd.read_feather(SI_TABLES[m]).rename(columns={'root_id': 'neuron', 'SI': f'SI_{m}'})
    assert t.neuron.dtype == np.int64 and t.neuron.is_unique
    nodesG = nodesG.merge(t, on='neuron', how='left', validate='1:1')
cells = nodesG.dropna(subset=[f'SI_{m}' for m in METHODS]).copy()
print(f"cells: {n_filtered:,} intrinsic with a type and axon/dendrite shares; {len(cells):,} with an SI under all "
      f"four rules")

# the SFC and MaxSI SI must equal step 08's, computed separately
cmp = pd.read_feather(SI_COMPARISONS_FTR, columns=['neuron_id', 'SFC_SI', 'MaxSI_SI'])
chk = cells.merge(cmp, left_on='neuron', right_on='neuron_id', validate='1:1')
for m, col in [('SFC', 'SFC_SI'), ('MaxSI', 'MaxSI_SI')]:
    assert (chk[f'SI_{m}'] - chk[col]).abs().max() < 1e-9, f"{m} SI differs from step 08's"
print(f"SFC and MaxSI SI equal step 08's on all {len(chk):,} cells in both")

#%% A: left/right agreement, the Supp 2-S2 A pairs
side = pd.read_csv(NEURON_ANNOTATIONS_CSV, usecols=['root_id', 'side']).rename(columns={'root_id': 'neuron'})
assert side.neuron.dtype == np.int64 and side.neuron.is_unique
lr = cells.merge(side, on='neuron', how='left', validate='1:1')
lr = lr[lr['side'] != 'center']
lr['p_type_count'] = lr.groupby('primary_type')['primary_type'].transform('count')
lr = lr[lr['p_type_count'] == 2]
pairs = {m: lr.pivot_table(values=f'SI_{m}', index='primary_type', columns='side') for m in METHODS}
types = pairs['SFC'].dropna(subset=['left', 'right']).index
for m in METHODS:   # the same pairs under every rule
    assert pairs[m].loc[types, ['left', 'right']].notna().all().all()
sc = cells.drop_duplicates('primary_type').set_index('primary_type').super_class
left = np.column_stack([pairs[m].loc[types, 'left'].to_numpy() for m in METHODS])
right = np.column_stack([pairs[m].loc[types, 'right'].to_numpy() for m in METHODS])
print(f"A: {len(types):,} cell types with one left and one right cell")

def icc1(values, groups):
    """One-way intraclass correlation ICC(1) (ANOVA estimator; unequal group sizes allowed)."""
    df = pd.DataFrame({'v': values, 'g': groups})
    n_i = df.groupby('g').v.size()
    mean_i = df.groupby('g').v.mean()
    N, g = len(df), len(n_i)
    msb = (n_i * (mean_i - df.v.mean()) ** 2).sum() / (g - 1)
    msw = ((df.v - df.g.map(mean_i)) ** 2).sum() / (N - g)
    n0 = (N - (n_i ** 2).sum() / N) / (g - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def icc1_pairs(a, b):
    """ICC(1) for pairs (two per group): (MSB - MSW) / (MSB + MSW)."""
    m = (a + b) / 2
    msb = 2 * ((m - np.concatenate([a, b]).mean()) ** 2).sum() / (len(a) - 1)
    msw = ((a - m) ** 2 + (b - m) ** 2).sum() / len(a)
    return (msb - msw) / (msb + msw)


def ccc(a, b):
    """Lin's concordance correlation coefficient."""
    return 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


typed = cells.groupby('primary_type').primary_type.transform('size') >= 2   # every type with two or more cells
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(types), size=(BOOT, len(types)))
rows = []
for j, m in enumerate(METHODS):
    a, b = left[:, j], right[:, j]
    boot = np.array([icc1_pairs(a[i], b[i]) for i in idx])
    rows.append({'method': m, 'pairs': len(types), 'icc1': icc1_pairs(a, b), 'icc1_low': np.percentile(boot, 2.5),
                 'icc1_high': np.percentile(boot, 97.5), 'pearson_r': pearsonr(a, b)[0], 'spearman_rho': spearmanr(a, b)[0],
                 'lin_ccc': ccc(a, b), 'median_abs_left_minus_right': np.median(np.abs(a - b)),
                 'mean_abs_left_minus_right': np.mean(np.abs(a - b)),
                 'icc1_all_types': icc1(cells.loc[typed, f'SI_{m}'].to_numpy(), cells.loc[typed, 'primary_type'].to_numpy()),
                 '_boot': boot})
k = METHODS.index('MaxSI')
for row in rows:   # each rule's ICC minus MaxSI's, on the same resampled pairs
    d = row['_boot'] - rows[k]['_boot']
    row['icc1_minus_maxsi_low'], row['icc1_minus_maxsi_high'] = np.percentile(d, [2.5, 97.5])
lr_table = pd.DataFrame(rows).drop(columns='_boot')
print(f"   ICC over all types: {int(typed.sum()):,} cells in {cells.loc[typed, 'primary_type'].nunique():,} types "
      f"with two or more cells")
for _, row in lr_table.iterrows():
    print(f"   {row.method}: ICC(1) {row.icc1:.4f} [{row.icc1_low:.4f}, {row.icc1_high:.4f}], minus MaxSI's "
          f"[{row.icc1_minus_maxsi_low:.5f}, {row.icc1_minus_maxsi_high:.5f}]; Pearson {row.pearson_r:.4f}, Spearman "
          f"{row.spearman_rho:.4f}, Lin {row.lin_ccc:.4f}; |left - right| median {row.median_abs_left_minus_right:.4f}, "
          f"mean {row.mean_abs_left_minus_right:.4f}; ICC(1) over all types {row.icc1_all_types:.4f}")
assert len(types) == 3555 and round(lr_table.pearson_r[k], 2) == 0.94, \
    "MaxSI pairs or R differ from maxsi_x_primary_types_mirror.py (3,555 pairs, R = 0.94)"
lr_table.to_csv(_out_dir / "left_right_r.csv", index=False, float_format='%.6f')

fig, axes = plt.subplots(1, 4, figsize=(7, 2))
colors = sc.loc[types].map(sclass_palette).to_numpy()
for j, (m, ax) in enumerate(zip(METHODS, axes)):
    ax.scatter(left[:, j], right[:, j], s=1, c=colors, linewidths=0, rasterized=True)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_title(m)
    ax.text(0.05, 0.9, f"ICC = {lr_table.icc1[j]:.2f}", transform=ax.transAxes)   # Pearson r is in the CSV
    ax.set_xlabel('SI left')
    if j == 0:
        ax.set_ylabel('SI right')
    sns.despine(ax=ax)
axes[-1].legend(handles=[mpl.lines.Line2D([], [], marker='o', linestyle='', color=c, label=s, markersize=3)
                         for s, c in sclass_palette.items()], frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(_out_dir / "A1_left_right_scatter.svg", dpi=300)

plt.figure(figsize=(2.4, 2))
order = lr_table.sort_values('icc1').reset_index(drop=True)   # lowest ICC at the top, highest at the bottom
y = np.arange(len(order))
plt.errorbar(order.icc1, y, xerr=[order.icc1 - order.icc1_low, order.icc1_high - order.icc1],
             fmt='o', color='black', markersize=3, capsize=2, linewidth=0.8)
for yi, row in order.iterrows():   # each rule's ICC written to the right of its interval
    plt.text(row.icc1_high + 0.006, yi, f"{row.icc1:.3f}", va='center', ha='left')
plt.yticks(y, order.method)
plt.gca().invert_yaxis()
plt.xlim(right=order.icc1_high.max() + 0.045)
plt.xlabel('Left/right ICC (95%)')
sns.despine()
plt.tight_layout()
plt.savefig(_out_dir / "A2_left_right_icc.svg")

#%% B: the four rules against each other, over all cells
si = cells[[f'SI_{m}' for m in METHODS]].set_axis(METHODS, axis=1)
spear = si.corr(method='spearman')
pear = si.corr(method='pearson')
print(f"B: Spearman between rules over {len(si):,} cells:\n{spear.round(4).to_string()}")
pd.concat({'spearman': spear, 'pearson': pear}).to_csv(_out_dir / "rule_correlations.csv", float_format='%.6f')
mixed = si < MIXED_SI
same_side = pd.DataFrame({a: {b: (mixed[a] == mixed[b]).mean() for b in METHODS} for a in METHODS})
print(f"B2: share of cells below SI {MIXED_SI}: " + ", ".join(f"{m} {mixed[m].mean():.4f}" for m in METHODS))
print(f"    share of cells on the same side of {MIXED_SI}:\n{same_side.round(4).to_string()}")
to_seg, to_mixed = int((mixed.SFC & ~mixed.MaxSI).sum()), int((~mixed.SFC & mixed.MaxSI).sum())
print(f"    SFC mixed, MaxSI not: {to_seg:,} cells; MaxSI mixed, SFC not: {to_mixed:,} cells")
n_syn = cells.in_synapses_count + cells.out_synapses_count
decile = pd.qcut(n_syn.rank(method='first'), 10, labels=False)
gap = (si.MaxSI - si.MinFisherP > 0.1).groupby(decile.to_numpy()).mean()
print("    share of cells with MaxSI SI more than 0.1 above MinFisherP SI, by synapse-count decile (0 = fewest): "
      + ", ".join(f"{d} {v:.4f}" for d, v in gap.items()) + f"; decile 9 median {n_syn[decile == 9].median():,.0f} synapses")
pd.concat({'same_side_of_0.1': same_side, 'share_below_0.1': mixed.mean().to_frame('share').T,
           'sfc_maxsi_changes': pd.DataFrame({'SFC mixed, MaxSI not': [to_seg], 'MaxSI mixed, SFC not': [to_mixed]}),
           'maxsi_minus_minfisherp_over_0.1_by_decile': gap.to_frame('share').T}).to_csv(
    _out_dir / "mixed_call_agreement.csv", float_format='%.6f')
tick = {m: f"{m} ({100 * mixed[m].mean():.1f}%)" for m in METHODS}   # each rule's share of mixed cells
for name, table, label, lo in [('B1_rule_correlations', spear, 'Spearman rho', 0.9),
                               ('B2_same_mixed_call', same_side, f'Same side of SI {MIXED_SI}', 0.9)]:
    plt.figure(figsize=(4, 3.4))
    t = table.iloc[1:, :-1]
    if name.startswith('B2'):
        t = t.rename(index=tick, columns=tick)
    sns.heatmap(t, annot=True, fmt='.4f', cmap='Greys', vmin=lo, vmax=1, square=True,
                mask=np.triu(np.ones((3, 3), dtype=bool), k=1), cbar_kws={'label': label})   # each pair of rules once
    plt.tight_layout()
    plt.savefig(_out_dir / f"{name}.svg")

#%% C: MaxSI SI minus SFC SI
dsi = cells['SI_MaxSI'] - cells['SI_SFC']
below, at, above = int((dsi < -1e-9).sum()), int((dsi.abs() <= 1e-9).sum()), int((dsi > 1e-9).sum())
print(f"C: delta SI = MaxSI SI - SFC SI over {len(dsi):,} cells: below 0 {below:,} ({100 * below / len(dsi):.2f}%), "
      f"0 {at:,} ({100 * at / len(dsi):.2f}%), above 0 {above:,} ({100 * above / len(dsi):.2f}%); median "
      f"{dsi.median():.4f}, below -0.01 {int((dsi < -0.01).sum()):,}, lowest {dsi.min():.4f}, highest {dsi.max():.4f}")
pd.DataFrame({'neuron': cells.neuron, 'super_class': cells.super_class, 'SI_SFC': cells.SI_SFC,
              'SI_MaxSI': cells.SI_MaxSI, 'delta_SI': dsi}).to_csv(_out_dir / "delta_si.csv", index=False,
                                                                    float_format='%.6f')
plt.figure(figsize=(2.6, 2))
equal = dsi.abs() <= 1e-9
plt.hist(dsi[~equal], bins=np.arange(np.floor(dsi.min() * 100) / 100, dsi.max() + 0.01, 0.01), color='#6E6E6E')
plt.bar(0, at, width=0.003, color='black', zorder=3)   # the cells where the two SIs are equal, in front
plt.yscale('log')
plt.xlabel('MaxSI SI - SFC SI')
plt.ylabel('Cells')
plt.text(0.03, 0.62, f"< 0: {below:,}\n= 0: {at:,}\n> 0: {above:,}", transform=plt.gca().transAxes,
         ha='left', va='top')   # the empty area left of zero
sns.despine()
plt.tight_layout()
plt.savefig(_out_dir / "C_delta_si.svg")
