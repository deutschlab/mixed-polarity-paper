# Two panels on the PC1 axes of Fig 4D, on the same population as the 4D model (syntype_x_pc1.py) before its
# balancing: synapses whose two cells have SI_pre and SI_post >= 0.1, a primary_type in the four intrinsic
# super-classes and a PC1 (PCA_TABLE_FTR), types AA, AD, DA and DD. SI_pre and SI_post are the synapse table's SI:
# on the published tables the older SI (self-synapses kept), and MaxSI_SI under SPLIT_METHOD = "maxsi", where PC1 is
# step 07's rerun on the MaxSI cells. Runs after processing/07_pca_morphology.py. The 5 optic cells with a
# primary_type but no PC1 are left out here (they can't be placed on the PC1 axes); 4D keeps their synapses.
#   B: the most common synapse type in each 0.25 x 0.25 square of PC1 pre x PC1 post (-2 to 5, squares with at least
#      200 synapses; synapses with a cell outside -2 to 5, about a third, are left out and their number printed).
#      4D draws the most over-represented type instead.
#   D: two-step AD chains a -> b -> c (c not a; each AD connection of at least one synapse counts once), split by
#      whether each step climbs PC1 (post PC1 above pre PC1; ties count as not climbing), as a share of all chains.
#      Against two nulls, 200 shuffles each: PC1 shuffled between cells of the same super-class, and PC1 shuffled
#      across all cells. With PC1 fully random, the 6 orders of three cells' PC1 are equally likely, so the second
#      null is 1/6, 1/3, 1/3, 1/6 on any network. The error bars of the real chains are a 95% range from a Poisson
#      bootstrap over the middle cells' primary types (1,000 draws); those of the nulls are the shuffles' 95% range.
#      Neither null keeps the link between PC1 and a cell's number of connections (PC1 is partly built from them).
# Each panel is written as an SVG with the numbers behind it as a CSV.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, PCA_TABLE_FTR, SYNAPSE_TABLE_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

INTRINSIC = ['central', 'optic', 'visual_projection', 'visual_centrifugal']
TYPES = ['AA', 'AD', 'DA', 'DD']
custom_palette = {'AA': '#8b2be2', 'AD': '#9F4800', 'DA': '#B3B3B3', 'DD': '#F4B95A'}
SHUFFLES, BOOT, SEED = 200, 1000, 0
_out_dir = OUTPUT_DIR / "fig4" / "ad_chains_x_pc1"
_out_dir.mkdir(parents=True, exist_ok=True)

#%% the population of the 4D model, before its balancing
cells = pd.read_feather(NEURON_TABLE_FTR, columns=['neuron', 'super_class', 'primary_type', 'size_nm'])
cells = cells.dropna(subset=['primary_type'])
cells = cells[cells.super_class.isin(INTRINSIC)].astype({'neuron': 'int64'})
pc1 = pd.read_feather(PCA_TABLE_FTR, columns=['neuron', 'PC1']).astype({'neuron': 'int64'})
cells = cells.merge(pc1, on='neuron', validate='1:1').dropna(subset=['PC1']).set_index('neuron')
assert cells.PC1.corr(cells.size_nm, method='spearman') > 0, "PC1's sign is flipped: high PC1 = smaller cell"

syn = pd.read_feather(SYNAPSE_TABLE_FTR, columns=['pre', 'post', 'comp', 'SI_pre', 'SI_post'])
syn = syn[(syn.SI_pre >= 0.1) & (syn.SI_post >= 0.1) & syn.comp.isin(TYPES)][['pre', 'post', 'comp']]
assert syn.pre.dtype == np.int64 and syn.post.dtype == np.int64
syn = syn[syn.pre.isin(cells.index) & syn.post.isin(cells.index)].reset_index(drop=True)
syn['comp'] = syn.comp.astype('category')
print(f"population: {len(syn):,} synapses between {len(cells):,} cells with a PC1")

conn = syn.groupby(['pre', 'post', 'comp'], observed=True).size().rename('n').reset_index()   # one row per connection and type
del syn
for side in ('pre', 'post'):
    conn[f'pc_{side}'] = conn[side].map(cells.PC1).values

#%% B: the most common type in each square of PC1 pre x PC1 post
LO, HI, STEP, MIN_SYN = -2.0, 5.0, 0.25, 200
n_sq = int(round((HI - LO) / STEP))
inside = conn.pc_pre.between(LO, HI, inclusive='left') & conn.pc_post.between(LO, HI, inclusive='left')
sq = conn[inside].assign(s_pre=((conn.pc_pre[inside] - LO) // STEP).astype(int).clip(upper=n_sq - 1),
                         s_post=((conn.pc_post[inside] - LO) // STEP).astype(int).clip(upper=n_sq - 1))
grid = sq.pivot_table(index=['s_pre', 's_post'], columns='comp', values='n', aggfunc='sum', fill_value=0,
                      observed=False).reindex(columns=TYPES, fill_value=0)
grid = grid[grid.sum(axis=1) >= MIN_SYN]
common = grid.idxmax(axis=1)
print(f"B: {len(grid)} of {n_sq * n_sq} squares, {grid.values.sum():,} synapses; most common type: "
      + "; ".join(f"{t} {(common == t).sum()}" for t in TYPES)
      + f"; {conn.n[~inside].sum():,} synapses have a cell outside {LO:g} to {HI:g}")

img = np.full((n_sq, n_sq), np.nan)
for (sp, spo), t in common.items():
    img[spo, sp] = TYPES.index(t)
cmap = mpl.colors.ListedColormap([custom_palette[t] for t in TYPES])
cmap.set_bad('white')
plt.figure(figsize=(2.6, 2.9))
plt.imshow(np.ma.masked_invalid(img), origin='lower', cmap=cmap, vmin=-0.5, vmax=3.5, extent=[LO, HI, LO, HI],
           interpolation='nearest')
plt.axvline(0.5, color='k', lw=0.5, ls='--')
plt.axhline(0.5, color='k', lw=0.5, ls='--')
plt.xticks(range(-2, 6), fontsize=7)
plt.yticks(range(-2, 6), fontsize=7)
plt.xlabel('PC1 pre neuron', fontsize=7)
plt.ylabel('PC1 post neuron', fontsize=7)
plt.title('most common', fontsize=7.5)
plt.legend(handles=[mpl.patches.Patch(color=custom_palette[t], label=t) for t in TYPES], fontsize=6, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
plt.tight_layout()
plt.savefig(_out_dir / "ad_chains_most_common_type.svg")
b = grid.copy()
b.index = pd.MultiIndex.from_arrays([LO + STEP * b.index.get_level_values(0), LO + STEP * b.index.get_level_values(1)],
                                    names=['pc1_pre_from', 'pc1_post_from'])
b.columns = [f'synapses_{c}' for c in b.columns]
b['most_common'] = common.values
b.reset_index().to_csv(_out_dir / "ad_chains_most_common_type.csv", index=False)

#%% D: two-step AD chains a -> b -> c, by whether each step climbs PC1
def chain_counts(edges, pc, recip):
    """Per middle cell, two-step chains a -> b -> c (c not a) by (step 1 climbs, step 2 climbs): an (n_cells, 4) array.
    recip holds the ordered AD edges (a, b) whose reverse (b, a) is also an AD edge; each is a walk a -> b -> a."""
    up = pc[edges[:, 1]] > pc[edges[:, 0]]
    n = len(pc)
    in_up = np.bincount(edges[up, 1], minlength=n)
    in_dn = np.bincount(edges[~up, 1], minlength=n)
    out_up = np.bincount(edges[up, 0], minlength=n)
    out_dn = np.bincount(edges[~up, 0], minlength=n)
    q = np.column_stack([in_up * out_up, in_up * out_dn, in_dn * out_up, in_dn * out_dn]).astype(float)
    a, b = recip[:, 0], recip[:, 1]
    s1, s2 = pc[b] > pc[a], pc[a] > pc[b]
    quad = np.where(s1, np.where(s2, 0, 1), np.where(s2, 2, 3))
    np.subtract.at(q, (b, quad), 1)
    return q


def chain_shares(edges, pc, recip):
    q = chain_counts(edges, pc, recip).sum(axis=0)
    return q / q.sum()


rng = np.random.default_rng(SEED)
LABELS = ['climb, climb', 'climb, not', 'not, climb', 'not, not']
codes = pd.Series(np.arange(len(cells)), index=cells.index)
pc_arr = cells.PC1.to_numpy()
groups = [np.flatnonzero((cells.super_class == s).to_numpy()) for s in cells.super_class.unique()]
type_codes = pd.factorize(cells.primary_type)[0]
ad = conn[conn.comp == 'AD']
edges = np.column_stack([codes[ad.pre].to_numpy(), codes[ad.post].to_numpy()])
assert (edges[:, 0] != edges[:, 1]).all(), "an AD connection from a cell to itself"
ek = pd.MultiIndex.from_arrays([edges[:, 0], edges[:, 1]])
recip = edges[ek.isin(pd.MultiIndex.from_arrays([edges[:, 1], edges[:, 0]]))]
per_mid = chain_counts(edges, pc_arr, recip)
obs = per_mid.sum(axis=0) / per_mid.sum()
indeg, outdeg = np.bincount(edges[:, 1], minlength=len(pc_arr)), np.bincount(edges[:, 0], minlength=len(pc_arr))
assert per_mid.sum() == (indeg * outdeg).sum() - len(recip)
by_type = np.zeros((type_codes.max() + 1, 4))
np.add.at(by_type, type_codes, per_mid)
boot = rng.poisson(1.0, size=(BOOT, len(by_type))) @ by_type
boot = boot / boot.sum(axis=1, keepdims=True)
null_sc, null_all = [], []
for _ in range(SHUFFLES):
    p = pc_arr.copy()
    for g in groups:
        p[g] = p[rng.permutation(g)]
    null_sc.append(chain_shares(edges, p, recip))
for _ in range(SHUFFLES):
    null_all.append(chain_shares(edges, pc_arr[rng.permutation(len(pc_arr))], recip))
null_sc, null_all = np.array(null_sc), np.array(null_all)
print(f"D: {len(edges):,} AD connections, {per_mid.sum():,.0f} two-step chains ({len(recip):,} walks a -> b -> a left out)")
for i, lab in enumerate(LABELS):
    print(f"   {lab}: {100 * obs[i]:.1f}% of chains; PC1 shuffled in super-class {100 * null_sc[:, i].mean():.1f}%; "
          f"across all cells {100 * null_all[:, i].mean():.1f}%")

x = np.arange(4)
err = np.abs(np.percentile(boot, [2.5, 97.5], axis=0) - obs)
err_sc = np.abs(np.percentile(null_sc, [2.5, 97.5], axis=0) - null_sc.mean(axis=0))
err_all = np.abs(np.percentile(null_all, [2.5, 97.5], axis=0) - null_all.mean(axis=0))
plt.figure(figsize=(3.2, 2.6))
plt.bar(x - 0.27, 100 * obs, 0.27, yerr=100 * err, capsize=2, color=custom_palette['AD'],
        label='AD chains (95%, cell types)')
plt.bar(x, 100 * null_sc.mean(axis=0), 0.27, yerr=100 * err_sc, capsize=2, color='#BBBBBB',
        label='PC1 shuffled in super-class')
plt.bar(x + 0.27, 100 * null_all.mean(axis=0), 0.27, yerr=100 * err_all, capsize=2, color='#6E6E6E',
        label='PC1 shuffled across all cells')
plt.xticks(x, ['climb,\nclimb', 'climb,\nnot', 'not,\nclimb', 'not,\nnot'], fontsize=6.5)
plt.yticks(fontsize=6.5)
plt.ylabel('% of two-step AD chains', fontsize=7)
plt.title('steps of AD -> AD chains', fontsize=7.5)
plt.legend(fontsize=6, frameon=False)
plt.tight_layout()
plt.savefig(_out_dir / "ad_chains_steps.svg")
pd.DataFrame({'steps': LABELS, 'chains': per_mid.sum(axis=0).astype(np.int64), 'chains_pct': 100 * obs,
              'chains_pct_low': 100 * np.percentile(boot, 2.5, axis=0),
              'chains_pct_high': 100 * np.percentile(boot, 97.5, axis=0),
              'shuffled_in_super_class_pct_mean': 100 * null_sc.mean(axis=0),
              'shuffled_in_super_class_pct_low': 100 * np.percentile(null_sc, 2.5, axis=0),
              'shuffled_in_super_class_pct_high': 100 * np.percentile(null_sc, 97.5, axis=0),
              'shuffled_all_cells_pct_mean': 100 * null_all.mean(axis=0),
              'shuffled_all_cells_pct_low': 100 * np.percentile(null_all, 2.5, axis=0),
              'shuffled_all_cells_pct_high': 100 * np.percentile(null_all, 97.5, axis=0)}).to_csv(
    _out_dir / "ad_chains_steps.csv", index=False, float_format='%.2f')
